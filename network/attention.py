import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from config_file import *


class Identity_TransformerBlock(nn.Module):
    def __init__(self):
        super(Identity_TransformerBlock, self).__init__()

    def forward(self, Q, K, V, episilon=1e-8):
#        assert (Q == K and Q == V and K == V)
        return Q


class TransformerBlock(nn.Module):
    def __init__(self, input_size, is_layer_norm=False):
        super(TransformerBlock, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_norm1 = nn.LayerNorm(normalized_shape=input_size)
            self.layer_norm2 = nn.LayerNorm(normalized_shape=input_size)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)
        init.constant_(self.linear1.bias, 0.0)
        init.constant_(self.linear2.bias, 0.0)

        init.constant_(self.layer_norm1.weight, 1.)
        init.constant_(self.layer_norm1.bias, 0.)
        init.constant_(self.layer_norm2.weight, 1.)
        init.constant_(self.layer_norm2.bias, 0.)

    def FFN(self, X):
        return self.linear2(self.relu(self.linear1(X)))

    def forward(self, Q, K, V, attention_mask=None, episilon=1e-8):
        """
        :param Q: (batch_size, max_r_words, embedding_dim)
        :param K: (batch_size, max_u_words, embedding_dim)
        :param V: (batch_size, max_u_words, embedding_dim)
        :return: output: (batch_size, max_r_words, embedding_dim)  same size as Q
        """
        attention_mask = torch.zeros(size=(Q.size(0), Q.size(1), K.size(1))) if attention_mask is None else attention_mask
        attention_mask = attention_mask.to(self.args.device)

        dk = torch.Tensor([max(1.0, Q.size(-1))]).to(self.args.device)
        Q_K = Q.bmm(K.permute(0, 2, 1)) / (torch.sqrt(dk) + episilon)
        Q_K = Q_K + attention_mask    # mask some scores

        # (batch_size, max_r_words, max_u_words)
        Q_K_score = F.softmax(Q_K, dim=-1)
        V_att = Q_K_score.bmm(V)
        if self.is_layer_norm:
            # (batch_size, max_r_words, embedding_dim)
            X = self.layer_norm1(Q + V_att)
            output = self.layer_norm2(self.FFN(X) + X)
        else:
            X = Q + V_att
            output = self.FFN(X) + X
        return output


class AttentionBlock(nn.Module):
    "refer: DGMN codes provided by Zhao"
    def __init__(self, args):
        self.args = args
        super(AttentionBlock, self).__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=(args.emb_size))
        self.layernorm_ffn = nn.LayerNorm(normalized_shape=(args.emb_size))
        self.ffn = nn.Sequential(
            nn.Linear(args.emb_size, args.emb_size, bias=True),
            nn.ReLU(),
            nn.Linear(args.emb_size, args.emb_size, bias=True)
        )
        self.init_weight()

    def init_weight(self):
        init.constant_(self.layernorm.weight, 1.)
        init.constant_(self.layernorm.bias, 0.)
        init.constant_(self.layernorm_ffn.weight, 1.)
        init.constant_(self.layernorm_ffn.bias, 0.)
        init.xavier_uniform_(self.ffn[0].weight)
        init.xavier_uniform_(self.ffn[2].weight)

    def attention_dot(self, queries, keys, query_masks, key_masks, episilon=1e-8):
        """
        :param queries:
        :param keys:
        :param query_masks: (B, L_q)
        :param key_masks: (B, L_k) e.g. [[1,1,1,0],[1,1,1,1]]
        :param episilon:
        :return:
        """
        sim = torch.einsum('bik,bjk->bij', queries, keys)  # [B, L_q, L_k]
        scale = torch.Tensor([max(1.0, queries.size(-1))]).to(self.args.device)

        sim = sim / (torch.sqrt(scale) + episilon)

        # Key Masking
        masks = key_masks.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (B, L_q, L_k)
        paddings = (torch.ones_like(sim) * (-2 ** 32 + 1)).to(self.args.device)
        sim = torch.where(masks == 0, paddings, sim)  # (B, L_q, L_k)

        # Activation
        sim = torch.softmax(sim, dim=-1)

        # Query Masking
        sim = sim * query_masks.unsqueeze(-1)

        outputs = torch.einsum('bij,bjk->bik', sim, keys)
        return outputs

    def feedforward(self, inputs):
        outputs = self.ffn(inputs)
        outputs = outputs + inputs
        outputs = self.layernorm_ffn(outputs)
        return outputs

    def forward(self, queries, keys, query_masks, key_masks, residual=True, epsilon=1e-8):
        outputs = self.attention_dot(queries, keys, query_masks, key_masks, epsilon)
        if residual:
            outputs = self.layernorm(outputs + queries)
        else:
            outputs = self.layernorm(outputs)
        outputs = self.feedforward(outputs)
        return outputs


class NNSubmulti(nn.Module):
    def __init__(self, args):
        self.args = args
        super(NNSubmulti, self).__init__()
        self.linear_ff_sim = nn.Sequential(
            nn.Linear(in_features=args.emb_size * 2, out_features=100, bias=True),
            nn.Tanh(),
            nn.Linear(in_features=100, out_features=1, bias=False)
        )
        self.linear_last = nn.Linear(in_features=args.emb_size * 2, out_features=args.emb_size, bias=True)
        self.init_weight()

    def init_weight(self):
        init.xavier_uniform_(self.linear_ff_sim[0].weight)
        init.xavier_uniform_(self.linear_ff_sim[2].weight)
        init.xavier_uniform_(self.linear_last.weight)

    def ff_sim(self, queries, keys):
        T_q = queries.shape[1]
        T_k = keys.shape[1]
        expand_queries = queries.unsqueeze(2).repeat(1, 1, T_k, 1)
        expand_keys = keys.unsqueeze(1).repeat(1, T_q, 1, 1)

        # TODO: add a vector >> ref: source codes of Xueliang Zhao
        features = torch.cat([expand_queries, expand_keys], dim=-1)
        outputs = self.linear_ff_sim(features)

        outputs = outputs.view(-1, T_q, T_k)
        return outputs

    def attention_fc(self, queries, keys, query_masks, key_masks):
        sim = self.ff_sim(queries, keys)  # [B, L_q, L_k]

        # Key Masking
        masks = key_masks.unsqueeze(1).repeat(1, queries.shape[1], 1)  # (B, L_q, L_k)
        paddings = torch.ones_like(sim) * (-2 ** 32 + 1)
        sim = torch.where(masks == 0, paddings, sim)  # (B, L_q, L_k)

        # Activation
        sim = torch.softmax(sim, dim=-1)  # (B, L_q, L_k)

        # Query Masking
        sim = sim * query_masks.unsqueeze(-1)

        # Weighted sum
        outputs = torch.einsum('bij,bjk->bik', sim, keys)  # (B, T_q, C)
        return outputs

    def forward(self, queries, keys, query_masks, key_masks):
        keys_attn = self.attention_fc(keys, queries, key_masks, query_masks)    # TODO: check有没有反了
        feature_mul = keys_attn * keys
        feature_sub = (keys_attn - keys) * (keys_attn - keys)
        feature_last = torch.cat([feature_mul, feature_sub], dim=-1)
        feature_last = torch.relu(self.linear_last(feature_last))
        return feature_last


class HierarchicalNNSubmulti(nn.Module):
    def __init__(self, args):
        self.args = args
        super(HierarchicalNNSubmulti, self).__init__()
        self.linear_last = nn.Linear(in_features=args.emb_size * 2, out_features=args.emb_size, bias=True)
        self.init_weight()

    def init_weight(self):
        init.xavier_uniform_(self.linear_last.weight)

    def hierarchical_attention(self, queries, keys, query_masks, key_masks):
        L_q = queries.shape[1]
        N = keys.shape[1]
        sim1 = torch.einsum('bik,bnjk->binj', queries, keys)  # [B, L_q, N, L_k]

        # scale = torch.Tensor([max(1.0, queries.size(-1))]).to(self.args.device)
        # scale = torch.sqrt(scale)
        # sim1 = sim1 / scale

        masks = key_masks.unsqueeze(1).repeat(1, L_q, 1, 1)  # [B, L_q, N, L_k]
        paddings = torch.ones_like(sim1) * (-2 ** 32 + 1)
        sim1 = torch.where(masks == 0, paddings, sim1)  # [B, L_q, N, L_k]

        sim1 = torch.softmax(sim1, dim=-1)
        masks = query_masks.unsqueeze(2).repeat(1, 1, N)
        sim1 = sim1 * masks.unsqueeze(-1)  # [B, L_q, N, L_k]

        outputs1 = torch.einsum('binj,bnjk->bink', sim1, keys)
        sim2 = torch.einsum('bik,bink->bin', queries, outputs1)  # [B, L_k, N]

        # # Scale
        # scale = torch.Tensor([max(1.0, queries.size(-1))]).to(self.args.device)
        # scale = torch.sqrt(scale)
        # sim2 = sim2 / scale

        masks = torch.sign(torch.sum(key_masks, dim=-1))  # [B, N]
        masks = masks.unsqueeze(1).repeat(1, L_q, 1)  # [B, L_q, N]
        paddings = torch.ones_like(sim2) * (-2 ** 32 + 1)
        sim2 = torch.where(masks == 0, paddings, sim2)

        sim2 = torch.softmax(sim2, dim=-1)
        sim2 = sim2 * query_masks.unsqueeze(-1)

        outputs2 = torch.einsum('bin,bink->bik', sim2, outputs1)
        return outputs2

    def forward(self, queries, keys, query_masks, key_masks):
        keys_attn = self.hierarchical_attention(keys, queries, key_masks, query_masks)  # TODO: check有没有搞反
        feature_mul = keys_attn * keys
        feature_sub = (keys_attn - keys) * (keys_attn - keys)
        feature_last = torch.cat([feature_mul, feature_sub], dim=-1)
        feature_last = torch.relu(self.linear_last(feature_last))
        return feature_last


class FusionBlock(nn.Module):
    def __init__(self, input_size, is_layer_norm=False):
        super(FusionBlock, self).__init__()
        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_norm1 = nn.LayerNorm(normalized_shape=input_size)
            self.layer_norm2 = nn.LayerNorm(normalized_shape=input_size)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)
        init.constant_(self.linear1.bias, 0.0)
        init.constant_(self.linear2.bias, 0.0)

        init.constant_(self.layer_norm1.weight, 1.)
        init.constant_(self.layer_norm1.bias, 0.)
        init.constant_(self.layer_norm2.weight, 1.)
        init.constant_(self.layer_norm2.bias, 0.)

    def FFN(self, X):
        return self.linear2(self.relu(self.linear1(X)))

    def forward(self, Q, K, V, attention_mask=None, episilon=1e-8, output_score=False):
        """
        :param Q: (batch size, n_turn, max_u_words, embedding_dim)
        :param K: (batch size, n_doc, max_d_words, embedding_dim)
        :param V: (batch size, n_doc, max_d_words, embedding_dim)
        :param episilon:
        :return: output: (batch size, n_turn, n_doc, max_u_words, embedding_dim)
        """
        attention_mask = torch.zeros(size=(Q.size(0), Q.size(1), K.size(1), Q.size(2), K.size(2))) if attention_mask is None else attention_mask
        attention_mask = attention_mask.to(self.args.device)

        batch_size, n_turn, max_u_words, embedding_dim = Q.shape
        batch_size, n_doc, max_d_words, embedding_dim = K.shape

        dk = torch.Tensor([max(1.0, Q.size(-1))]).to(self.args.device)
        Q_K = torch.einsum('btue,bdpe->btdup', Q, K) / (torch.sqrt(dk) + episilon)
        Q_K = Q_K + attention_mask
        Q_K_score = F.softmax(Q_K, dim=-1)
        V_att = torch.einsum('btdup,bdpe->btdue', Q_K_score, V)

        Q_repeat = Q.view(batch_size, n_turn, 1, max_u_words, embedding_dim).repeat(1, 1, n_doc, 1, 1)
        X = Q_repeat + V_att
        if self.is_layer_norm:
            X = self.layer_norm1(X)
            output = self.layer_norm2(self.FFN(X) + X)
        else:
            output = self.FFN(X) + X

        if output_score:
            return output, Q_K_score
        else:
            return output


class MLP_Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP_Attention, self).__init__()
        self.linear_X = nn.Linear(input_size, hidden_size, bias=True)
        self.linear_ref = nn.Linear(input_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, out_features=1)

    def init_weight(self):
        init.xavier_normal_(self.linear_X.weight)
        init.xavier_normal_(self.linear_ref.weight)
        init.xavier_normal_(self.v.weight)
        init.constant_(self.linear1.bias, 0.0)
        init.constant_(self.linear2.bias, 0.0)
        init.constant_(self.v.bias, 0.0)

    def forward(self, X, ref):
        batch_size, n_X, _ = X.shape
        _, n_ref, _ = ref.shape

        stacking_X = self.linear_X(X).view(batch_size, n_X, 1, -1).repeat(1, 1, n_ref, 1)
        stacking_ref = self.linear_ref(ref).view(batch_size, 1, n_ref, -1).repeat(1, n_X, 1, 1)

        out = self.v(torch.tanh(stacking_X + stacking_ref)).squeeze()
        attention_scores = torch.softmax(out, dim=1)
        weighted_X = torch.einsum('bxe,bxr->bre', X, attention_scores)
        return weighted_X


if __name__ == '__main__':
    mlp_attention = MLP_Attention(300, 128)
    X = torch.rand(16, 25, 300)
    ref = torch.rand(16, 25, 300)
    out = mlp_attention(X, ref)
    print(out.shape)
