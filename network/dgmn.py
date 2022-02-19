from itertools import chain
import argparse
import torch
import torch.nn as nn
import torch.nn.init as init
from transformers import AutoModel

from retrieval_based.data_provider.utils import load_word_embedding
from retrieval_based.network import *
from retrieval_based.network.attention import *
from retrieval_based.network.layout import PretrainedLayoutEmbedding
from retrieval_based.network.skim_attention import *
from retrieval_based.network.basic import WordEmbeddingLayer
from config_file import *


class DGMNConv3DLayer(nn.Module):
    def __init__(self, args):
        self.args = args
        super(DGMNConv3DLayer, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=(1, 0, 0))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3), padding=(1, 0, 0))
        self.flatten = nn.Flatten()

    def forward(self, cube):
        outputs = self.pool1(torch.relu(self.conv1(cube)))
        outputs = self.pool2(torch.relu(self.conv2(outputs)))
        outputs = self.flatten(outputs)
        return outputs


class DGMN(nn.Module):
    def __doc__(self):
        DGMN_doc = """It is pytorch-version of DGMN, refering to the source code provided by first author of DGMN, Xueliang Zhao."""

    def __init__(self, args):
        self.args = args
        super(DGMN, self).__init__()
        self.embeddings = WordEmbeddingLayer(args, cmudog_glove_path)
        self.self_attention_block = AttentionBlock(args)

        self.DA_attention_block = AttentionBlock(args)
        self.CA_attention_block = AttentionBlock(args)

        self.qr_matching_nnsubmulti = NNSubmulti(args)
        self.car_matching_nnsubmulti = HierarchicalNNSubmulti(args)
        self.dar_matching_nnsubmulti = HierarchicalNNSubmulti(args)

        self.qr_matching_conv3 = DGMNConv3DLayer(args)
        self.car_matching_conv3 = DGMNConv3DLayer(args)
        self.dar_matching_conv3 = DGMNConv3DLayer(args)

        dim_dar = 16 * ((((args.max_doc_num + 2) // 3) + 2) // 3) * ((args.seq_len // 3) // 3) * ((args.emb_size // 3) // 3)
        dim_qr = 16 * ((((args.max_turn_num + 2) // 3) + 2) // 3) * ((args.seq_len // 3) // 3) * ((args.emb_size // 3) // 3)
        dim_car = dim_qr
        print(dim_dar, dim_qr, dim_car)
        self.classifier = nn.Linear(in_features=dim_qr + dim_car + dim_dar, out_features=1)

        self.debug_data = {'context': torch.randint(low=0, high=100, size=(16, 10, self.args.seq_len)),
                           'response': torch.randint(low=0, high=100, size=(16, self.args.seq_len)),
                           'doc_text': torch.randint(low=0, high=100, size=(16, self.args.doc_len + 2)),
                           'layout': torch.zeros(size=(16, self.args.doc_len + 2, 4), dtype=torch.long),
                           'n_turn': torch.randint(low=1, high=self.args.max_turn_num, size=(16, 1))}

    def forward(self, inputs):
        u, r, d, dl = inputs['context'], inputs['response'], inputs['document'], inputs['layout']

        batch_size, max_turn, max_u_words = u.shape
        _, max_r_words = r.shape
        _, max_sentence, max_d_words = d.shape
        # max_sentence = 1

        self.context_mask = torch.ones(batch_size, max_turn, max_u_words).to(DEVICE)
        self.document_mask = torch.ones(batch_size, max_sentence, max_d_words).to(DEVICE)
        self.response_mask = torch.ones(batch_size, max_r_words).to(DEVICE)
        self.parall_document_mask = self.document_mask.view(-1, max_d_words)
        self.parall_context_mask = self.context_mask.view(-1, max_u_words)

        context_embeddings = self.embeddings(u.view(-1, max_u_words)).view(batch_size, max_turn, max_u_words, -1)
        response_embeddings = self.embeddings(r.view(-1, max_r_words)).view(batch_size, max_r_words, -1)
        document_embeddings = self.embeddings(d.view(-1, max_d_words))
        document_embeddings = document_embeddings.view(batch_size, max_sentence, max_d_words, -1)

        response_rep = response_embeddings
        parall_context_rep = context_embeddings.view(-1, max_u_words, self.args.emb_size)
        parall_document_rep = document_embeddings.view(-1, max_d_words, self.args.emb_size)

        self.context_rep = self.self_attention_block(parall_context_rep, parall_context_rep,
                                                     self.parall_context_mask, self.parall_context_mask)
        self.response_rep = self.self_attention_block(response_rep, response_rep,
                                                      self.response_mask, self.response_mask)
        # [16, 4098, 300]
        self.document_rep = self.self_attention_block(parall_document_rep, parall_document_rep,
                                                      self.parall_document_mask, self.parall_document_mask)
        ################################ Grounding ###########################################
        # [B * D, l, d] -> [B, D, l, d] -> [B * T, D, l, d] -> [B * T * D, l, d]
        # B=16, D=1, T=10, l=4098, d=300
        document_rep = self.document_rep.view(batch_size, max_sentence, max_d_words, self.args.emb_size)
        document_rep = document_rep.unsqueeze(1).repeat(1, max_turn, 1, 1, 1)
        document_rep = document_rep.view(-1, max_sentence, max_d_words, self.args.emb_size)
        document_rep = document_rep.view(-1, max_d_words, self.args.emb_size)   # [160, 4098, 300]

        document_mask = self.document_mask.unsqueeze(1).repeat(1, max_turn, 1, 1)
        document_mask = document_mask.view(-1, max_sentence, max_d_words)
        document_mask = document_mask.view(-1, max_d_words)

        # [B * T, l, d] -> [B * T * D, l, d]
        # B=16, T=10, l=25, d=300
        context_rep = self.context_rep.unsqueeze(1).repeat(1, max_sentence, 1, 1)
        context_rep = context_rep.view(-1, max_u_words, self.args.emb_size)

        context_mask = self.parall_context_mask.unsqueeze(1).repeat(1, max_sentence, 1)
        context_mask = context_mask.view(-1, max_u_words)

        # [B * T * D, l, d]
        ## [16 * 10, 25, 300]
        c_a_d = self.DA_attention_block(context_rep, document_rep, context_mask, document_mask, residual=False)
        ## [16 * 10, 4098, 300]
        d_a_c = self.CA_attention_block(document_rep, context_rep, document_mask, context_mask, residual=True)

        #################################### Matching ################################3#
        # [B, l, d] -> [B * T, l, d]
        response_rep1 = self.response_rep.unsqueeze(1).repeat(1, max_turn, 1, 1)
        response_rep1 = response_rep1.view(-1, max_u_words, self.args.emb_size)
        response_mask1 = self.response_mask.unsqueeze(1).repeat(1, max_turn, 1)
        response_mask1 = response_mask1.view(-1, max_u_words)

        # Q-R
        q_r_rep = self.qr_matching_nnsubmulti(self.context_rep, response_rep1, self.parall_context_mask, response_mask1)
        q_r_rep = q_r_rep.view(-1, max_turn, max_r_words, self.args.emb_size)   # [16, 10, 25, 300]

        # CA-R
        # [B * T * D, l, d] -> [B * T, D, l, d]
        c_a_d = c_a_d.view(-1, max_sentence, max_u_words, self.args.emb_size)
        # [B * T, D, l, d] -> [B * T, D+1, l, d]
        c_a_d = torch.cat([c_a_d, self.context_rep.unsqueeze(1)], dim=1)

        # [B * T, l] -> [B * T, D, l]
        c_a_d_mask = self.parall_context_mask.unsqueeze(1).repeat(1, max_sentence, 1)
        document_mask = document_mask.view(-1, max_sentence, max_d_words)               # [B * T, D, l]
        document_mask = torch.sign(torch.sum(document_mask, dim=-1, keepdim=True))   # [B * T, D, 1]
        c_a_d_mask = c_a_d_mask * document_mask
        c_a_d_mask = torch.cat([c_a_d_mask, self.parall_context_mask.unsqueeze(1)], dim=1)

        q_r_rep2 = self.car_matching_nnsubmulti(c_a_d, response_rep1, c_a_d_mask, response_mask1)
        q_r_rep2 = q_r_rep2.view(-1, max_turn, max_r_words, self.args.emb_size)

        # DA-R
        # [B * T * D, l, d] -> [B * T, D, l, d] -> [B, T, D, l, d]
        d_a_c = d_a_c.view(-1, max_sentence, max_d_words, self.args.emb_size)
        d_a_c = d_a_c.view(-1, max_turn, max_sentence, max_d_words, self.args.emb_size)

        d_a_c_mask = self.document_mask.unsqueeze(1).repeat(1, max_turn, 1, 1)
        d_a_c_mask = d_a_c_mask.view(-1, max_sentence, max_d_words)
        d_a_c_mask = d_a_c_mask.view(-1, max_turn, max_sentence, max_d_words)

        # [B, T, D, l, d] -> [B, D, T, l, d] -> [B * D, T, l, d]
        d_a_c = d_a_c.permute(0, 2, 1, 3, 4).contiguous()
        d_a_c = d_a_c.view(-1, max_turn, max_d_words, self.args.emb_size)
        d_a_c_mask = d_a_c_mask.permute(0, 2, 1, 3).contiguous()
        d_a_c_mask = d_a_c_mask.view(-1, max_turn, max_d_words)

        # TODO: refine d_a_c_mask >> ref: source codes of Xueliang Zhao
        response_rep2 = self.response_rep.unsqueeze(1).repeat(1, max_sentence, 1, 1).view(-1, max_r_words, self.args.emb_size)
        response_mask2 = self.response_mask.unsqueeze(1).repeat(1, max_sentence, 1, 1).view(-1, max_r_words)

        d_r_rep = self.dar_matching_nnsubmulti(d_a_c, response_rep2, d_a_c_mask, response_mask2)
        d_r_rep = d_r_rep.view(-1, max_sentence, max_r_words, self.args.emb_size)

        qr_vec = self.qr_matching_conv3(torch.stack([q_r_rep], dim=1))
        car_vec = self.car_matching_conv3(torch.stack([q_r_rep2], dim=1))
        dar_vec = self.dar_matching_conv3(torch.stack([d_r_rep], dim=1))

        final_vec = torch.cat([qr_vec, car_vec, dar_vec], dim=-1)
        logits = self.classifier(final_vec)
        logits = torch.sigmoid(logits)
        logits = logits.squeeze()
        return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--gru_hidden", default=300, type=int, help="The hidden size of GRU in layer 1")
    parser.add_argument("--emb_size", default=300, type=int, help="The embedding size")
    parser.add_argument("--weight_decay", default=0, type=float, help="weight decay coefficient")
    parser.add_argument("--doc_len", default=40, type=int, help='Maximum #tokens/doc')
    parser.add_argument("--seq_len", default=40, type=int, help='Maximum #tokens/turn')
    parser.add_argument("--max_turn_num", default=4, type=int, help='Maximum #turn')
    parser.add_argument("--max_doc_num", default=20, type=int, help='Maximum #turn')
    parser.add_argument("--focusing_sample", default=0, type=int, help='Keep training n samples without testing.')
    parser.add_argument("--valid_every", default=100000, type=int)
    parser.add_argument("--test_every", default=100000, type=int)
    args = parser.parse_args()

    net = DGMN(args)
    g = net(net.debug_data)
    print(g)
