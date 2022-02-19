import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from retrieval_based.data_provider.utils import load_word_embedding
from config_file import *


class WordEmbeddingLayer(nn.Module):
    def __init__(self, args, glove_path):
        self.args = args
        super(WordEmbeddingLayer, self).__init__()
        if args.use_bert:
            self.embeddings = AutoModel.from_pretrained(args.pretrained_model)
            self.ffn = nn.Linear(in_features=768, out_features=args.emb_size)
            print('Using %s' % self.embeddings.__class__.__name__)
        else:
            word_embeddings, word2id, id2word = load_word_embedding(glove_path)
            n_vocab, n_dim = word_embeddings.shape
            self.embeddings = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_dim)
            print('Using Glove embedding.')

    def forward(self, x):
        if self.args.use_bert:
            out = self.ffn(self.embeddings(x, return_dict=True)['last_hidden_state'])
        else:
            out = self.embeddings(x)
        return out