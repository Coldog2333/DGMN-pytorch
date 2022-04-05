import os
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import nltk


class GloveTokenizer():
    def __init__(self, glove_path):
        super(GloveTokenizer, self).__init__()
        _, self.word2id, self.id2word = load_word_embedding(glove_path)

    def tokenize(self, sentence):
        # tokens = sentence.strip().split(' ')
        tokens = nltk.word_tokenize(sentence.lower().strip(), language='english')
        return tokens

    def convert_tokens_to_ids(self, tokens):
        ids = [self.word2id[token] if token in self.word2id.keys() else 1 for token in tokens]  # <unk>: 1
        return ids

    def convert_ids_to_tokens(self, ids):
        tokens = [self.id2word[id] for id in ids]
        return tokens


def load_word_embedding(embedding_filename):
    """
    :param embedding_filename:
    :return:
    >>> word_embeddings, word2id, id2word = load_word_embedding(embedding_filename)
    """
    n_vocab = int(os.popen('wc -l %s' % embedding_filename).read().strip().split(' ')[0])
    n_dim = len(os.popen('head -1 %s' % embedding_filename).read().strip().split(' ')) - 1

    limit = np.sqrt(6. / (1 + n_dim))
    word_embeddings = [np.zeros(shape=n_dim),
                       np.random.uniform(low=-limit, high=limit, size=n_dim)]   # Xavier_initializer
    word2id, id2word = {'<pad>': 0, '<unk>': 1}, {0: '<pad>', 1: '<unk>'}
    with open(embedding_filename) as f:
        for line in f:
            items = line.strip().split(' ')
            word, embedding = items[0], np.array(items[1:], dtype=float)
            word_embeddings.append(embedding)
            word2id[word] = len(word2id)
            id2word[len(id2word)] = word

    word_embeddings = np.array(word_embeddings)

    return word_embeddings, word2id, id2word
