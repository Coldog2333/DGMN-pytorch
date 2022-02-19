import os
import numpy as np
from tqdm import tqdm
import argparse
import nltk
import torch
from torch.utils.data import Dataset, DataLoader

from data_provider.utils import GloveTokenizer
from config_file import *


class CMUDoGDataset(Dataset):
    def __init__(self, args, dial_path, glove_path, data_cache_path=None):
        self.args = args
        super(CMUDoGDataset, self).__init__()

        self.tokenizer = GloveTokenizer(glove_path=glove_path)

        # if True:
        if not os.path.exists(data_cache_path):
            self.contexts, self.candidate_lists, self.labels, self.document_lists, self.n_turns = [], [], [], [], []
            with open(dial_path) as f:
                for index, line in enumerate(f):
                    # if index > 100:
                    #     break
                    fields = line.strip().split('\t')
                    context = (fields[0] + ' ').split(' _eos_ ')[:-1]
                    candidate_list = fields[1].split('|')
                    label = int(fields[2])
                    document_list = fields[3].split('|')

                    self.contexts.append(context)
                    self.candidate_lists.append(candidate_list)
                    self.labels.append(label)
                    self.document_lists.append(document_list)
                    self.n_turns.append(len(context))

            self.w2v_preprocessing()
            self.samples = self.make_samples(self.contexts, self.candidate_lists, self.document_lists, self.labels, self.n_turns)
            cache = {'samples': self.samples}
            torch.save(cache, data_cache_path)
        else:
            cache = torch.load(data_cache_path)
            self.samples = cache['samples']

    def w2v_preprocessing(self):
        # tokenization & numericalization
        ## tokenize & numericalize dials and docs
        print('tokenzing & numericalizing dialogues and docs...')

        def tokenize_and_numericalize(sentence_lists):
            all_sentence_lists = []
            for i in tqdm(range(len(sentence_lists))):
                preprocessed_sentence_list = []
                for j in range(len(sentence_lists[i])):
                    tokens = self.tokenizer.tokenize(sentence_lists[i][j])
                    tokenized_utterance = self.tokenizer.convert_tokens_to_ids(tokens)
                    preprocessed_sentence_list.append(tokenized_utterance)
                all_sentence_lists.append(preprocessed_sentence_list)
            return all_sentence_lists

        self.contexts = tokenize_and_numericalize(self.contexts)
        self.candidate_lists = tokenize_and_numericalize(self.candidate_lists)
        self.document_lists = tokenize_and_numericalize(self.document_lists)

        # truncate and pad
        ## truncate & pad dials and docs
        ## dial
        for i in tqdm(range(len(self.contexts))):
            self.contexts[i] = self.contexts[i][-self.args.max_turn_num:]
            for j in range(len(self.contexts[i])):
                self.contexts[i][j] = self.contexts[i][j][-self.args.seq_len:]
                self.contexts[i][j] += [0] * (self.args.seq_len - len(self.contexts[i][j]))
            self.contexts[i] = [[0] * self.args.seq_len] * (self.args.max_turn_num - len(self.contexts[i])) + self.contexts[i]

        for i in tqdm(range(len(self.candidate_lists))):
            for j in range(len(self.candidate_lists[i])):
                self.candidate_lists[i][j] = self.candidate_lists[i][j][-self.args.seq_len:]
                self.candidate_lists[i][j] += [0] * (self.args.seq_len - len(self.candidate_lists[i][j]))

        ## docs
        print('truncating and padding documents...')
        for i in tqdm(range(len(self.document_lists))):
            self.document_lists[i] = self.document_lists[i][-self.args.max_doc_num:]
            for j in range(len(self.document_lists[i])):
                self.document_lists[i][j] = self.document_lists[i][j][-self.args.doc_len:]
                self.document_lists[i][j] += [0] * (self.args.doc_len - len(self.document_lists[i][j]))
            self.document_lists[i] = [[0] * self.args.doc_len] * (self.args.max_doc_num - len(self.document_lists[i])) + self.document_lists[i]

    def make_samples(self, contexts, candidate_lists, document_lists, sparse_labels, n_turns):
        samples = []
        for context, candidate_list, document_list, sparse_label, n_turn in zip(contexts, candidate_lists, document_lists, sparse_labels, n_turns):
            for index, candidate in enumerate(candidate_list):
                label = 1 if index == sparse_label else 0
                sample = {'context': context,
                          'response': candidate,
                          'label': label,
                          'document': document_list,
                          'n_turn': min(n_turn, self.args.max_turn_num)}
                samples.append(sample)
        return samples

    def __getitem__(self, index):
        sample = self.samples[index]

        context = torch.tensor(sample['context'], dtype=torch.long)
        response = torch.tensor(sample['response'], dtype=torch.long)
        document = torch.tensor(sample['document'], dtype=torch.long)
        n_turn = torch.tensor(sample['n_turn'], dtype=torch.int)
        label = torch.tensor(sample['label'], dtype=torch.float32)

        inputs = {'context': context,
                  'response': response,
                  'document': document,
                  'label': label,
                  'n_turn': n_turn}

        return inputs

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_len", default=40, type=int, help='Maximum #tokens/doc')
    parser.add_argument("--seq_len", default=40, type=int, help='Maximum #tokens/turn')
    parser.add_argument("--max_turn_num", default=4, type=int, help='Maximum #turn')
    parser.add_argument("--max_doc_num", default=20, type=int, help='Maximum #doc')
    args = parser.parse_args()

    dataset = CMUDoGDataset(args=args,
                            dial_path='../dataset/cmudog/processed_valid_self_original_fullSection.txt',
                            glove_path='../model/cmudog/glove_42B_300d.txt',
                            data_cache_path='temp.pkl')

    dataloader = DataLoader(dataset=dataset,
                            batch_size=40,
                            shuffle=False)

    for index, inputs in enumerate(dataloader):
        print(inputs)
        for key, value in inputs.items():
            print(key, ':', value.shape)
