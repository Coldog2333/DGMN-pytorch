import logging
import os
import numpy as np
from tqdm import tqdm
import argparse
from pprint import pprint, pformat
import time
import logging
import nltk
import torch
from torch.utils.data import Dataset, DataLoader

from data_provider.utils import GloveTokenizer
from config_file import *


class CMUDoGDataset(Dataset):
    def __init__(self, args, dial_path, glove_path=None, char_path=None, data_cache_path=None):
        self.args = args
        super(CMUDoGDataset, self).__init__()

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if args.local_rank in [-1, 0] else logging.CRITICAL,
        )
        self.logger = logging.getLogger('data_provider')

        self.tokenizer = GloveTokenizer(glove_path=glove_path)

        # if True:
        if args.debug or not os.path.exists(data_cache_path):
            start = time.time()
            self.contexts, self.candidate_lists, self.sparse_labels, self.document_lists, self.n_turns, self.n_documents = [], [], [], [], [], []
            self.statistics = {'min_#token/context': 99999, 'aver_#token/context': 0, 'max_#token/context': 0,
                               'min_#token/document': 99999, 'aver_#token/document': 0, 'max_#token/document': 0,
                               'min_#token/response': 99999, 'aver_#token/response': 0, 'max_#token/response': 0}
            with open(dial_path) as f:
                for index, line in enumerate(f):
                    if args.debug and index > 100:
                        break
                    fields = line.strip().split('\t')
                    context = (fields[0] + ' ').split(' _eos_ ')[:-1]
                    candidate_list = fields[1].split('|')
                    label = int(fields[2])
                    document_list = fields[3].split('|')

                    self.contexts.append(context)
                    self.candidate_lists.append(candidate_list)
                    self.sparse_labels.append(label)
                    self.document_lists.append(document_list)
                    self.n_turns.append(len(context))

            self.context_lens, self.document_lens, self.candidate_list_lens = None, None, None
            self.context_lens, self.document_lens, self.candidate_list_lens = self.w2v_preprocessing()
            self.samples = self.make_samples()
            cache = {'samples': self.samples,
                     'statistics': self.statistics}
            end = time.time()
            self.logger.info('Preprocessing done, costing %s mins' % ((end - start) / 60))
            if not args.debug:
                torch.save(cache, data_cache_path)
        else:
            start = time.time()
            self.logger.info('loading cache from [%s]' % data_cache_path)
            cache = torch.load(data_cache_path)
            self.samples = cache['samples']
            self.statistics = cache['statistics']
            end = time.time()
            self.logger.info('Cache loaded, costing %s mins' % ((end - start) / 60))
        self.logger.info(pformat(self.statistics, indent=4))

    def w2v_preprocessing(self):
        # tokenization & numericalization
        ## tokenize & numericalize dials and docs
        self.logger.debug('tokenzing & numericalizing dialogues and docs...')

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
        context_lens, document_lens, candidate_list_lens = [], [], []
        ## dial
        for i in tqdm(range(len(self.contexts))):
            context_len = []
            self.contexts[i] = self.contexts[i][-self.args.max_turn_num:]
            for j in range(len(self.contexts[i])):
                self.contexts[i][j] = self.contexts[i][j][:self.args.seq_len]
                context_len.append(len(self.contexts[i][j]))
                self.contexts[i][j] += [0] * (self.args.seq_len - len(self.contexts[i][j]))
            self.contexts[i] = [[0] * self.args.seq_len] * (self.args.max_turn_num - len(self.contexts[i])) + self.contexts[i]
            context_len = [0] * (self.args.max_turn_num - len(context_len)) + context_len
            context_lens.append(context_len)

        for i in tqdm(range(len(self.candidate_lists))):
            candidate_list_len = []
            for j in range(len(self.candidate_lists[i])):
                self.candidate_lists[i][j] = self.candidate_lists[i][j][:self.args.seq_len]
                candidate_list_len.append(len(self.candidate_lists[i][j]))
                self.candidate_lists[i][j] += [0] * (self.args.seq_len - len(self.candidate_lists[i][j]))
            candidate_list_lens.append(candidate_list_len)

        ## docs
        self.logger.debug('truncating and padding documents...')
        for i in tqdm(range(len(self.document_lists))):
            document_len = []
            self.document_lists[i] = self.document_lists[i][:self.args.max_doc_num]
            for j in range(len(self.document_lists[i])):
                self.document_lists[i][j] = self.document_lists[i][j][:self.args.doc_len]
                document_len.append(len(self.document_lists[i][j]))
                self.document_lists[i][j] += [0] * (self.args.doc_len - len(self.document_lists[i][j]))
            self.document_lists[i] += [[0] * self.args.doc_len] * (self.args.max_doc_num - len(self.document_lists[i]))
            document_len += [0] * (self.args.max_doc_num - len(document_len))
            document_lens.append(document_len)

        return context_lens, document_lens, candidate_list_lens

    def make_samples(self):
        samples = []
        for i in range(len(self.contexts)):
        # for context, candidate_list, document_list, sparse_label, n_turn, n_document, context_len, document_len, candidate_list_len in zip(self.contexts, self.candidate_lists, self.document_lists, self.context_chars, self.candidate_chars, self.document_chars, self.sparse_labels, self.n_turns, self.n_documents, self.context_lens, self.document_lens, self.candidate_list_lens):
            for index, (candidate, candidate_chars, response_len) in enumerate(zip(self.candidate_lists[i], self.candidate_chars[i], self.candidate_list_lens[i])):
                label = 1 if index == self.sparse_labels[i] else 0
                sample = {'context': self.contexts[i],
                          'document': self.document_lists[i],
                          'response': candidate,
                          'context_char': self.context_chars[i],
                          'document_char': self.document_chars[i],
                          'response_char': candidate_chars,
                          'context_len': self.context_lens[i],
                          'document_len': self.document_lens[i],
                          'response_len': response_len,
                          'label': label,
                          'n_turn': min(self.n_turns[i], self.args.max_turn_num),
                          'n_document': min(self.n_documents[i], self.args.max_doc_num)}
                samples.append(sample)

        return samples

    def __getitem__(self, index):
        sample = self.samples[index]

        context = torch.tensor(sample['context'], dtype=torch.long)
        response = torch.tensor(sample['response'], dtype=torch.long)
        document = torch.tensor(sample['document'], dtype=torch.long)
        context_char = torch.tensor(sample['context_char'], dtype=torch.long)
        response_char = torch.tensor(sample['response_char'], dtype=torch.long)
        document_char = torch.tensor(sample['document_char'], dtype=torch.long)
        context_len = torch.tensor(sample['context_len'], dtype=torch.long)
        response_len = torch.tensor(sample['response_len'], dtype=torch.long)
        document_len = torch.tensor(sample['document_len'], dtype=torch.long)
        n_turn = torch.tensor(sample['n_turn'], dtype=torch.int)
        n_document = torch.tensor(sample['n_document'], dtype=torch.int)
        label = torch.tensor(sample['label'], dtype=torch.long)
        pos1d_ids = torch.tensor(np.arange(0, len(document)), dtype=torch.long)

        inputs = {'context': context,
                  'response': response,
                  'document': document,
                  'context_char': context_char,
                  'response_char': response_char,
                  'document_char': document_char,
                  'context_len': context_len,
                  'document_len': document_len,
                  'response_len': response_len,
                  'label': label,
                  'pos1d_ids': pos1d_ids,
                  'n_turn': n_turn,
                  'n_document': n_document}

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
                            dial_path=cmudog_train_dial_path,
                            glove_path=cmudog_glove_path,
                            data_cache_path='temp.pkl')

    dataloader = DataLoader(dataset=dataset,
                            batch_size=40,
                            shuffle=False)

    for index, inputs in enumerate(dataloader):
        print(inputs)
        for key, value in inputs.items():
            print(key, ':', value.shape)
