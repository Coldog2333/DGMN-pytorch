import os
import torch

from secret_config import *

# cmudog
cmudog_root_dir = dataset_root_dir + '/' + 'cmudog'
cmudog_train_dial_path = cmudog_root_dir + '/' + 'processed_train_self_original_fullSection.txt'
cmudog_test_dial_path = cmudog_root_dir + '/' + 'processed_test_self_original_fullSection.txt'
cmudog_valid_dial_path = cmudog_root_dir + '/' + 'processed_valid_self_original_fullSection.txt'
cmudog_glove_path = model_root_dir + '/' + 'cmudog' + '/' + 'glove_42B_300d_vec_plus_word2vec_100.txt'

# personachat TODO: to be completed
personachat_root_dir = dataset_root_dir + '/' + 'personachat'
personachat_train_dial_path = personachat_root_dir + '/' + 'processed_train_self_original_fullSection.txt'
personachat_test_dial_path = personachat_root_dir + '/' + 'processed_test_self_original_fullSection.txt'
personachat_valid_dial_path = personachat_root_dir + '/' + 'processed_valid_self_original_fullSection.txt'
personachat_glove_path = model_root_dir + '/' + 'personachat' + '/' + 'glove_42B_300d_vec_plus_word2vec_100.txt'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
