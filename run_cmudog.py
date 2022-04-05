import os
import time
import logging
import random
import argparse
import fitlog
import numpy as np
# fitlog.debug()

from pipeline import Pipeline
from network.dgmn import DGMN
from data_provider.data_provider_cmudog import CMUDoGDataset

from config_file import *


# Required parameters
parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument("--task", default='cmudog', type=str, help="The dataset used for training and test. Options: [cmudog, personachat]")
parser.add_argument("--is_training", default=True, type=bool, help="Training model or evaluating model?")
parser.add_argument("--batch_size", default=32, type=int, help="The batch size.")
parser.add_argument("--gru_hidden", default=300, type=int, help="The hidden size of GRU in layer 1")
parser.add_argument("--emb_size", default=300, type=int, help="The embedding size")
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay coefficient")
parser.add_argument("--epochs", default=15, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--save_path", default="checkpoint", type=str, help="The path to save model.")
parser.add_argument("--gpuno", default='0', type=str, help='Number of GPU. e.g. 0,1,2,3->[0,1,2,3]')
parser.add_argument("--model", default='dgmn', help='The name of model.')
parser.add_argument("--doc_len", default=40, type=int, help='Maximum #tokens/doc')
parser.add_argument("--seq_len", default=40, type=int, help='Maximum #tokens/turn')
parser.add_argument("--max_turn_num", default=4, type=int, help='Maximum #turn')
parser.add_argument("--max_doc_num", default=20, type=int, help='Maximum #turn')
parser.add_argument("--focusing_sample", default=0, type=int, help='Keep training n samples without testing.')
parser.add_argument("--valid_every", default=100000, type=int)
parser.add_argument("--test_every", default=100000, type=int)
args = parser.parse_args()

fitlog.set_log_dir(log_dir='fitlogs/')
# fitlog.commit(__file__)
fitlog.add_hyper(args)
fitlog.add_hyper_in_file(__file__)

save_model_name = args.model + \
                  '_bs%s' % args.batch_size + \
                  '_lr%s' % args.learning_rate + \
                  '_doclen%s' % args.doc_len + \
                  '_seqlen%s' % args.seq_len + \
                  '_seed%s' % args.seed + \
                  '.pkl'

data_cache_name = cache_root_dir + '/' + args.task + '/' + \
    'cache_DocLen%s' % args.doc_len + \
    '_SeqLen%s' % args.seq_len + \
    '_MaxTurnNum%s' % args.max_turn_num + \
    '_MaxDocNum%s' % args.max_doc_num

args.save_path = os.path.join(cmudog_root_dir, args.save_path + 'gpu%s' % args.gpuno, save_model_name)

if not os.path.exists(os.path.split(args.save_path)[0]):
    os.mkdir(os.path.split(args.save_path)[0])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_list = {'dgmn': DGMN}

NETWORK_f = model_list[args.model.lower()]

print(args)
print("Task: ", args.task)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    # torch.backends.cudnn.deterministic = True


def train_pipeline():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.CRITICAL,
    )

    args.logger = logging.getLogger(__name__)
    args.disable_tqdm = False if args.local_rank in [-1, 0] else True

    network = NETWORK_f(args=args)
    n_params = sum([p.numel() for p in network.parameters() if p.requires_grad])
    args.logger.info('* number of parameters: %d' % n_params)

    train_dataset = CMUDoGDataset(args=args,
                                  dial_path=cmudog_train_dial_path,
                                  glove_path=cmudog_glove_path,
                                  char_path=cmudog_char_path,
                                  data_cache_path=data_cache_name + '_train.pkl')
    valid_dataset = CMUDoGDataset(args=args,
                                  dial_path=cmudog_valid_dial_path,
                                  glove_path=cmudog_glove_path,
                                  char_path=cmudog_char_path,
                                  data_cache_path=data_cache_name + '_valid.pkl')
    test_dataset = CMUDoGDataset(args=args,
                                 dial_path=cmudog_test_dial_path,
                                 glove_path=cmudog_glove_path,
                                 char_path=cmudog_char_path,
                                 data_cache_path=data_cache_name + '_test.pkl')

    pipeline = Pipeline(args=args,
                        network=network,
                        optimizer_f=torch.optim.Adam,
                        loss_function=torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing),
                        datasets={'train': train_dataset,
                                  'valid': valid_dataset,
                                  'test': test_dataset})

    pipeline.train()


def test_pipeline():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.CRITICAL,
    )

    args.logger = logging.getLogger(__name__)

    network = NETWORK_f(args=args)

    test_dataset = CMUDoGDataset(args=args,
                                 dial_path=cmudog_test_dial_path,
                                 glove_path=cmudog_glove_path,
                                 char_path=cmudog_char_path,
                                 data_cache_path=data_cache_name + '_test.pkl')

    pipeline = Pipeline(args=args,
                        network=network,
                        optimizer_f=None,
                        loss_function=None,
                        datasets={'train': None,
                                  'valid': None,
                                  'test': test_dataset})

    pipeline.load_model(load_from_file=True, model_path=args.inference_model)
    pipeline.evaluate(dataloader=pipeline.test_dataloader, is_test=True, full_evaluate=True)


if __name__ == '__main__':
    task_dic = {
        'CMUDoG': "./data/",
    }

    # Required parameters
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument("--seed", default=1234, type=int, help="random seed")
    parser.add_argument("--task", default='cmudog', type=str, help="The dataset used for training and test. Options: [doc2dial, CMUDoG, persona]")
    parser.add_argument("--is_training", action='store_true', help="Training model or evaluating model?")
    parser.add_argument("--local_rank", default=-1, type=int, help='local_rank')
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level", type=str, default="O1", help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].""See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--save_path", default="checkpoint", type=str, help="The path to save model.")
    parser.add_argument("--inference_model", default='', type=str, help='Trained model for downstream-task inference')
    # hyper-parameters
    ## model
    parser.add_argument("--hidden_size", default=300, type=int, help="The hidden size of GRU in layer 1")
    parser.add_argument("--emb_size", default=300, type=int, help="The embedding size")
    parser.add_argument("--gamma", default=0.3, type=float)
    parser.add_argument("--pretrained_model", default='', type=str, help='Follow what includes in Hugging Face Models. Options: [EMPTY, bert-base-uncased, allenai/longformer-base-4096]')
    parser.add_argument("--freeze", default=False, type=bool, help="freeze pretrained model or not.")
    parser.add_argument("--n_layer", default=2, type=bool, help="#Layer of model.")
    parser.add_argument("--model", default='bert', help='The name of model.')
    ## data
    parser.add_argument("--percent", default=100, type=int, help='Proportion of the dataset size.')
    parser.add_argument("--doc_len", default=40, type=int, help='Maximum #tokens/doc')
    parser.add_argument("--seq_len", default=40, type=int, help='Maximum #tokens/turn')
    parser.add_argument("--word_len", default=18, type=int, help='Maximum #chars/word')
    parser.add_argument("--max_turn_num", default=4, type=int, help='Maximum #turn')
    parser.add_argument("--max_doc_num", default=20, type=int, help='Maximum #turn')
    parser.add_argument("--bert_document_len", default=256, type=int, help='Max document length in BERT.')
    parser.add_argument("--bert_history_len", default=168, type=int, help='Max conversation history length in BERT.')
    parser.add_argument("--bert_query_len", default=40, type=int, help='Max query length in BERT.')
    parser.add_argument("--bert_response_len", default=40, type=int, help='Max response length in BERT.')
    parser.add_argument("--bert_max_seq_len", default=512, type=int, help='Max total sequence length in BERT.')
    parser.add_argument("--bert_document_len_overlap", default=128, type=int, help='Overlap document length in BERT.')
    ## training
    parser.add_argument("--batch_size", default=16, type=int, help="The batch size.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-5, type=float, help="weight decay coefficient")
    parser.add_argument("--label_smoothing", default=0.0, type=float)
    parser.add_argument("--epochs", default=15, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--valid_focusing_sample", default=0, type=int, help='Keep training n samples without testing.')
    parser.add_argument("--test_focusing_sample", default=0, type=int, help='Keep training n samples without testing.')
    parser.add_argument("--valid_every", default=100000, type=int)
    parser.add_argument("--test_every", default=100000, type=int)
    parser.add_argument("--debug", action='store_true', default=False)
    ## inference
    parser.add_argument("--n_try", default=3, type=int, help='#Ensemble')
    args = parser.parse_args()

    start = time.time()
    set_seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger = logging.getLogger(__name__)

    ## easy calculation

    ## initialize CUDA distributed training
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        args.n_gpu = 1 if torch.cuda.is_available() else 0
    else:
        # 每个进程根据自己的local_rank设置应该使用的GPU
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    fitlog.set_log_dir(log_dir='fitlogs/')
    # fitlog.commit(__file__)
    fitlog.add_hyper(args)
    fitlog.add_hyper_in_file(__file__)

    data_cache_name = cache_root_dir + '/' + args.task + '/' + \
                      'cache_DocLen%s' % args.doc_len + \
                      '_SeqLen%s' % args.seq_len + \
                      '_MaxTurnNum%s' % args.max_turn_num + \
                      '_MaxDocNum%s' % args.max_doc_num + \
                      '_%sPercent' % args.percent

    args.save_path = os.path.join(model_root_dir, args.task, 'gpu%s' % args.local_rank, args.save_path)

    if not os.path.exists(os.path.split(args.save_path)[0]):
        os.mkdir(os.path.split(args.save_path)[0])

    model_list = {'dgmn': DGMN}

    NETWORK_f = model_list[args.model.lower()]

    logger.info(args)
    logger.info("Task: %s" % args.task)

    if args.is_training:
        # train_model()
        logger.info('=' * 25 + ' dump files ' + '=' * 25)
        recording_file_list = ['retrieval_based/run.py', 'config_file.py', 'retrieval_based/pipeline.py'] + \
                              ['retrieval_based/network/%s.py' % f for f in ['__init__',
                                                                             'attention',
                                                                             'basic',
                                                                             'csn',
                                                                             'dam',
                                                                             'dgmn',
                                                                             'layout',
                                                                             'network',
                                                                             'skim_attention',
                                                                             'smn']]
        for filename in recording_file_list:
            logger.info('-' * 25 + filename + '-' * 25)
            with open(os.path.join(dgds_code_root_dir, filename)) as f:
                logger.info(f.read())
        logger.info('=' * 25 + ' dump finished. ' + '=' * 25)
        train_pipeline()
    else:
        logger.info('Testing......')
        test_pipeline()
    end = time.time()
    logger.info("use time: %s mins" % ((end - start) / 60))
