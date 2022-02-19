import os
import time
import random
import logging
import argparse
import fitlog
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
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def train_pipeline():
    network = NETWORK_f(args=args)
    n_params = sum([p.numel() for p in network.parameters() if p.requires_grad])
    print('* number of parameters: %d' % n_params)

    train_dataset = CMUDoGDataset(args=args,
                                  dial_path=cmudog_train_dial_path,
                                  glove_path=cmudog_glove_path,
                                  data_cache_path=data_cache_name + '_train.pkl')
    valid_dataset = CMUDoGDataset(args=args,
                                  dial_path=cmudog_valid_dial_path,
                                  glove_path=cmudog_glove_path,
                                  data_cache_path=data_cache_name + '_valid.pkl')
    test_dataset = CMUDoGDataset(args=args,
                                 dial_path=cmudog_test_dial_path,
                                 glove_path=cmudog_glove_path,
                                 data_cache_path=data_cache_name + '_test.pkl')


    pipeline = Pipeline(args=args,
                        logger=logger,
                        network=network,
                        optimizer_f=torch.optim.Adam,
                        loss_function=torch.nn.BCELoss(),
                        datasets={'train': train_dataset,
                                  'valid': valid_dataset,
                                  'test': test_dataset})

    pipeline.train()


def test_pipeline():
    network = NETWORK_f(args=args)

    test_dataset = CMUDoGDataset(args=args,
                                 dial_path=cmudog_test_dial_path,
                                 glove_path=cmudog_glove_path,
                                 data_cache_path=data_cache_name + '_test.pkl')

    pipeline = Pipeline(args=args,
                        logger=logger,
                        network=network,
                        optimizer_f=None,
                        loss_function=None,
                        datasets={'train': None,
                                  'valid': None,
                                  'test': test_dataset})

    pipeline.predict(dataloader=pipeline.test_dataloader)


if __name__ == '__main__':
    start = time.time()
    set_seed(args.seed)

    print('=' * 25 + ' dump files ' + '=' * 25)
    recording_file_list = ['run.py', 'config_file.py', 'pipeline.py'] + \
                          ['network/%s.py' % f for f in ['__init__',
                                                         'basic',
                                                         'dgmn',
                                                         'network']]
    for filename in recording_file_list:
        print('-' * 25 + filename + '-' * 25)
        with open(os.path.join(dgds_code_root_dir, filename)) as f:
            print(f.read())
    print('=' * 25 + ' dump finished. ' + '=' * 25)

    if args.is_training:
        train_pipeline()
    else:
        test_pipeline()
    end = time.time()
    print("use time: ", (end - start) / 60, " min")
