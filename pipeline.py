from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from torch.utils.data import Dataset, DataLoader
import fitlog
# fitlog.debug()

from config_file import *


class Metrics(object):

    def __init__(self, score_file_path=None, segment=10):
        super(Metrics, self).__init__()
        self.score_file_path = score_file_path
        self.segment = segment

    def __collect_session(self, y_pred, y_gt):
        sessions = []
        one_sess = []
        i = 0
        for score, label in zip(y_pred, y_gt):
            i += 1
            one_sess.append((float(score), float(label)))
            if i % self.segment == 0:
                one_sess_tmp = np.array(one_sess)
                if one_sess_tmp[:, 1].sum() > 0:
                    sessions.append(one_sess)
                one_sess = []
        return sessions

    def __read_score_file(self, score_file_path):
        sessions = []
        one_sess = []
        with open(score_file_path, 'r') as infile:
            i = 0
            for line in infile.readlines():
                i += 1
                tokens = line.strip().split('\t')
                one_sess.append((float(tokens[0]), float(tokens[1])))
                if i % self.segment == 0:
                    one_sess_tmp = np.array(one_sess)
                    if one_sess_tmp[:, 1].sum() > 0:
                        sessions.append(one_sess)
                    one_sess = []
        return sessions

    def __mean_average_precision(self, sort_data):
        # to do
        count_1 = 0
        sum_precision = 0
        for index in range(len(sort_data)):
            if sort_data[index][1] == 1:
                count_1 += 1
                sum_precision += 1.0 * count_1 / (index + 1)
        return sum_precision / count_1

    def __mean_reciprocal_rank(self, sort_data):
        sort_lable = [s_d[1] for s_d in sort_data]
        assert 1 in sort_lable
        return 1.0 / (1 + sort_lable.index(1))

    def __precision_at_position_1(self, sort_data):
        if sort_data[0][1] == 1:
            return 1
        else:
            return 0

    def __recall_at_position_k_in_10(self, sort_data, k):
        sort_label = [s_d[1] for s_d in sort_data]
        select_label = sort_label[:k]
        return 1.0 * select_label.count(1) / sort_label.count(1)

    def evaluation_one_session(self, data):
        '''
        :param data: one conversion session, which layout is [(score1, label1), (score2, label2), ..., (score10, label10)].
        :return: all kinds of metrics used in paper.
        '''
        np.random.shuffle(data)
        sort_data = sorted(data, key=lambda x: x[0], reverse=True)
        m_a_p = self.__mean_average_precision(sort_data)
        m_r_r = self.__mean_reciprocal_rank(sort_data)
        p_1 = self.__precision_at_position_1(sort_data)
        r_1 = self.__recall_at_position_k_in_10(sort_data, 1)
        r_2 = self.__recall_at_position_k_in_10(sort_data, 2)
        r_5 = self.__recall_at_position_k_in_10(sort_data, 5)
        return m_a_p, m_r_r, p_1, r_1, r_2, r_5

    def evaluate_all_metrics(self, y_pred=None, y_gt=None):
        sum_r_1, sum_r_2, sum_r_5, sum_p_1, sum_m_a_p, sum_m_r_r = [0] * 6

        if self.score_file_path:
            sessions = self.__read_score_file(self.score_file_path)
        else:
            sessions = self.__collect_session(y_pred, y_gt)

        total_s = len(sessions)
        for session in sessions:
            m_a_p, m_r_r, p_1, r_1, r_2, r_5 = self.evaluation_one_session(session)
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5
            sum_p_1 += p_1
            sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r

        return (sum_r_1 / total_s, sum_r_2 / total_s, sum_r_5 / total_s, sum_p_1 / total_s, sum_m_a_p / total_s,
                sum_m_r_r / total_s)

    def evaluate_all(self):
        all_r_1 = []
        all_r_2 = []
        all_r_5 = []

        sessions = self.__read_score_file(self.score_file_path)
        total_s = len(sessions)
        for session in sessions:
            m_a_p, m_r_r, p_1, r_1, r_2, r_5 = self.evaluation_one_session(session)
            all_r_1.append(r_1)
            all_r_2.append(r_2)
            all_r_5.append(r_5)

        return all_r_1, all_r_2, all_r_5


class Pipeline():
    def __init__(self, args, logger, network, optimizer_f, loss_function, datasets):
        super(Pipeline, self).__init__()
        self.args = args
        self.logger = logger

        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.best_result = [0, 0, 0]
        self.metrics = Metrics(segment=10)
        self.device = DEVICE

        self.train_dataloader, self.valid_dataloader, self.test_dataloader = self.prepare_data(datasets)

        self.network = network.to(self.device)
        if args.gpuno:
            self.network = nn.DataParallel(self.network, device_ids=[int(c) for c in args.gpuno.split(',')])
            self.network = network.cuda()

        try:
            pretrained_2DEmbedding_params = list(map(id, self.network.layout_embedding.parameters()))
            base_params = filter(lambda p: id(p) not in pretrained_2DEmbedding_params, self.network.parameters())
            self.optimizer = optimizer_f(params=[{'params': base_params},
                                                 {'params': self.network.layout_embedding.parameters(),
                                                  'lr': self.args.learning_rate}],
                                         lr=self.args.learning_rate,
                                         weight_decay=self.args.weight_decay)
        except:
            self.optimizer = optimizer_f(self.network.parameters(),
                                         lr=self.args.learning_rate,
                                         weight_decay=self.args.weight_decay)
        self.loss_function = loss_function
        self.best_state_dict = self.network.state_dict()

    def prepare_data(self, datasets):
        train_dataloader = DataLoader(dataset=datasets['train'],
                                      batch_size=self.args.batch_size,
                                      shuffle=True,
                                      num_workers=0) if 'train' in datasets else None
        valid_dataloader = DataLoader(dataset=datasets['valid'],
                                      batch_size=self.args.batch_size,
                                      shuffle=False,
                                      num_workers=0) if 'valid' in datasets else None
        test_dataloader = DataLoader(dataset=datasets['test'],
                                     batch_size=self.args.batch_size,
                                     shuffle=False,
                                     num_workers=0) if 'test' in datasets else None
        return train_dataloader, valid_dataloader, test_dataloader

    def train_step(self, inputs):
        with torch.no_grad():
            batch_y = inputs['label'].to(self.device)
            for key in ['context', 'response', 'document', 'n_turn']:
                inputs[key] = inputs[key].to(self.device)

        self.optimizer.zero_grad()
        logits = self.network(inputs)
        loss = self.loss_function(logits, target=batch_y)
        torch.clamp(loss, -10, 10)
        loss.backward()
        self.optimizer.step()
        return loss, batch_y.size(0), batch_y

    def train(self):
        train_losses = []
        n_samples = 0
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            print("\nEpoch ", epoch + 1, "/", self.args.epochs)
            avg_loss = 0

            self.network.train()
            y_train = []
            with tqdm(total=self.train_dataloader.__len__() * self.args.batch_size, ncols=90) as pbar:
                for i, inputs in enumerate(self.train_dataloader):
                    loss, batch_size, batch_y = self.train_step(inputs)
                    n_samples += batch_size
                    y_train.append(batch_y)
                    pbar.set_postfix(lr=self.args.learning_rate, loss=loss.item())

                    # if i > 0 and i % 2000 == 0:
                    if n_samples % self.args.valid_every == 0:
                        self.network.eval()
                        result = self.evaluate(dataloader=self.valid_dataloader)
                        fitlog.add_metric({"validation":
                                               {"R@1": result[0],
                                                "R@2": result[1],
                                                "R@5": result[2]
                                               }
                                          }, step=epoch)
                        self.network.train()

                    if n_samples > self.args.focusing_sample and n_samples % self.args.test_every == 0:
                        self.network.eval()
                        result = self.evaluate(dataloader=self.test_dataloader, is_test=True)
                        fitlog.add_metric({"test":
                                               {"R@1": result[0],
                                                "R@2": result[1],
                                                "R@5": result[2]
                                               }
                                          }, step=epoch)
                        self.network.train()

                    if epoch >= 1 and self.patience >= 3:
                        # tqdm.write("Reload the best model...")
                        self.load_model(load_from_file=False)
                        self.adjust_learning_rate()
                        self.patience = 0

                    if self.init_clip_max_norm is not None:
                        utils.clip_grad_norm_(self.network.parameters(), max_norm=self.init_clip_max_norm)

                    pbar.update(batch_size)
                    avg_loss += loss.item()
            cnt = len(y_train) // self.args.batch_size + 1
            tqdm.write("Average loss:{:.6f} ".format(avg_loss / cnt))
            tqdm.write("Best Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (self.best_result[0], self.best_result[1], self.best_result[2]))
            fitlog.add_loss(avg_loss / cnt, step=epoch, name='Average loss')
            fitlog.add_best_metric({"validation":
                                        {"R@1": self.best_result[0],
                                         "R@2": self.best_result[1],
                                         "R@5": self.best_result[2]
                                         }
                                    })
            result = self.evaluate(dataloader=self.test_dataloader, is_test=True)
            fitlog.add_metric({"test":
                                   {"R@1": result[0],
                                    "R@2": result[1],
                                    "R@5": result[2]
                                    }
                               }, step=epoch)
        self.load_model(load_from_file=False)
        result = self.evaluate(dataloader=self.test_dataloader, is_test=True)
        self.save_model(save_to_file=True)
        fitlog.finish()

    def adjust_learning_rate(self, decay_rate=0.5):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.args.learning_rate = param_group['lr']
        self.optimizer.lr = self.args.learning_rate
        # tqdm.write("Decay learning rate to: " + str(self.args.learning_rate))

    def evaluate(self, dataloader, is_test=False):
        y_pred, y_gt = self.predict(dataloader)
        # with open(self.args.score_file_path, 'w') as output:
        #     for score, label in zip(y_pred, y_gt):
        #         output.write(str(score) + '\t' + str(label) + '\n')

        result = self.metrics.evaluate_all_metrics(y_pred=y_pred, y_gt=y_gt)

        # if not is_test and result[0] + result[1] + result[2] > self.best_result[0] + self.best_result[1] + self.best_result[2]:
        # We record when R@1 reaches the best value. When R@1 is the same as the recorded one, check the sum of R@1, R@2 and R@5.
        if not is_test and result[0] > self.best_result[0] or (result[0] == self.best_result[0] and result[0] + result[1] + result[2] > self.best_result[0] + self.best_result[1] + self.best_result[2]):
            # tqdm.write("save model!!!")
            self.best_result = result
            tqdm.write("Best Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (self.best_result[0], self.best_result[1], self.best_result[2]))
            self.logger.info("Best Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (self.best_result[0], self.best_result[1], self.best_result[2]))
            self.patience = 0
            self.save_model(save_to_file=False)
        else:
            tqdm.write("This Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (result[0], result[1], result[2]))
            self.logger.info("Best Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (
            self.best_result[0], self.best_result[1], self.best_result[2]))
            self.patience += 1

        if is_test:
            print("Evaluation Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (result[0], result[1], result[2]))

        return result

    def predict(self, dataloader):
        self.network.eval()
        y_pred, y_gt = [], []

        with torch.no_grad():
            for i, inputs in enumerate(dataloader):
                y = inputs['label'].to(self.device)
                for key in ['context', 'response', 'document', 'n_turn']:
                    inputs[key] = inputs[key].to(self.device)

                logits = self.network(inputs)
                y_pred.append(logits.data.cpu().numpy().reshape(-1))
                y_gt.append(y.data.cpu().numpy().reshape(-1))
        y_pred = np.concatenate(y_pred, axis=0).tolist()
        y_gt = np.concatenate(y_gt, axis=0).tolist()
        return y_pred, y_gt

    def run_case_by_docid(self, docid):
        n_max = 2
        with torch.no_grad():
            for i, inputs in enumerate(self.test_dataloader):
                if docid == inputs['doc_id'][0]:
                    if n_max == 2:
                        logits_positive, debugging_collection_positive = self.network(inputs, debug=True)
                        inputs_positive = inputs
                        n_max -= 1
                    else:
                        logits_negative, debugging_collection_negative = self.network(inputs, debug=True)
                        inputs_negative = inputs
                        return inputs_positive, logits_positive, debugging_collection_positive, \
                               inputs_negative, logits_negative, debugging_collection_negative

    def save_model(self, save_to_file=False):
        self.network.cpu()
        if save_to_file:
            torch.save(self.best_state_dict, self.args.save_path)
        else:
            model_to_save = self.network.module if hasattr(self.network, 'module') else self.network
            self.best_state_dict = model_to_save.state_dict()
        self.network.to(self.device)

    def load_model(self, load_from_file=False):
        self.network.cpu()
        if load_from_file:
            self.network.load_state_dict(torch.load(self.args.save_path))
        else:
            self.network.load_state_dict(self.best_state_dict)
        self.network.to(self.device)


if __name__ == '__main__':
    score_file_path = '../score_file.txt'
    metrics = Metrics(score_file_path)
    results = metrics.evaluate_all_metrics()
    print(results)
