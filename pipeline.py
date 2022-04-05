from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils as utils
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import fitlog
# fitlog.debug()

try:
    from apex import amp
except:
    pass

from metric.evaluation import Metrics
from config_file import *


class Pipeline():
    def __init__(self, args, network, optimizer_f, loss_function, datasets):
        super(Pipeline, self).__init__()
        self.args = args
        self.logger = args.logger
        self.disable_tqdm = args.disable_tqdm if args.local_rank in [-1, 0] else False

        # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
        # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
        # remove the need for this code, but it is still valid.
        if args.fp16:
            try:
                import apex

                apex.amp.register_half_function(torch, "einsum")
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        self.patience = 0
        self.init_clip_max_norm = 5.0
        self.best_result = [0., 0., 0.]
        segment_dict = {'cmudog': 20, 'personachat': 20}
        self.metrics = Metrics(segment=segment_dict[self.args.task])

        self.network = network.to(self.args.device)
        optimizer_f = optimizer_f or optim.SGD

        self.optimizer = optimizer_f(self.network.parameters(),
                                     lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)

        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

            self.network, self.optimizer = amp.initialize(self.network, self.optimizer, opt_level=args.fp16_opt_level)


        if args.n_gpu > 1:
            self.network = torch.nn.DataParallel(self.network)

        if args.local_rank != -1:
            self.network = torch.nn.parallel.DistributedDataParallel(self.network,
                                                                     device_ids=[args.local_rank],
                                                                     output_device=args.local_rank)
                                                                     #find_unused_parameters=True)

        self.datasets = datasets
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = self.prepare_data(self.datasets)

        self.loss_function = loss_function
        self.best_state_dict = self.network.state_dict()

    def prepare_data(self, datasets):
        if self.args.local_rank not in [-1, 0]:
            torch.distributed.barrier()

        train_dataloader, valid_dataloader, test_dataloader = None, None, None

        if datasets['train'] is not None:
            train_sampler = RandomSampler(datasets['train']) if self.args.local_rank == -1 else DistributedSampler(datasets['train'], shuffle=True, seed=self.args.seed)
            train_dataloader = DataLoader(dataset=datasets['train'],
                                          sampler=train_sampler,
                                          batch_size=self.args.batch_size,
                                          drop_last=True,
                                          num_workers=0)

        if datasets['valid'] is not None:
            valid_sampler = SequentialSampler(datasets['valid'])
            valid_dataloader = DataLoader(dataset=datasets['valid'],
                                          sampler=valid_sampler,
                                          batch_size=self.args.batch_size,
                                          num_workers=0)

        if datasets['test'] is not None:
            test_sampler = SequentialSampler(datasets['test'])
            test_dataloader = DataLoader(dataset=datasets['test'],
                                         sampler=test_sampler,
                                         batch_size=self.args.batch_size,
                                         num_workers=0)

        if self.args.local_rank == 0:
            torch.distributed.barrier()

        return train_dataloader, valid_dataloader, test_dataloader

    def train_step(self, inputs):
        with torch.no_grad():
            batch_y = inputs['label'].to(self.args.device)
            no_cuda_keys = []
            for key in inputs.keys():
                if key not in no_cuda_keys:
                    inputs[key] = inputs[key].to(self.args.device)

        self.optimizer.zero_grad()

        logits = self.network(inputs)
        loss = self.loss_function(logits, target=batch_y)

        if self.args.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        # if self.args.fp16:
        #     torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), -10, 10)
        # else:
        #     torch.nn.utils.clip_grad_norm_(self.network.parameters(), -10, 10)

        self.optimizer.step()
        return loss, batch_y.size(0), batch_y

    def train(self):
        train_losses = []
        n_batches = 0
        for epoch in range(self.args.epochs):
            self.epoch = epoch
            self.logger.info("\nEpoch %s/%s" % (epoch + 1, self.args.epochs))
            avg_loss = 0

            self.network.train()
            y_train = []
            with tqdm(total=self.train_dataloader.__len__() * self.args.batch_size, ncols=90, disable=self.disable_tqdm) as pbar:
                for i, inputs in enumerate(self.train_dataloader):
                    loss, batch_size, batch_y = self.train_step(inputs)
                    n_batches += 1
                    y_train.append(batch_y)
                    pbar.set_postfix(lr=self.args.learning_rate, loss=loss.item())

                    # if i > 0 and i % 2000 == 0:
                    if n_batches * batch_size > self.args.valid_focusing_sample and n_batches % int(self.args.valid_every / batch_size) == 0:
                        self.network.eval()
                        result = self.evaluate(dataloader=self.valid_dataloader, is_test=False)
                        fitlog.add_metric({"validation":
                                               {"R@1": result[0],
                                                "R@2": result[1],
                                                "R@5": result[2]
                                               }
                                          }, step=epoch)
                        self.network.train()

                    if n_batches * batch_size > self.args.test_focusing_sample and n_batches % int(self.args.test_every / batch_size) == 0:
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

            self.logger.info("Average loss:{:.6f} ".format(avg_loss / cnt))
            self.logger.info("Best Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (self.best_result[0], self.best_result[1], self.best_result[2]))
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
        if self.args.local_rank in [-1, 0]:
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

        if is_test:
            self.logger.info("Testing Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (result[0], result[1], result[2]))
        else:
            # if not is_test and result[0] + result[1] + result[2] > self.best_result[0] + self.best_result[1] + self.best_result[2]:
            # We record when R@1 reaches the best value. When R@1 is the same as the recorded one, check the sum of R@1, R@2 and R@5.
            if result[0] > self.best_result[0] or (result[0] == self.best_result[0] and result[0] + result[1] + result[2] > self.best_result[0] + self.best_result[1] + self.best_result[2]):
                self.best_result = result
                self.logger.info("\nBest Evaluation Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (self.best_result[0], self.best_result[1], self.best_result[2]))
                self.patience = 0
                self.save_model(save_to_file=False)
            else:
                self.logger.info("\nThis Evalution Result: R@1: %.3f R@2: %.3f R@5: %.3f" % (result[0], result[1], result[2]))
                self.patience += 1

        return result

    def predict(self, dataloader):
        self.network.eval()
        y_pred, y_gt = [], []

        with torch.no_grad():
            for i, inputs in enumerate(dataloader):
                y = inputs['label'].to(self.args.device)
                no_cuda_keys = ['doc_id']
                for key in inputs.keys():
                    if key not in no_cuda_keys:
                        inputs[key] = inputs[key].to(self.args.device)

                logits = self.network(inputs)
                scores = torch.softmax(logits, dim=-1)[:, 1]
                y_pred.append(scores.data.cpu().numpy().reshape(-1))
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
            self.logger.info('Save model -> [%s]' % self.args.save_path)
        else:
            model_to_save = self.network.module if hasattr(self.network, 'module') else self.network
            self.best_state_dict = model_to_save.state_dict()
        self.network.to(self.args.device)

    def load_model(self, load_from_file=False, model_path=None):
        self.network.cpu()
        if load_from_file:
            best_state_dict = torch.load(model_path or self.args.save_path)
        else:
            best_state_dict = self.best_state_dict

        if hasattr(self.network, 'module'):
            self.network.module.load_state_dict(best_state_dict)
        else:
            self.network.load_state_dict(best_state_dict)

        self.network.to(self.args.device)


if __name__ == '__main__':
    score_file_path = '../score_file.txt'
    metrics = Metrics(score_file_path)
    results = metrics.evaluate_all_metrics()
    print(results)
