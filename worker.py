import argparse
import os
import random
from typing import Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

# from loader.bert_aggregator import BertAggregator
from loader.data import Data
from loader.task_depot.pretrain_task import PretrainTask
from model.auto_bert import AutoBert
from utils.config_initializer import init_config
from utils.gpu import GPU
from utils.random_seed import seeding
from utils.time_printer import printer as print
from utils.logger import Logger


class Worker:
    def __init__(self, project_args, project_exp, cuda=None):
        self.args = project_args
        self.exp = project_exp

        self.logging = Logger(self.args.store.log_path, print)

        self.device = self.get_device(cuda)

        self.data = Data(
            project_args=self.args,
            project_exp=self.exp,
            device=self.device,
        )

        self.model = AutoBert(
            device=self.device,
            bert_init=self.data.bert_init,
            pretrain_depot=self.data.pretrain_depot,
        )

        self.model.to(self.device)
        self.logging(self.model.bert.config)
        self.save_model = self.model

        if self.exp.mode == 'export':
            self.m_optimizer = self.m_scheduler = None
        else:
            self.m_optimizer = torch.optim.Adam(
                params=filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.exp.policy.lr
            )
            self.m_scheduler = get_linear_schedule_with_warmup(
                self.m_optimizer,
                num_warmup_steps=self.exp.policy.n_warmup,
                num_training_steps=len(self.data.t_set) // self.exp.policy.batch_size * self.exp.policy.epoch,
            )

            print('training params')
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    print(name, p.data.shape)

        self.attempt_loading()

    @staticmethod
    def get_device(cuda):
        if cuda == -1:
            return 'cpu'
        if not cuda:
            return GPU.auto_choose(torch_format=True)
        return "cuda:{}".format(cuda)

    def attempt_loading(self):
        if self.exp.load.load_ckpt is not None:
            load_path = os.path.join(self.args.store.save_dir, self.exp.load.load_ckpt)
            self.logging("load model from exp {}".format(load_path))
            state_dict = torch.load(load_path, map_location=self.device)

            if '__rec__' in state_dict:
                model_ckpt = state_dict['model']
            else:
                model_ckpt = state_dict

            self.save_model.load_state_dict(model_ckpt, strict=not self.exp.load.relax_load)
            load_status = False
            if '__rec__' in state_dict:
                if self.exp.mode != 'export' and not self.exp.load.load_model_only:
                    load_status = True
                    self.m_optimizer.load_state_dict(state_dict['optimizer'])
                    self.m_scheduler.load_state_dict(state_dict['scheduler'])
            print('Load optimizer and scheduler:', load_status)

    def log_interval(self, epoch, step, task: PretrainTask, loss):
        self.logging(
            "epoch {}, step {}, "
            "task {}, "
            "loss {:.4f}".format(
                epoch,
                step + 1,
                task.name,
                loss.item()
            ))

    def train(self):
        self.logging('Start Training')
        tasks = self.exp.tasks
        tasks = [self.data.pretrain_depot[task] if isinstance(task, str) else task for task in tasks]

        train_steps = len(self.data.t_set) // self.exp.policy.batch_size
        accumulate_step = 0
        assert self.exp.policy.accumulate_batch >= 1

        self.m_optimizer.zero_grad()
        for epoch in range(self.exp.policy.epoch_start, self.exp.policy.epoch + self.exp.policy.epoch_start):
            self.model.train()
            task = random.choice(tasks)
            t_loader = self.data.get_t_loader(task)

            for step, batch in enumerate(tqdm(t_loader)):
                task_output = self.model(
                    batch=batch,
                    task=task,
                )

                loss = task.calculate_loss(batch, task_output)

                loss.loss.backward()

                accumulate_step += 1
                if accumulate_step == self.exp.policy.accumulate_batch:
                    self.m_optimizer.step()
                    self.m_scheduler.step()
                    self.m_optimizer.zero_grad()
                    accumulate_step = 0

                if self.exp.policy.check_interval:
                    if self.exp.policy.check_interval < 0:  # step part
                        if (step + 1) % max(train_steps // (-self.exp.policy.check_interval), 1) == 0:
                            self.log_interval(epoch, step, task, loss.loss)
                    else:
                        if (step + 1) % self.exp.policy.check_interval == 0:
                            self.log_interval(epoch, step, task, loss.loss)

            print('end epoch')
            avg_loss = self.dev(task=task)
            self.logging("epoch {} finished,"
                         "task {}, "
                         "loss {:.4f}".format(epoch, task.name, avg_loss))

            if (epoch + 1) % self.exp.policy.store_interval == 0:
                epoch_path = os.path.join(self.args.store.ckpt_path, 'epoch_{}.bin'.format(epoch))
                state_dict = dict(
                    model=self.model.state_dict(),
                    optimizer=self.m_optimizer.state_dict(),
                    scheduler=self.m_scheduler.state_dict(),
                    __rec__=True
                )
                torch.save(state_dict, epoch_path)
        self.logging('Training Ended')

    def dev(self, task: Union[str, PretrainTask]):
        if isinstance(task, str):
            task = self.data.pretrain_depot[task]

        avg_loss = torch.tensor(.0).to(self.device)
        self.model.eval()
        d_loader = self.data.get_d_loader(task)
        for step, batch in enumerate(tqdm(d_loader)):
            with torch.no_grad():
                task_output = self.model(
                    batch=batch,
                    task=task,
                )

                loss = task.calculate_loss(batch, task_output)
                avg_loss += loss.loss

        avg_loss /= len(self.data.d_set) / self.exp.policy.batch_size

        return avg_loss.item()

    def export(self):
        # bert_aggregator = BertAggregator(
        #     layers=self.exp.save.layers,
        #     layer_strategy=self.exp.save.layer_strategy,
        #     union_strategy=self.exp.save.union_strategy,
        # )
        features = torch.zeros(
            self.data.depot.sample_size,
            self.args.bert_config.hidden_size,
            dtype=torch.float
        ).to(self.device)

        print(self.exp.save.layer_strategy)

        for loader in [self.data.get_t_loader(self.data.non_task), self.data.get_d_loader(self.data.non_task)]:
            for batch in tqdm(loader):
                with torch.no_grad():
                    task_output = self.model(batch=batch, task=self.data.non_task)
                    task_output = task_output.last_hidden_state.detach()  # [B, S, D]
                    if self.exp.save.layer_strategy == 'mean':
                        attention_sum = batch['attention_mask'].to(self.device).sum(-1).unsqueeze(-1).repeat(1, 1, self.args.bert_config.hidden_size)
                        attention_mask = batch['attention_mask'].to(self.device).unsqueeze(-1).repeat(1, 1, self.args.bert_config.hidden_size)
                        features[batch['append_info'][self.exp.save.key]] = (task_output * attention_mask).sum(1) / attention_sum
                    elif self.exp.save.layer_strategy == 'cls':
                        features[batch['append_info'][self.exp.save.key]] = task_output[:, 0, :]

        save_path = os.path.join(self.args.store.ckpt_path, self.exp.save.feature_path)
        np.save(save_path, features.cpu().numpy(), allow_pickle=False)

    def run(self):
        if self.exp.mode == 'train':
            self.train()
        elif self.exp.mode == 'dev':
            loss = self.dev(task='mlm')
            print("dev loss {:.4f}".format(loss))
        elif self.exp.mode == 'export':
            self.export()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--exp', type=str)
    parser.add_argument('--cuda', type=int, default=None)

    args = parser.parse_args()

    config, exp = init_config(args.config, args.exp)

    seeding(2021)

    worker = Worker(project_args=config, project_exp=exp, cuda=args.cuda)
    worker.run()
