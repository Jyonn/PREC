import copy
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from UniTok import UniDep
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from utils.dictifier import Dictifier
from utils.metric import Metric


class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class DevDataset(Dataset):
    def __init__(self, dev_depot: UniDep):
        self.dev_depot = dev_depot
        self.df_stack = Dictifier(aggregator=torch.tensor)

    def __getitem__(self, index):
        dev_sample = self.dev_depot[index]
        samples = []
        for predict in dev_sample['predict']:
            samples.append(dict(
                uid=dev_sample['uid'],
                nid=predict[0],
                labels=predict[1],
                imp=dev_sample['imp'],
            ))
        return self.df_stack(samples)

    def __len__(self):
        return self.dev_depot.sample_size


class Data:
    def __init__(self, config):
        self.config = config
        os.makedirs(self.config.data.store_dir, exist_ok=True)

        self.user_depot = UniDep(self.config.data.user)
        self.news_depot = UniDep(self.config.data.news)
        self.train_depot = UniDep(self.config.data.train)
        self.dev_depot = UniDep(self.config.data.dev)

        self.train_data = []
        for user_sample in self.user_depot:
            for nid in user_sample['history']:
                self.train_data.append(dict(
                    uid=user_sample['uid'],
                    nid=nid,
                    labels=1
                ))

        for train_sample in self.train_depot:
            for predict in train_sample['predict']:
                self.train_data.append(dict(
                    uid=train_sample['uid'],
                    nid=predict[0],
                    labels=predict[1]
                ))

        self.map = dict()
        for sample in self.train_data:
            if sample['uid'] not in self.map:
                self.map[sample['uid']] = set()
            self.map[sample['uid']].add(sample['nid'])

        self.train_set = TrainDataset(self.train_data)
        self.dev_set = DevDataset(self.dev_depot)
        self.train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.config.train.batch_size,
            shuffle=True,
        )

        self.df_stack = Dictifier(aggregator=torch.tensor)
        self.df_concat = Dictifier(aggregator=torch.cat)

    def get_negative_samples(self, uid, count=1):
        return [self.get_negative_sample(uid) for _ in range(count)]

    def get_negative_sample(self, uid):
        num_item_types = self.news_depot.sample_size

        sample = dict(uid=uid, labels=0)
        while True:
            nid = random.randint(0, num_item_types - 1)
            if uid not in self.map or nid not in self.map[uid]:
                sample['nid'] = nid
                break
        return sample

    def append_batch_negative_samples(self, batch, count=1):
        samples = []
        for uid in batch['uid']:
            samples.extend(self.get_negative_samples(uid=uid, count=count))
        neg_batch = self.df_stack(samples)
        return self.df_concat([batch, neg_batch])


class DevResult:
    def __init__(self, n_ndcg):
        self.n_ndcg = n_ndcg
        self.metric = Metric()

    @staticmethod
    def get_dcg(outputs, labels, k):
        df = pd.DataFrame({"outputs": outputs, "labels": labels})
        df = df.sort_values(by="outputs", ascending=False)
        df = df.iloc[:k, :]
        dcg = (2 ** df["labels"] - 1) / np.log2(np.arange(1, df["labels"].count() + 1) + 1)
        return np.sum(dcg)

    def get_ndcg(self, outputs, labels, k):
        dcg = self.get_dcg(outputs, labels, k)
        idcg = self.get_dcg(labels, labels, k)
        ndcg = dcg / idcg
        return ndcg

    def get_metric(self, outputs, labels):
        for k in self.n_ndcg:
            metric_key = 'n@{}'.format(k)
            self.metric.append(metric_key, self.get_ndcg(outputs, labels, k))
        self.metric.append('auc', roc_auc_score(labels, outputs))
