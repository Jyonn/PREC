from multiprocessing import Pool
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch


class Metric:
    def __init__(self, groups, scores, labels):
        self.metrics = dict()  # type: Dict[str, Union[list, float]]

        df = pd.DataFrame(dict(groups=groups, scores=scores, labels=labels))
        self.groups = df.groupby('groups')

    def append(self, key, value):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)

    def extend(self, key, values):
        if key not in self.metrics:
            self.metrics[key] = []

        self.metrics[key].extend(values)

    def aggregate(self):
        for key in self.metrics:
            self.metrics[key] = torch.tensor(self.metrics[key]).mean().item()

    def __setitem__(self, key, value):
        self.metrics[key] = value

    def __getitem__(self, key):
        return self.metrics[key]

    def __iter__(self):
        for key in self.metrics:
            yield key

    def pool_metric(self, key, handler, num_workers=5):
        tasks = []
        pool = Pool(processes=num_workers)
        for g in self.groups:
            group = g[1]
            tasks.append(pool.apply_async(handler, args=(group.labels.tolist(), group.scores.tolist())))
        pool.close()
        pool.join()

        self.extend(key, [task.get() for task in tasks])
