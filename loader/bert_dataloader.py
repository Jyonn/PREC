import random
from typing import List

from torch.utils.data import DataLoader

from loader.bert_dataset import BertDataset
from loader.task_depot.pretrain_task import PretrainTask


class BertDataLoader(DataLoader):
    def __init__(self, dataset: BertDataset, pretrain_tasks: List[PretrainTask], **kwargs):
        super().__init__(
            dataset=dataset,
            pin_memory=True,
            **kwargs
        )

        self.auto_dataset = dataset
        self.pretrain_tasks = pretrain_tasks

    def __iter__(self):
        iterator = super().__iter__()

        while True:
            try:
                batch = next(iterator)
                task = random.choice(self.pretrain_tasks)
                batch = task.rebuild_batch(batch)
                batch['task'] = task
                yield batch
            except StopIteration:
                return
