from typing import Dict

from torch import nn

from loader.bert_dataset import BertDataset
from loader.bert_init import BertInit
from loader.task_depot.pretrain_task import PretrainTask
from utils.time_printer import printer as print


class PretrainDepot:
    def __init__(self, dataset: BertDataset, bert_init: BertInit, device):
        self.dataset = dataset
        self.bert_init = bert_init
        self.device = device

        self.depot = dict()  # type: Dict[str, PretrainTask]

    def register(self, *tasks: PretrainTask):
        for task in tasks:
            self.depot[task.name] = task
            task.init(
                dataset=self.dataset,
                bert_init=self.bert_init,
                device=self.device,
            )
        return self

    def __getitem__(self, item):
        return self.depot[item]

    def get_extra_modules(self):
        extra_modules = dict()
        print('[CREATE extra modules]')
        for task_name in self.depot:
            extra_module = self.depot[task_name].init_extra_module()
            extra_modules[task_name] = extra_module
        return nn.ModuleDict(extra_modules)
