from typing import List

from UniTok.classify import Classify

from loader.bert_dataloader import BertDataLoader
from loader.bert_dataset import BertDataset
from loader.bert_init import BertInit
from loader.embedding_init import EmbeddingInit
from loader.task_depot.pretrain_depot import PretrainDepot
from loader.task_depot.pretrain_task import PretrainTask


class BertData:
    def __init__(
        self,
        project_args: Classify,
        project_exp: Classify,
        device: str,
        t_set: BertDataset,
        d_set: BertDataset,
    ):
        self.args = project_args
        self.exp = project_exp
        self.device = device

        self.tasks = []  # type: List[PretrainTask]

        self.embedding_init = EmbeddingInit()
        for embedding_info in self.args.embedding:
            self.embedding_init.append(**embedding_info.d, global_freeze=self.exp.freeze_emb)

        self.t_set = t_set  # type: BertDataset
        self.d_set = d_set  # type: BertDataset

        self.bert_init = BertInit(
            dataset=self.t_set,
            embedding_init=self.embedding_init,
            global_freeze=self.exp.freeze_emb,
            **self.args.bert_config.d,
        )

        self.pretrain_depot = PretrainDepot(
            dataset=self.t_set,
            bert_init=self.bert_init,
            device=self.device,
        ).register(*self.tasks)

    def get_t_loader(self, *tasks: PretrainTask):
        self.t_set.build_dataset()
        return BertDataLoader(
            dataset=self.t_set,
            pretrain_tasks=list(tasks),
            shuffle=self.args.data.shuffle,
            batch_size=self.exp.policy.batch_size,
        )

    def get_d_loader(self, *tasks: PretrainTask):
        return BertDataLoader(
            dataset=self.d_set,
            pretrain_tasks=list(tasks),
            shuffle=False,
            batch_size=self.exp.policy.batch_size,
        )
