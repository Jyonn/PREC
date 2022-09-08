from UniTok import UniDep

from loader.bert_dataloader import BertDataLoader
from loader.bert_dataset import BertDataset
from loader.bert_init import BertInit
from loader.depot_filter import DepotFilter
from loader.embedding_init import EmbeddingInit
from loader.task_depot.align_task import AlignTask
from loader.task_depot.categorize_task import CategorizeTask
from loader.task_depot.mlm_task import MLMTask
from loader.task_depot.non_task import NonTask
from loader.task_depot.pretrain_depot import PretrainDepot
from loader.task_depot.pretrain_task import PretrainTask
from utils.splitter import Splitter
from utils.time_printer import printer as print


class Data:
    TRAIN = 'train'
    DEV = 'dev'

    def __init__(self,
                 project_args,
                 project_exp,
                 device,
                 ):
        self.args = project_args
        self.exp = project_exp
        self.device = device

        self.depot = DepotFilter(self.args.store.data_dir)
        if 'filter' in self.args.data.d:
            print('Origin Depot', self.depot.sample_size)
            for col in self.args.data.filter.remove_empty:
                self.depot.remove_empty(col)
                print('Remove', col, self.depot.sample_size)

        self.splitter = Splitter().add(
            name=self.TRAIN,
            weight=self.args.data.split.train
        ).add(
            name=self.DEV,
            weight=self.args.data.split.dev
        )

        self.mlm_task = MLMTask(old_mask='old_mask' in self.exp.d, apply_cols=self.exp.apply_cols)
        self.cat_task = CategorizeTask(cat_col='cat')
        self.align_task = AlignTask()
        self.non_task = NonTask()
        self.tasks = [self.mlm_task, self.non_task, self.cat_task, self.align_task]
        # self.tasks = [self.mlm_task, self.non_task, self.align_task]
        # self.tasks = [self.mlm_task, self.non_task]

        expand_tokens = []
        for task in self.tasks:
            expand_tokens.extend(task.get_expand_tokens())
        print('Expand Tokens:', expand_tokens)

        # expand_tokens = []
        self.t_set = BertDataset(
            depot=self.depot,
            splitter=self.splitter,
            mode=self.TRAIN,
            order=self.args.data.order,
            append=self.args.data.append,
            expand_tokens=expand_tokens,
        )
        self.d_set = BertDataset(
            depot=self.depot,
            splitter=self.splitter,
            mode=self.DEV,
            order=self.args.data.order,
            append=self.args.data.append,
            expand_tokens=expand_tokens,
        )

        self.embedding_init = EmbeddingInit()
        for embedding_info in self.args.embedding:
            self.embedding_init.append(**embedding_info.d, global_freeze=self.exp.freeze_emb)

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
