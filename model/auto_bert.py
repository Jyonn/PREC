from typing import Union

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertModel

from loader.bert_init import BertInit
from loader.task_depot.pretrain_depot import PretrainDepot
from loader.task_depot.pretrain_task import PretrainTask

from utils.time_printer import printer as print


class AutoBert(nn.Module):
    bert_init: BertInit

    def __init__(
            self,
            bert_init: BertInit,
            device,
            pretrain_depot: PretrainDepot
    ):
        super(AutoBert, self).__init__()

        self.bert_init = bert_init
        self.device = device
        self.pretrain_depot = pretrain_depot

        self.hidden_size = self.bert_init.hidden_size

        self.bert = BertModel(self.bert_init.bert_config)  # use compatible code
        self.embedding_tables = self.bert_init.get_embedding_tables()
        self.extra_modules = self.pretrain_depot.get_extra_modules()
        print('Extra Modules', self.extra_modules)

    def forward(self, batch, task: Union[str, PretrainTask]):
        attention_mask = batch['attention_mask'].to(self.device)  # type: torch.Tensor # [B, S]
        segment_ids = batch['segment_ids'].to(self.device)  # type: torch.Tensor # [B, S]

        if isinstance(task, str):
            task = self.pretrain_depot[task]

        input_embeds = task.get_embedding(
            batch=batch,
            table_dict=self.embedding_tables,
            embedding_size=self.hidden_size,
        )

        bert_output = self.bert(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            token_type_ids=segment_ids,
            output_hidden_states=True,
            return_dict=True
        )

        return task.produce_output(bert_output)
