from typing import Dict, Optional

import torch
from UniTok import UniDep
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from loader.bert_init import BertInit
from loader.bert_dataset import BertDataset


class TaskLoss:
    def __init__(self, loss):
        self.loss = loss

    def backward(self):
        self.loss.backward()


class PretrainTask:
    def __init__(self, name):
        self.name = name
        self.dataset = None  # type: Optional[BertDataset]
        self.depot = None  # type: Optional[UniDep]
        self.bert_init = None  # type: Optional[BertInit]
        self.device = None

        self.extra_module = None  # type: Optional[nn.ModuleDict]

    def __str__(self):
        return self.name

    @staticmethod
    def get_expand_tokens():
        return []

    def init(self, dataset: BertDataset, bert_init: BertInit, device):
        self.dataset = dataset
        self.depot = dataset.depot
        self.bert_init = bert_init
        self.device = device

    def init_extra_module(self):
        self.extra_module = self._init_extra_module()
        return self.extra_module

    def rebuild_batch(self, batch):
        raise NotImplementedError

    def _init_extra_module(self):
        raise NotImplementedError

    def _get_special_seg_embedding(self, matrix: torch.Tensor, table: nn.Embedding):
        return table(matrix)

    def _get_seg_embedding(self, matrix: torch.Tensor, table: nn.Embedding):
        raise NotImplementedError

    def get_embedding(self, batch, table_dict: Dict[str, nn.Embedding], embedding_size):
        input_ids = batch['input_ids'].to(self.device)  # type: torch.Tensor
        input_embeds = torch.zeros(*input_ids.shape, embedding_size, dtype=torch.float).to(self.device)

        for col_name in batch['col_mask']:
            col_mask = batch['col_mask'][col_name].to(self.device)  # type: torch.Tensor
            matrix = torch.mul(input_ids, col_mask)

            if col_name == self.dataset.special_id:
                table = table_dict[col_name]
                seg_embedding = self._get_special_seg_embedding(matrix, table).to(self.device)
            else:
                vocab = self.depot.col_info[col_name].vocab
                table = table_dict[vocab]
                seg_embedding = self._get_seg_embedding(matrix, table).to(self.device)
            col_mask = col_mask.unsqueeze(-1).repeat(1, 1, embedding_size).to(self.device)
            input_embeds += torch.mul(col_mask.float(), seg_embedding)

        return input_embeds

    def produce_output(self, bert_output: BaseModelOutputWithPoolingAndCrossAttentions, **kwargs):
        raise NotImplementedError

    def calculate_loss(self, batch, output, **kwargs) -> TaskLoss:
        raise NotImplementedError
