import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from loader.task_depot.pretrain_task import PretrainTask


class NonTask(PretrainTask):
    def _init_extra_module(self):
        return None

    def init_parallel(self):
        pass

    def produce_output(self, bert_output: BaseModelOutputWithPoolingAndCrossAttentions, **kwargs):
        return bert_output

    def rebuild_batch(self, batch):
        return batch

    def _get_seg_embedding(self, matrix: torch.Tensor, table: nn.Embedding):
        return table(matrix)

    def calculate_loss(self, batch, output, **kwargs):
        pass

    def __init__(self):
        super(NonTask, self).__init__(name='non')
