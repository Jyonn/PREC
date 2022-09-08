import torch
from torch import nn
from transformers import BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from loader.task_depot.pretrain_task import PretrainTask, TaskLoss
from utils.time_printer import printer as print


class ClassificationModule(nn.Module):
    def __init__(self, config: BertConfig, vocab_size):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        self.decoder = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.decoder(pooled_output)
        return pooled_output


class CategorizeTask(PretrainTask):
    def __init__(self, cat_col):
        super().__init__(name='categorize')

        self.loss_fct = nn.CrossEntropyLoss()
        self.cat_col = cat_col

    def rebuild_batch(self, batch):
        return batch

    def _init_extra_module(self):
        print('[IN CATEGORIZE TASK]')
        vocab = self.depot.get_vocab(self.cat_col)
        vocab_size = self.depot.get_vocab_size(vocab, as_vocab=True)
        return ClassificationModule(self.bert_init.bert_config, vocab_size)

    def _get_seg_embedding(self, matrix: torch.Tensor, table: nn.Embedding):
        return table(matrix)

    def produce_output(self, bert_output: BaseModelOutputWithPoolingAndCrossAttentions, **kwargs):
        return bert_output.last_hidden_state

    def calculate_loss(self, batch, output, **kwargs):
        cat_labels = batch['append_info'][self.cat_col].to(self.device)  # type: torch.Tensor
        loss = self.loss_fct(
            self.extra_module(output),
            cat_labels
        )
        return TaskLoss(loss=loss)
