import copy
from typing import Dict

import numpy as np
import torch
from torch import nn
from transformers import BertConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from loader.task_depot.pretrain_task import PretrainTask, TaskLoss
from utils.time_printer import printer as print


class ClassificationModule(nn.Module):
    def __init__(self, config: BertConfig, vocab_size):
        super(ClassificationModule, self).__init__()
        self.transform = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class MLMTask(PretrainTask):
    def __init__(self, select_prob=0.15, mask_prob=0.8, random_prob=0.1, loss_pad=-100, apply_cols=None, old_mask=False):
        super(MLMTask, self).__init__(name='mlm')
        self.select_prob = select_prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self.loss_pad = loss_pad
        self.apply_cols = apply_cols  # type: list
        self.old_mask = old_mask

        self.loss_fct = nn.CrossEntropyLoss()

    def get_expand_tokens(self):
        return ['MASK'] if self.old_mask else ['MASK_{col}']

    def do_mask(self, mask, tok, vocab_size):
        tok = int(tok)
        if np.random.uniform() < self.select_prob:
            mask_type = np.random.uniform()
            if mask_type < self.mask_prob:
                return mask, tok, True
            elif mask_type < self.mask_prob + self.random_prob:
                return np.random.randint(vocab_size), tok, False
            return tok, tok, False
        return tok, self.loss_pad, False

    def rebuild_batch(self, batch):
        input_ids = batch['input_ids']  # type: torch.Tensor
        col_mask = batch['col_mask']  # type: Dict[str, torch.Tensor]
        batch_size = int(input_ids.shape[0])

        mask_labels = torch.ones(batch_size, self.dataset.max_sequence, dtype=torch.long) * -100
        batch['mask_labels_col'] = copy.deepcopy(col_mask)

        for col_name in self.depot.col_info.d:
            if col_name not in col_mask:
                continue

            if self.apply_cols and col_name not in self.apply_cols:
                continue

            vocab_size = self.depot.get_vocab_size(col_name)

            for i_batch in range(batch_size):
                for i_tok in range(self.dataset.max_sequence):
                    if col_mask[col_name][i_batch][i_tok]:
                        input_id, mask_label, use_special_col = self.do_mask(
                            mask=self.dataset.TOKENS['MASK'] if self.old_mask else self.dataset.TOKENS[f'MASK_{col_name}'],
                            tok=input_ids[i_batch][i_tok],
                            vocab_size=vocab_size
                        )
                        input_ids[i_batch][i_tok] = input_id
                        mask_labels[i_batch][i_tok] = mask_label
                        if use_special_col:
                            col_mask[col_name][i_batch][i_tok] = 0
                            col_mask[self.dataset.special_id][i_batch][i_tok] = 1
        batch['mask_labels'] = mask_labels
        return batch

    def _init_extra_module(self):
        module_dict = dict()
        print('[IN MLM TASK]')
        for col_name in self.dataset.order:
            vocab = self.depot.col_info.d[col_name].vocab
            if vocab in module_dict:
                print('Escape create modules for', col_name, '(', vocab, ')')
                continue
            vocab_size = self.depot.get_vocab_size(vocab, as_vocab=True)
            module_dict[vocab] = ClassificationModule(self.bert_init.bert_config, vocab_size)
            print('Classification Module for', col_name, '(', vocab, ')', 'with vocab size', vocab_size)
        return nn.ModuleDict(module_dict)

    def _get_seg_embedding(self, matrix: torch.Tensor, table: nn.Embedding):
        return table(matrix)

    def produce_output(self, bert_output: BaseModelOutputWithPoolingAndCrossAttentions, **kwargs):
        last_hidden_state = bert_output.last_hidden_state
        output_dict = dict()
        for col_name in self.dataset.order:
            vocab = self.depot.col_info.d[col_name].vocab
            classification_module = self.extra_module[vocab]
            output_dict[col_name] = classification_module(last_hidden_state)
        return output_dict

    def calculate_loss(self, batch, output, **kwargs):
        mask_labels_col = batch['mask_labels_col']
        mask_labels = batch['mask_labels'].to(self.device)  # type: torch.Tensor

        total_loss = torch.tensor(0, dtype=torch.float).to(self.device)
        for col_name in mask_labels_col:
            if col_name == self.dataset.special_id:
                continue
            col_mask = mask_labels_col[col_name].to(self.device)  # type: torch.Tensor
            col_labels = torch.mul(col_mask, mask_labels) + \
                         torch.ones(mask_labels.shape, dtype=torch.long).to(self.device) * (col_mask - 1) * 100
            col_labels = col_labels.view(-1).to(self.device)
            vocab_size = self.depot.get_vocab_size(col_name)
            loss = self.loss_fct(
                output[col_name].view(-1, vocab_size),
                col_labels
            )
            total_loss += loss
        return TaskLoss(loss=total_loss)
