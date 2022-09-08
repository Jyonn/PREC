from typing import Optional, Union

import torch
from torch import nn
from transformers import BertConfig

from loader.bert_dataset import BertDataset
from loader.embedding_init import EmbeddingInit
from utils.time_printer import printer as print


class BertInit:
    def __init__(
            self,
            dataset: BertDataset,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            embedding_init: EmbeddingInit = None,
            global_freeze: bool = False,
    ):
        self.dataset = dataset
        self.depot = self.dataset.depot
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.embedding_init = embedding_init
        self.global_freeze = global_freeze

        self._embedding_tables = None
        self._bert_config = None

    @property
    def bert_config(self):
        if self._bert_config:
            return self._bert_config
        self._bert_config = BertConfig(
            vocab_size=1,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.hidden_size * 4,
            max_position_embeddings=self.dataset.max_sequence,
            type_vocab_size=self.dataset.token_types,
        )
        return self._bert_config

    def get_embedding_tables(self):
        if self._embedding_tables:
            return self._embedding_tables

        embedding_tables = dict()
        required_vocabs = set()
        for col_name in self.dataset.order:
            required_vocabs.add(self.dataset.depot.col_info.d[col_name].vocab)

        print('global freeze:', self.global_freeze)

        for vocab in required_vocabs:
            embedding = self.embedding_init.get_embedding(vocab)  # type: Optional[torch.Tensor]
            if embedding is not None:
                print('load', vocab, '( require_grad =', not self.embedding_init.is_freezing(vocab), '), embedding with shape', embedding.shape,
                      'and the expected shape is', self.depot.get_vocab_size(vocab, as_vocab=True), 'x', self.hidden_size)
                assert embedding.shape == (self.depot.get_vocab_size(vocab, as_vocab=True), self.hidden_size)
                embedding_tables[vocab] = nn.Embedding.from_pretrained(embedding)
                embedding_tables[vocab].weight.requires_grad = not self.embedding_init.is_freezing(vocab)
            else:
                print('create', vocab, '( require_grad =', not self.global_freeze, '), embedding with shape', self.depot.get_vocab_size(vocab, as_vocab=True), 'x', self.hidden_size)
                embedding_tables[vocab] = nn.Embedding(
                    num_embeddings=self.depot.get_vocab_size(vocab, as_vocab=True),
                    embedding_dim=self.hidden_size
                )
                embedding_tables[vocab].weight.requires_grad = not self.global_freeze

        print('create', self.dataset.special_id, 'embedding with shape', len(self.dataset.special_tokens), 'x', self.hidden_size)
        embedding_tables[self.dataset.special_id] = nn.Embedding(
            num_embeddings=len(self.dataset.special_tokens),
            embedding_dim=self.hidden_size
        )
        embedding_tables[self.dataset.special_id].weight.requires_grad = not self.global_freeze

        self._embedding_tables = nn.ModuleDict(embedding_tables)
        return self._embedding_tables
