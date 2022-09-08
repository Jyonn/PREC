import os

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, BertForMaskedLM, BertConfig, BertModel, load_tf_weights_in_bert


class EmbeddingInit:
    def __init__(self):
        self.embedding_dict = dict()

    def append(self, vocab_name, vocab_type, path, freeze, global_freeze=False):
        print(vocab_name, freeze, global_freeze)
        self.embedding_dict[vocab_name] = dict(
            vocab_name=vocab_name,
            vocab_type=vocab_type,
            path=path,
            embedding=None,
            freeze=freeze or global_freeze,
        )
        return self

    @staticmethod
    def get_numpy_embedding(path):
        embedding = np.load(path)
        assert isinstance(embedding, np.ndarray)
        return torch.tensor(embedding)

    @staticmethod
    def get_bert_torch_embedding(path):
        bert_for_masked_lm = AutoModelForMaskedLM.from_pretrained(path)  # type: BertForMaskedLM
        bert = bert_for_masked_lm.bert
        return bert.embeddings.word_embeddings.weight

    @staticmethod
    def get_bert_tf_embedding(path):
        config = BertConfig.from_json_file(os.path.join(path, 'bert_config.json'))
        bert = BertModel(config)
        load_tf_weights_in_bert(bert, config, os.path.join(path, 'bert_model.ckpt.index'))
        return bert.embeddings.word_embeddings.weight

    def get_embedding(self, vocab_name):
        if vocab_name not in self.embedding_dict:
            return None

        embedding_info = self.embedding_dict[vocab_name]
        if embedding_info['embedding'] is not None:
            return embedding_info['embedding']

        if hasattr(self, 'get_{}_embedding'.format(embedding_info['vocab_type'])):
            getter = getattr(self, 'get_{}_embedding'.format(embedding_info['vocab_type']))
            embedding_info['embedding'] = getter(embedding_info['path'])
            return embedding_info['embedding']

    def is_freezing(self, vocab_name):
        assert vocab_name in self.embedding_dict
        return self.embedding_dict[vocab_name]['freeze']
