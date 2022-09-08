from typing import Union, List, Callable

import torch


class BertAggregator:
    def __init__(
            self,
            layers: Union[int, List[int]] = 0,
            layer_strategy: Union[str, Callable] = 'mean',
            union_strategy: Union[str, Callable] = 'mean',
    ):
        self.layers = layers
        if isinstance(self.layers, int):
            self.layers = [self.layers]
        else:
            self.layers = list(self.layers)

        self.layer_strategy = layer_strategy
        if isinstance(self.layer_strategy, str):
            self.layer_strategy = self.layer_strategy.lower()
            if self.layer_strategy == 'mean':
                self.layer_strategy = lambda x: torch.mean(x, dim=1)
            elif self.layer_strategy == 'cls':
                self.layer_strategy = lambda x: x[:, 0, :]
            else:
                raise ValueError('Layer strategy should be mean or cls if it is a string')
        assert callable(self.layer_strategy), 'Layer strategy is not callable'

        self.union_strategy = union_strategy
        if isinstance(self.union_strategy, str):
            self.union_strategy = self.union_strategy.lower()
            if self.union_strategy == 'mean':
                self.union_strategy = lambda x: torch.mean(torch.stack(x), dim=0)
            elif self.union_strategy == 'sum':
                self.union_strategy = lambda x: torch.sum(torch.stack(x), dim=0)
            else:
                raise ValueError('Union strategy should be mean or sum if it is a string')
        assert callable(self.union_strategy)

    def __call__(self, hidden_states):
        layer_outputs = []
        for layer in self.layers:
            layer_outputs.append(self.layer_strategy(hidden_states[layer]))
        return self.union_strategy(layer_outputs)
