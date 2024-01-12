# This whole folder and code structure is taken and inspired from https://github.com/PyGCL/PyGCL/blob/main/GCL/augmentors/augmentor.py

from __future__ import annotations

import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple, List



class SAugmentor(ABC):
    """Base class for simplcial augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, x,OP1,OP2,num_n, num_e, num_t):
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(self,x,OP1,OP2,num_n, num_e, num_t,index):
        return self.augment(x,OP1,OP2,num_n, num_e, num_t,index)

class Compose(SAugmentor):
    def __init__(self, augmentors: List[SAugmentor]):
        super(Compose, self).__init__()
        self.augmentors = augmentors

    def augment(self, x,OP1,OP2,num_n, num_e, num_t,index):
        for aug in self.augmentors:
            x,OP1,OP2,num_n, num_e, num_t,index = aug.augment(x,OP1,OP2,num_n, num_e, num_t,index)
        return x,OP1,OP2,num_n, num_e, num_t,index


class RandomChoice(SAugmentor):
    def __init__(self, augmentors: List[SAugmentor], num_choices: int):
        super(RandomChoice, self).__init__()
        assert num_choices <= len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, batch):
        num_augmentors = len(self.augmentors)
        perm = torch.randperm(num_augmentors)
        idx = perm[:self.num_choices]
        for i in idx:
            aug = self.augmentors[i]
            batch = aug.augment(batch)
        return batch


class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class GAugmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight)).unfold()


class GCompose(GAugmentor):
    def __init__(self, augmentors: List[GAugmentor]):
        super(GCompose, self).__init__()
        self.augmentors = augmentors

    def augment(self, g: Graph) -> Graph:
        for aug in self.augmentors:
            g = aug.augment(g)
        return g


class GRandomChoice(GAugmentor):
    def __init__(self, augmentors: List[GAugmentor], num_choices: int):
        super(GRandomChoice, self).__init__()
        assert num_choices <= len(augmentors)
        self.augmentors = augmentors
        self.num_choices = num_choices

    def augment(self, g: Graph) -> Graph:
        num_augmentors = len(self.augmentors)
        perm = torch.randperm(num_augmentors)
        idx = perm[:self.num_choices]
        for i in idx:
            aug = self.augmentors[i]
            g = aug.augment(g)
        return g
