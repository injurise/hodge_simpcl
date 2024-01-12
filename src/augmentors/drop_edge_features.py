from src.augmentors.augmentor import SAugmentor
import torch
from torch.distributions.bernoulli import Bernoulli

class DropEdgeFeatures(SAugmentor):
    """
        Augmentor to drop edge features in a graph uniformly at random.

        Args:
            pkeep (float): Probability of keeping an edge feature.
            full_edge_drop (bool, optional): If True, drops all features of an edge.
                                             If False, drops individual features independently.
        """
    def __init__(self, pkeep: float, full_edge_drop: bool = True):
        super(DropEdgeFeatures, self).__init__()
        self.keep_prob = pkeep
        self.full_edge_drop = full_edge_drop

    def augment(self, x, B1, B2, num_n, num_e, num_t, index):
        # Duplicate edge features and incidence matrices to avoid modifying the original tensors
        features, B1_cor, B2_cor = x.detach().clone(), B1.detach().clone(), B2.detach().clone()
        if self.full_edge_drop:
            # Dropping entire rows (all features of an edge) based on keep probability
            row_indices = torch.nonzero(features, as_tuple=True)[0].unique()
            probs = torch.zeros(row_indices.shape[0]) + self.keep_prob
            dist = Bernoulli(probs)
            samples = dist.sample().view(row_indices.shape[0], 1)
            if torch.cuda.is_available():
                samples = samples.to('cuda')
            features[row_indices] = features[row_indices] * (samples).repeat(1, features.shape[1])

        else:
            # Dropping individual features independently
            nonzero_indices = torch.nonzero(features,as_tuple=True)
            probs = torch.zeros(nonzero_indices[0].shape[0]) + self.keep_prob
            dist = Bernoulli(probs)
            samples = dist.sample()
            if torch.cuda.is_available():
                samples = samples.to('cuda')
            features[nonzero_indices] = features[nonzero_indices] * samples

        return features, B1_cor, B2_cor, num_n, num_e, num_t, index
