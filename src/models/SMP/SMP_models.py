import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool as gap, global_mean_pool as gmp
from torch_geometric.nn import MLP
from src.models.SMP.network_layers import SCLayer

class BaseNet(nn.Module):
    """
        Base class for neural network models designed for processing simplicial complexes.

        Attributes:
            n_edge_features (int): Number of features per edge in the simplicial complex.
            in_node_features (int): Number of features per node in the simplicial complex.
            hidden_dim (int): Dimensionality of hidden layers.
            hidden_layers (int): Number of hidden layers.
            n_output_features (int): Number of output features.
            dense_channels (list): List of channel sizes for dense layers.
            kappa (int): Order of the Filter.
            p_dropout (float): Dropout probability.
            act (nn.Module): Activation function.
        """
    def __init__(self, n_edge_features, hidden_dim, hidden_layers,n_output_features, in_node_features=0, kappa=3, p_dropout=0.0):
        super().__init__()
        self.n_edge_features = n_edge_features
        self.in_node_features = in_node_features
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.n_output_features = n_output_features
        self.dense_channels = [hidden_dim * 2, hidden_dim, n_output_features]
        self.kappa = kappa
        self.p_dropout = p_dropout
        self.act = nn.ELU()

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, data):
        raise NotImplementedError

    def aggr_x(self, x_dict):
        return x_dict['z_low'] + x_dict['z_up'] + x_dict['z_har']


class CSCN(BaseNet):
    """
    Contrastive Simplicial model for processing simplicial complexes.
    Inherits from BaseNet and implements specific layers and forward logic for simplicial convolution.

    Attributes:
        convs (nn.ModuleList): List of convolutional layers tailored for simplicial complexes.
        orientation_invariant (bool): Flag for orientation invariance in output (relevant for simplicial complexes,
                                      implemented by taking the absolute value of the embedding values).
        readout (function): Readout function for pooling global simplex representations.
        act (nn.Module): Activation function.
        project (torch.nn.Sequential, optional): Projection head for feature transformation.
    """
    def __init__(self, n_edge_features, hidden_dim, hidden_layers, n_output_features, kappa=2,
                 p_dropout=0.0, projection_head_dim=False, bias=True, orientation_invariant=False):
        super().__init__(n_edge_features, hidden_dim, hidden_layers,n_output_features, kappa, p_dropout)

        self.convs = nn.ModuleList([])
        self.orientation_invariant = orientation_invariant
        self.convs.append(SCLayer(n_edge_features, hidden_dim, kappa, bias=bias))
        for _ in range(hidden_layers - 1):
            self.convs.append(SCLayer(hidden_dim, hidden_dim, kappa, bias=bias))
        self.convs.append(SCLayer(hidden_dim, n_output_features, kappa, bias=bias))
        self.readout = gmp
        self.act = nn.Tanh()

        if projection_head_dim:
            self.project = torch.nn.Sequential(
                nn.Linear(n_output_features, projection_head_dim),
                nn.ReLU(inplace=True),
                nn.Linear(projection_head_dim, projection_head_dim))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, lower_index, lower_values, upper_index, upper_values, features_batch):

        for i in range(self.hidden_layers):
            x = self.convs[i](x, lower_index, upper_index, lower_values, upper_values)
            x = self.aggr_x(x)
            x = self.act(x)

        x = self.convs[-1](x, lower_index, upper_index, lower_values, upper_values)
        x = self.aggr_x(x)
        x = self.act(x)

        if self.orientation_invariant:
            x = abs(x)

        global_rep = self.readout(x, batch=features_batch)

        return global_rep


class SupSCN(BaseNet):
    def __init__(self, n_edge_features, hidden_dim, hidden_layers, n_output_features, kappa=3,
                 p_dropout=0.0,batch_norm=True):
        """
            SupSCN model for supervised learning on simplicial complexes.

            Inherits from BaseNet and implements specific layers, batch normalization, and pooling for simplicial complexes.

            Attributes:
                batch_norm (bool): Flag to use batch normalization.
                convs (nn.ModuleList): List of convolutional layers for simplicial complexes.
                bn (nn.ModuleList): List of batch normalization layers.
                out_dense (MLP): Output dense layer for the final representation.
            """
        super().__init__(n_edge_features=n_edge_features, hidden_dim=hidden_dim, hidden_layers=hidden_layers,
                         n_output_features=n_output_features, kappa=kappa, p_dropout=p_dropout)

        self.batch_norm = batch_norm
        self.convs = nn.ModuleList([])
        if self.batch_norm:
            self.bn = nn.ModuleList([])

        self.convs.append(SCLayer(n_edge_features, hidden_dim, kappa))
        if self.batch_norm:
            self.bn.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(hidden_layers - 1):
            self.convs.append(SCLayer(hidden_dim, hidden_dim, kappa))
            if self.batch_norm:
                self.bn.append(nn.BatchNorm1d(hidden_dim))
        self.out_dense = MLP(self.dense_channels, dropout=p_dropout)

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batch_norm:
            for bn in self.bn:
                bn.reset_parameters()
        self.out_dense.reset_parameters()

    def forward(self, x, lower_edge_index, lower_edge_values,
                upper_edge_index, upper_edge_values, batch_index):

        for i in range(self.hidden_layers):
            x = self.convs[i](x, lower_edge_index, upper_edge_index, lower_edge_values, upper_edge_values)
            x = self.aggr_x(x)
            if self.batch_norm:
                x = self.bn[i](x)
            x = self.act(x)

        x = self.act(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
        return self.out_dense(x)
