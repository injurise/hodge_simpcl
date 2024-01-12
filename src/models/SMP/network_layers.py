import torch
from torch import Tensor, nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros, glorot
from torch_geometric.typing import (
    Adj,
    OptTensor,
)
from torch_sparse import SparseTensor, matmul

##### Adjusted from https://github.com/domenicocinque/spm

class SMConv(MessagePassing):
    """
    Convolutional layer for simplicial complexes.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        K (int, optional): The size of the neighborhood to consider in the convolution.
        bias (bool, optional): If set to `False`, the layer will not learn an additive bias.
        normalize (bool, optional): Whether to apply GCN normalization.
    """
    def __init__(self, in_channels: int, out_channels: int, K: int = 3,
                 bias: bool = True, normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.normalize = normalize

        # Linear transformations for filter order
        self.lins = torch.nn.ModuleList([Linear(in_channels, out_channels, bias=False) for _ in range(K + 1)])

        # Optional bias parameter
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        if self.normalize:
            if isinstance(edge_index, Tensor):
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    improved=False, add_self_loops=False, dtype=x.dtype)

            elif isinstance(edge_index, SparseTensor):
                edge_index = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, x.size(self.node_dim),
                    add_self_loops=False, dtype=x.dtype)

        out = self.lins[0](x)
        for lin in self.lins[1:]:
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            out += lin.forward(x)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={self.K})')


class SCLayer(nn.Module):
    """
     Layer for handling lower and upper neigbourhoods in simplicial complexes.

     Args:
         in_channels (int): Number of input channels.
         hidden_channels (int): Number of hidden channels.
         K (int, optional): The size of the neighborhood for the convolution.
         bias (bool, optional): If set to `False`, the layer will not learn an additive bias.
     """
    def __init__(self, in_channels, hidden_channels, K=3, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.K = K

        self.lower_conv = SMConv(in_channels, hidden_channels, K, normalize=False, bias=bias)
        self.upper_conv = SMConv(in_channels, hidden_channels, K, normalize=False, bias=bias)
        self.harmonic = nn.Linear(in_channels, hidden_channels, bias=False)

    def reset_parameters(self):
        self.lower_conv.reset_parameters()
        self.upper_conv.reset_parameters()
        self.harmonic.reset_parameters()

    def forward(self, x, lower_index, upper_index, lower_values=None, upper_values=None, sum_components=False):
        z_low = self.lower_conv(x, edge_index=lower_index, edge_weight=lower_values)
        z_up = self.upper_conv(x, edge_index=upper_index, edge_weight=upper_values)
        z_har = self.harmonic(x)

        if sum_components:
            return z_low + z_up + z_har
        else:
            return {'z_low': z_low, 'z_up': z_up, 'z_har': z_har}


class Harm_Lower_SCLayer(SCLayer):
    def __init__(self, in_channels, hidden_channels, normalize, K=3):
        super().__init__(in_channels, hidden_channels, K)
        self.lower_conv = SMConv(in_channels, hidden_channels, K, normalize=normalize)
        self.harmonic = nn.Linear(in_channels, hidden_channels, bias=False)

    def forward(self, x, lower_index, upper_index, lower_values=None, upper_values=None, sum_components=False):
        z_low = self.lower_conv(x, edge_index=lower_index, edge_weight=lower_values)
        z_har = self.harmonic(x)

        if sum_components:
            return z_low + z_har
        else:
            return {'z_low': z_low, 'z_up': 0, 'z_har': z_har}


class Lower_SCLayer(SCLayer):
    def __init__(self, in_channels, hidden_channels, normalize, K=3):
        super().__init__(in_channels, hidden_channels, K)
        self.lower_conv = SMConv(in_channels, hidden_channels, K, normalize=normalize)

    def forward(self, x, lower_index, upper_index, lower_values=None, upper_values=None, sum_components=False):
        z_low = self.lower_conv(x, edge_index=lower_index, edge_weight=lower_values)

        if sum_components:
            return z_low
        else:
            return {'z_low': z_low, 'z_up': 0, 'z_har': 0}
