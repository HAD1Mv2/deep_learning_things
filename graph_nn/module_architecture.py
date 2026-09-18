from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import MLP, HeteroConv, SAGEConv
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm


# custom SAGEConv layer where it integrate edge features to source node features when message passing
class EdgeSAGEConv(MessagePassing):
    """Modification of GraphSAGE layer by incorporating edge features.

    The layer computes the output node features as

    .. math::
        \\mathbf{x}^{\\prime}_i = \\mathbf{W}_1 \\mathbf{x}_i +
        \\mathbf{W}_2 \\cdot \\mathrm{mean}_{j \\in \\mathcal{N}(i)}
        \\mathbf{x}_j

    If ``project=True``, the source node features are first projected via

    .. math::
        \\mathbf{x}_j \\leftarrow \\sigma(
        \\mathbf{W}_3 \\mathbf{x}_j + \\mathbf{b})

    as described in Eq. (3) of the paper. The projection is applied only to
    node features.

    Parameters
    ----------
    in_channels : int or tuple
        Size of each input sample, or ``-1`` to derive the size from the
        first input(s) to the ``forward`` method. A tuple corresponds to
        the source and target dimensionalities.
    edge_channels : int
        Size of the edge features. Use ``-1`` to derive the size from the
        first input(s) to the ``forward`` method.
    out_channels : int
        Size of each output sample.
    aggr : str or Aggregation, optional
        Aggregation scheme to use. Any aggregation from
        :obj:`torch_geometric.nn.aggr` can be used, such as ``"mean"``,
        ``"max"``, or ``"lstm"``. Default is ``"mean"``.
    normalize : bool, optional
        If ``True``, output features are :math:`\\ell_2`-normalized, i.e.,

        .. math::
            \\frac{\\mathbf{x}^{\\prime}_i}
            {\\|\\mathbf{x}^{\\prime}_i\\|_2}.

        Default is ``False``.
    root_weight : bool, optional
        If ``False``, transformed root node features are not added to the
        output. Default is ``True``.
    project : bool, optional
        If ``True``, a linear transformation followed by an activation
        function is applied before aggregation, as described in Eq. (3) of
        the paper. Default is ``False``.
    bias : bool, optional
        If ``False``, the layer does not learn an additive bias.
        Default is ``True``.
    **kwargs
        Additional arguments passed to
        :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes
    ------
    inputs : tuple
        The inputs consist of:

        - Node features:
        ``(|V|, F_in)`` or
        ``((|V_s|, F_s), (|V_t|, F_t))`` for bipartite graphs.
        - Edge indices: ``(2, |E|)``.
        - Edge features: ``(|E|, F_edge)``.

    outputs : torch.Tensor
        Node features with shape ``(|V|, F_out)`` or
        ``(|V_t|, F_out)`` for bipartite graphs.
    """
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        edge_channels: int,
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels 
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if in_channels[0] < 0 or edge_channels < 0:
            in_channels = (-1, -1)
        else:
            in_channels = (in_channels[0]+edge_channels, in_channels[1])


        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                f"support lazy initialization with "
                                f"`project=True`")
            self.lin = Linear(in_channels[0]-edge_channels, in_channels[0]-edge_channels, bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=False)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=bias)

        self.reset_parameters()


    def reset_parameters(self):
            super().reset_parameters()
            if self.project:
                self.lin.reset_parameters()
            self.lin_l.reset_parameters()
            if self.root_weight:
                self.lin_r.reset_parameters()


    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: Optional[Tensor] = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor, edge_attr: Optional[Tensor])
        out = self.propagate(edge_index, x=x, size=size, edge_attr=edge_attr)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_attr: Optional[Tensor]) -> Tensor:
            if isinstance(edge_attr, Tensor):
                msg = torch.cat([x_j, edge_attr], dim=-1)
            else:
                msg = x_j
            return msg

    def message_and_aggregate(self, adj_t: Adj, x: OptPairTensor) -> Tensor:
        if isinstance(adj_t, SparseTensor):
            adj_t = adj_t.set_value(None, layout=None)
        return spmm(adj_t, x[0], reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')


# the GNN part
class HeteroGNNEncoder(torch.nn.Module):

    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = HeteroConv({
                        ('user', 'rates', 'anime'): SAGEConv((-1, -1), hidden_channels),
                        ('anime', 'rev_rates', 'user'): SAGEConv((-1, -1), hidden_channels),
                    }, aggr='mean')
        self.conv2 = HeteroConv({
                        ('user', 'rates', 'anime'): SAGEConv((-1, -1), hidden_channels),
                        ('anime', 'rev_rates', 'user'): SAGEConv((-1, -1), hidden_channels),
                    }, aggr='mean')

    def forward(self, x_dict, edge_index_dict, *args, **kwargs):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict 

# the GNN part but with EdgeSAGEConv
class HeteroGNNEncoderEdge(torch.nn.Module):

    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = HeteroConv({
                        ('user', 'rates', 'anime'): EdgeSAGEConv((-1, -1), -1, hidden_channels),
                        ('anime', 'rev_rates', 'user'): EdgeSAGEConv((-1, -1), -1, hidden_channels),
                    }, aggr='mean')
        self.conv2 = HeteroConv({
                        ('user', 'rates', 'anime'): EdgeSAGEConv((-1, -1), -1, hidden_channels),
                        ('anime', 'rev_rates', 'user'): EdgeSAGEConv((-1, -1), -1, hidden_channels),
                    }, aggr='mean')

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {key: x.relu() for key, x in x_dict.items()}
        return x_dict 

# Predictor part to predict the regression output
class MLPPredictor(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.mlp = MLP(in_channels = 2 * hidden_channels, hidden_channels = hidden_channels, out_channels=1, num_layers=2, dropout=0.05)

    def forward(self, x_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([x_dict['user'][row], x_dict['anime'][col]], dim=-1)
        z = self.mlp(z)
        return z.view(-1)

# create model class that combine graph encoder and regression predictor
# we do this since we want to save the encoder and predictor separately
class GraphPredictorModel(torch.nn.Module):
    def __init__(self, gnn_encoder, predictor):
        super().__init__()
        self.gnn_encoder = gnn_encoder
        self.predictor = predictor

    def forward(self, x_dict, edge_index_dict, edge_label_index, edge_attr_dict):
        x_dict = self.gnn_encoder(x_dict, edge_index_dict, edge_attr_dict)
        out = self.predictor(x_dict, edge_label_index)
        return out