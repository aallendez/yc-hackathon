import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import SAGEConv


class TrackSAGE(nn.Module):
    """GraphSAGE encoder: input projection -> L x SAGEConv(mean) -> output, plus a skip
    connection from the input so a track's own features always reach its embedding.
    layers=0 is a plain MLP on features (the "no graph" ablation)."""

    def __init__(self, in_dim: int, hidden: int = 128, out: int = 64, layers: int = 2,
                 dropout: float = 0.25):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(SAGEConv(hidden, hidden, aggr="mean") for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.out = nn.Linear(hidden, out)
        self.skip = nn.Linear(in_dim, out)
        self.dropout = dropout
        self.logit_scale = nn.Parameter(torch.tensor(2.3))  # exp(2.3) ~ 10, cosine -> logits

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.inp(x))
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.dropout(F.relu(norm(conv(h, edge_index))), self.dropout, self.training)
        return F.normalize(self.out(h) + self.skip(x), dim=-1)

    def score(self, z: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        return (z[src] * z[dst]).sum(-1) * self.logit_scale.exp()
