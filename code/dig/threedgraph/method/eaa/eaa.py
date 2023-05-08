from math import pi as PI
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Sequential, Linear
from torch_scatter import scatter
from torch_geometric.nn import radius_graph
from torch_geometric.nn.inits import glorot_orthogonal
from torch_geometric.nn import GATConv,NNConv
from torch_geometric.utils import softmax


class update_e(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters, num_gaussians, cutoff,weight_init,num_heads=10,):
        super(update_e, self).__init__()
        self.cutoff = cutoff
        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.lin = Linear(hidden_channels, num_filters*num_heads, bias=False)
        self.attn_l = torch.nn.Parameter(torch.FloatTensor(size=(1, num_heads, num_filters)))
        self.attn_r = torch.nn.Parameter(torch.FloatTensor(size=(1, num_heads, num_filters)))
        self.attn_edge = torch.nn.Parameter(torch.FloatTensor(size=(1, num_heads, num_filters)))
        self.act = ShiftedSoftplus()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters*num_heads),
        )
        self.weight_init=weight_init
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.attn_l, scale=2.0)
            glorot_orthogonal(self.attn_r, scale=2.0)
            glorot_orthogonal(self.attn_edge, scale=2.0)
            glorot_orthogonal(self.lin.weight, scale=2.0)
            glorot_orthogonal(self.mlp[0].weight, scale=2.0)
            self.mlp[0].bias.data.fill_(0)
            glorot_orthogonal(self.mlp[2].weight, scale=2.0)
            self.mlp[2].bias.data.fill_(0)
        else:
            torch.nn.init.xavier_uniform_(self.attn_l)
            torch.nn.init.xavier_uniform_(self.attn_r)
            torch.nn.init.xavier_uniform_(self.attn_edge)
            torch.nn.init.xavier_uniform_(self.lin.weight)
            torch.nn.init.xavier_uniform_(self.mlp[0].weight)
            self.mlp[0].bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(self.mlp[2].weight)
            self.mlp[2].bias.data.fill_(0)

    def forward(self, v, dist, dist_emb, edge_index):
        src_prefix_shape = dst_prefix_shape = v.shape[:-1]
        edge_prefix_shape = dist_emb.shape[:-1]
        j, i = edge_index
        #u,v index
        #C = 0.5 * (torch.cos(dist * PI / self.cutoff) + 1.0)
        #W = self.mlp(dist_emb) * C.view(-1, 1)
        W = self.mlp(dist_emb)
        v = self.lin(v)
        v = v.view(*src_prefix_shape, self.num_heads, self.num_filters)
        W = W.view(*edge_prefix_shape, self.num_heads, self.num_filters)
        #get weight score
        left = (v * self.attn_l).sum(dim=-1).unsqueeze(-1)
        right = (v * self.attn_r).sum(dim=-1).unsqueeze(-1)
        eedge = (W * self.attn_edge).sum(dim=-1).unsqueeze(-1)
        #e = v[j] * W * v[i]
        e = left[j] + eedge + right[i]

        # e = v[j] + v[i]
        e = self.act(e)
        #e = softmax(e,index=j,dim=-1)
        e = v[j] * e * W
        return e


class update_v(torch.nn.Module):
    def __init__(self, hidden_channels, num_filters,weight_init, num_heads=10):
        super(update_v, self).__init__()
        self.act = ShiftedSoftplus()
        self.lin1 = Linear(num_filters*num_heads, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.num_filters = num_filters
        self.num_heads = num_heads
        self.weight_init = weight_init
        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_init == 'GlorotOrthogonal':
            glorot_orthogonal(self.lin1.weight, scale=2.0)
            self.lin1.bias.data.fill_(0)
            glorot_orthogonal(self.lin2.weight, scale=2.0)
            self.lin2.bias.data.fill_(0)
        else:
            torch.nn.init.xavier_uniform_(self.lin1.weight)
            self.lin1.bias.data.fill_(0)
            torch.nn.init.xavier_uniform_(self.lin2.weight)
            self.lin2.bias.data.fill_(0)

    def forward(self, v, e, edge_index):
        _, i = edge_index
        out = scatter(e, i, dim=0)
        out = out.view(-1, self.num_filters * self.num_heads)
        out = self.lin1(out)
        out = self.act(out)
        out = self.lin2(out)
        return v + out


class update_u(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(update_u, self).__init__()
        self.lin1 = Linear(hidden_channels, hidden_channels // 2)
        self.act = ShiftedSoftplus()
        self.lin2 = Linear(hidden_channels // 2, 1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.lin1.weight)
        self.lin1.bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.lin2.weight)
        self.lin2.bias.data.fill_(0)

    def forward(self, v, batch,use_stand,mean,std):
        v = self.lin1(v)
        v = self.act(v)
        v = self.lin2(v)
        if use_stand:
            v = v * std + mean
        u = scatter(v, batch, dim=0)
        return u


class emb(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super(emb, self).__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class EAA(torch.nn.Module):
    r"""
        The re-implementation for SchNet from the `"SchNet: A Continuous-filter Convolutional Neural Network for Modeling Quantum Interactions" <https://arxiv.org/abs/1706.08566>`_ paper
        under the 3DGN gramework from `"Spherical Message Passing for 3D Molecular Graphs" <https://openreview.net/forum?id=givsRXsOt9r>`_ paper.
        
        Args:
            energy_and_force (bool, optional): If set to :obj:`True`, will predict energy and take the negative of the derivative of the energy with respect to the atomic positions as predicted forces. (default: :obj:`False`)
            num_layers (int, optional): The number of layers. (default: :obj:`6`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            num_filters (int, optional): The number of filters to use. (default: :obj:`128`)
            num_gaussians (int, optional): The number of gaussians :math:`\mu`. (default: :obj:`50`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`).
    """
    def __init__(self, energy_and_force=False, cutoff=10.0,
                 num_layers=6, hidden_channels=128, num_filters=128,
                 num_gaussians=50,num_heads=10,use_stand=False,mean=0,std=0,
                 weight_init='xavier_uniform_'):
        super(EAA, self).__init__()
        self.use_stand = use_stand
        self.mean = mean
        self.std = std
        self.energy_and_force = energy_and_force
        self.cutoff = cutoff
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_gaussians = num_gaussians

        self.init_v = Embedding(100, hidden_channels)
        self.dist_emb = emb(0.0, cutoff, num_gaussians)

        self.update_vs = torch.nn.ModuleList([update_v(hidden_channels, num_filters,weight_init,num_heads) for _ in range(num_layers)])

        self.update_es = torch.nn.ModuleList([
            update_e(hidden_channels, num_filters, num_gaussians, cutoff,weight_init,num_heads) for _ in range(num_layers)])
        
        self.update_u = update_u(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.init_v.reset_parameters()
        for update_e in self.update_es:
            update_e.reset_parameters()
        for update_v in self.update_vs:
            update_v.reset_parameters()
        self.update_u.reset_parameters()

    def forward(self, batch_data):
        z, pos, batch = batch_data.z, batch_data.pos, batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()

        edge_index = radius_graph(pos, r=self.cutoff, batch=batch)
        row, col = edge_index
        dist = (pos[row] - pos[col]).norm(dim=-1)
        dist_emb = self.dist_emb(dist)

        v = self.init_v(z)

        for update_e, update_v in zip(self.update_es, self.update_vs):
            e = update_e(v, dist, dist_emb, edge_index)
            v = update_v(v, e, edge_index)
        u = self.update_u(v, batch,self.use_stand,self.mean,self.std)

        return u
