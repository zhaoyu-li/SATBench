import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from satbench.models.mlp import MLP
from satbench.models.ln_lstm_cell import LayerNormBasicLSTMCell
from torch_scatter import scatter_sum, scatter_mean
from torch_geometric.utils import softmax


class NeuroSAT(nn.Module):
    def __init__(self, opts):
        super(NeuroSAT, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c_update = LayerNormBasicLSTMCell(self.opts.dim, self.opts.dim)
        self.l_update = LayerNormBasicLSTMCell(self.opts.dim * 2, self.opts.dim)
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_state = torch.zeros(l_size, self.opts.dim).to(self.opts.device)
        c_state = torch.zeros(c_size, self.opts.dim).to(self.opts.device)

        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)
            c_emb, c_state = self.c_update(l2c_msg_aggr, (c_emb, c_state))
            c_embs.append(c_emb)

            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)
            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            l_emb, l_state = self.l_update(torch.cat([c2l_msg_aggr, l2l_msg], dim=1), (l_emb, l_state))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GGNN_LCG(nn.Module):
    def __init__(self, opts):
        super(GGNN_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c_update = nn.GRUCell(self.opts.dim, self.opts.dim)
        self.l_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg_feat = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            l2l_msg = self.l2l_msg_func(l2l_msg_feat)

            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)
            c_emb = self.c_update(input=l2c_msg_aggr, hx=c_emb)
            c_embs.append(c_emb)

            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)
            l_emb = self.l_update(input=torch.cat([c2l_msg_aggr, l2l_msg], dim=1), hx=l_emb)
            l_embs.append(l_emb)

        return l_embs, c_embs


class GGNNv2_LCG(nn.Module):
    def __init__(self, opts):
        super(GGNNv2_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        self.c2l_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        self.l2l_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        self.c_update = nn.ModuleList(
            [nn.GRUCell(self.opts.dim, self.opts.dim) for _ in range(self.opts.n_iterations)]
        )
        self.l_update = nn.ModuleList(
            [nn.GRUCell(self.opts.dim * 2, self.opts.dim) for _ in range(self.opts.n_iterations)]
        )
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            l_msg_feat = self.l2c_msg_func[i](l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            c_msg_feat = self.c2l_msg_func[i](c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg_feat = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            l2l_msg = self.l2l_msg_func[i](l2l_msg_feat)

            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)
            c_emb = self.c_update[i](input=l2c_msg_aggr, hx=c_emb)
            c_embs.append(c_emb)

            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)
            l_emb = self.l_update[i](input=torch.cat([c2l_msg_aggr, l2l_msg], dim=1), hx=l_emb)
            l_embs.append(l_emb)

        return l_embs, c_embs


class GCN_LCG(nn.Module):
    def __init__(self, opts):
        super(GCN_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)

        self.c_update = nn.Linear(self.opts.dim * 2, self.opts.dim)
        self.l_update = nn.Linear(self.opts.dim * 3, self.opts.dim)
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_one = torch.ones((l_edge_index.size(0), 1), device=self.opts.device)
        l_deg = scatter_sum(l_one, l_edge_index, dim=0, dim_size=l_size)
        c_one = torch.ones((c_edge_index.size(0), 1), device=self.opts.device)
        c_deg = scatter_sum(c_one, c_edge_index, dim=0, dim_size=c_size)
        degree_norm = l_deg[l_edge_index].pow(0.5) * c_deg[c_edge_index].pow(0.5)

        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg_feat = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            l2l_msg = self.l2l_msg_func(l2l_msg_feat)

            l2c_msg_aggr = scatter_sum(l2c_msg / degree_norm, c_edge_index, dim=0, dim_size=c_size)
            c_emb = self.c_update(torch.cat([c_emb, l2c_msg_aggr], dim=1))
            c_embs.append(c_emb)

            c2l_msg_aggr = scatter_sum(c2l_msg / degree_norm, l_edge_index, dim=0, dim_size=l_size)
            l_emb = self.l_update(torch.cat([l_emb, c2l_msg_aggr, l2l_msg], dim=1))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GCNv2_LCG(nn.Module):
    def __init__(self, opts):
        super(GCNv2_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        self.c2l_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        self.l2l_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )

        self.c_update = nn.ModuleList(
            [nn.Linear(self.opts.dim * 2, self.opts.dim) for _ in range(self.opts.n_iterations)]
        )
        self.l_update = nn.ModuleList(
            [nn.Linear(self.opts.dim * 3, self.opts.dim) for _ in range(self.opts.n_iterations)]
        )
        
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_one = torch.ones((l_edge_index.size(0), 1), device=self.opts.device)
        l_deg = scatter_sum(l_one, l_edge_index, dim=0, dim_size=l_size)
        c_one = torch.ones((c_edge_index.size(0), 1), device=self.opts.device)
        c_deg = scatter_sum(c_one, c_edge_index, dim=0, dim_size=c_size)
        degree_norm = l_deg[l_edge_index].pow(0.5) * c_deg[c_edge_index].pow(0.5)

        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            l_msg_feat = self.l2c_msg_func[i](l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            c_msg_feat = self.c2l_msg_func[i](c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg_feat = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            l2l_msg = self.l2l_msg_func[i](l2l_msg_feat)

            l2c_msg_aggr = scatter_sum(l2c_msg / degree_norm, c_edge_index, dim=0, dim_size=c_size)
            c_emb = self.c_update[i](torch.cat([c_emb, l2c_msg_aggr], dim=1))
            c_embs.append(c_emb)

            c2l_msg_aggr = scatter_sum(c2l_msg / degree_norm, l_edge_index, dim=0, dim_size=l_size)
            l_emb = self.l_update[i](torch.cat([l_emb, c2l_msg_aggr, l2l_msg], dim=1))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GAT_LCG(nn.Module):
    def __init__(self, opts):
        super(GAT_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        
        self.c_attention = nn.Linear(self.opts.dim * 2, 1, bias=False)
        self.c_update = nn.Linear(self.opts.dim * 2, self.opts.dim)
        self.l_attention = nn.Linear(self.opts.dim * 2, 1, bias=False)
        self.l_update = nn.Linear(self.opts.dim * 3, self.opts.dim)
        self.negative_slope = 0.2
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_weight = self.c_attention(torch.cat([c_emb[c_edge_index], l2c_msg], dim=1))
            l2c_msg_weight = F.leaky_relu(l2c_msg_weight, self.negative_slope)
            l2c_msg_weight = softmax(l2c_msg_weight, c_edge_index, dim=0)

            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_weight = self.l_attention(torch.cat([l_emb[l_edge_index], c2l_msg], dim=1))
            c2l_msg_weight = F.leaky_relu(c2l_msg_weight, self.negative_slope)
            c2l_msg_weight = softmax(c2l_msg_weight, c_edge_index, dim=0)

            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg_feat = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            l2l_msg = self.l2l_msg_func(l2l_msg_feat)
            
            l2c_msg_aggr = scatter_sum(l2c_msg * l2c_msg_weight, c_edge_index, dim=0, dim_size=c_size)
            c_emb = self.c_update(torch.cat([c_emb, l2c_msg_aggr], dim=1))
            c_embs.append(c_emb)

            c2l_msg_aggr = scatter_sum(c2l_msg * c2l_msg_weight, l_edge_index, dim=0, dim_size=l_size)
            l_emb = self.l_update(torch.cat([l_emb, c2l_msg_aggr, l2l_msg], dim=1))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GATv2_LCG(nn.Module):
    def __init__(self, opts):
        super(GATv2_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        self.c2l_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        self.l2l_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        
        self.c_attention = nn.ModuleList(
            [nn.Linear(self.opts.dim * 2, 1, bias=False) for _ in range(self.opts.n_iterations)]
        )
        self.c_update = nn.ModuleList(
            [nn.Linear(self.opts.dim * 2, self.opts.dim) for _ in range(self.opts.n_iterations)]
        )
        self.l_attention = nn.ModuleList(
            [nn.Linear(self.opts.dim * 2, 1, bias=False) for _ in range(self.opts.n_iterations)]
        )
        self.l_update = nn.ModuleList(
            [nn.Linear(self.opts.dim * 3, self.opts.dim) for _ in range(self.opts.n_iterations)]
        )
        self.negative_slope = 0.2
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            l_msg_feat = self.l2c_msg_func[i](l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            l2c_msg_weight = self.c_attention[i](torch.cat([c_emb[c_edge_index], l2c_msg], dim=1))
            l2c_msg_weight = F.leaky_relu(l2c_msg_weight, self.negative_slope)
            l2c_msg_weight = softmax(l2c_msg_weight, c_edge_index, dim=0)
            
            c_msg_feat = self.c2l_msg_func[i](c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            c2l_msg_weight = self.l_attention[i](torch.cat([l_emb[l_edge_index], c2l_msg], dim=1))
            c2l_msg_weight = F.leaky_relu(c2l_msg_weight, self.negative_slope)
            c2l_msg_weight = softmax(c2l_msg_weight, c_edge_index, dim=0)

            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg_feat = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            l2l_msg = self.l2l_msg_func[i](l2l_msg_feat)
            
            l2c_msg_aggr = scatter_sum(l2c_msg * l2c_msg_weight, c_edge_index, dim=0, dim_size=c_size)
            c_emb = self.c_update[i](torch.cat([c_emb, l2c_msg_aggr], dim=1))
            c_embs.append(c_emb)

            c2l_msg_aggr = scatter_sum(c2l_msg * c2l_msg_weight, l_edge_index, dim=0, dim_size=l_size)
            l_emb = self.l_update[i](torch.cat([l_emb, c2l_msg_aggr, l2l_msg], dim=1))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GIN_LCG(nn.Module):
    def __init__(self, opts):
        super(GIN_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.c2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l2l_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        
        self.c_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)
        self.l_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim, self.opts.activation)
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            l_msg_feat = self.l2c_msg_func(l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            c_msg_feat = self.c2l_msg_func(c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg_feat = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            l2l_msg = self.l2l_msg_func(l2l_msg_feat)
            
            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)
            c_emb = self.c_update(torch.cat([c_emb, l2c_msg_aggr], dim=1))
            c_embs.append(c_emb)
            
            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)
            l_emb = self.l_update(torch.cat([l_emb, c2l_msg_aggr, l2l_msg], dim=1))
            l_embs.append(l_emb)

        return l_embs, c_embs


class GINv2_LCG(nn.Module):
    def __init__(self, opts):
        super(GINv2_LCG, self).__init__()
        self.opts = opts
        self.l2c_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        self.c2l_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        self.l2l_msg_func = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        
        self.c_update = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
        self.l_update = nn.ModuleList(
            [MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim, self.opts.activation) for _ in range(self.opts.n_iterations)]
        )
    
    def forward(self, l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb):
        l_embs = [l_emb]
        c_embs = [c_emb]

        for i in range(self.opts.n_iterations):
            l_msg_feat = self.l2c_msg_func[i](l_emb)
            l2c_msg = l_msg_feat[l_edge_index]
            c_msg_feat = self.c2l_msg_func[i](c_emb)
            c2l_msg = c_msg_feat[c_edge_index]
            pl_emb, ul_emb = torch.chunk(l_emb.reshape(l_size // 2, -1), 2, 1)
            l2l_msg_feat = torch.cat([ul_emb, pl_emb], dim=1).reshape(l_size, -1)
            l2l_msg = self.l2l_msg_func[i](l2l_msg_feat)
            
            l2c_msg_aggr = scatter_sum(l2c_msg, c_edge_index, dim=0, dim_size=c_size)
            c_emb = self.c_update[i](torch.cat([c_emb, l2c_msg_aggr], dim=1))
            c_embs.append(c_emb)
            
            c2l_msg_aggr = scatter_sum(c2l_msg, l_edge_index, dim=0, dim_size=l_size)
            l_emb = self.l_update[i](torch.cat([l_emb, c2l_msg_aggr, l2l_msg], dim=1))
            l_embs.append(l_emb)

        return l_embs, c_embs
     

class GNN_LCG(nn.Module):
    def __init__(self, opts):
        super(GNN_LCG, self).__init__()
        self.opts = opts
        if self.opts.init_emb == 'learned':
            self.l_init = nn.Parameter(torch.empty(1, self.opts.dim))
            nn.init.kaiming_normal_(self.l_init)
            self.c_init = nn.Parameter(torch.empty(1, self.opts.dim))
            nn.init.kaiming_normal_(self.c_init)
        
        if self.opts.model == 'neurosat':
            self.gnn = NeuroSAT(self.opts)
        elif self.opts.model == 'ggnn':
            self.gnn = GGNN_LCG(self.opts)
        elif self.opts.model == 'ggnn*':
            self.gnn = GGNNv2_LCG(self.opts)
        elif self.opts.model == 'gcn':
            self.gnn = GCN_LCG(self.opts)
        elif self.opts.model == 'gcn*':
            self.gnn = GCNv2_LCG(self.opts)
        elif self.opts.model == 'gat':
            self.gnn = GAT_LCG(self.opts)
        elif self.opts.model == 'gat*':
            self.gnn = GATv2_LCG(self.opts)
        elif self.opts.model == 'gin':
            self.gnn = GIN_LCG(self.opts)
        elif self.opts.model == 'gin*':
            self.gnn = GINv2_LCG(self.opts)
        
        if self.opts.task == 'satisfiability':
            self.g_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        else:
            # self.opts.task == 'assignment' or self.opts.task == 'core_variable'
            self.l_readout = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, 1, self.opts.activation)
        
        if self.opts.use_contrastive_learning:
            self.tau = 0.5
            self.proj = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
    
    def forward(self, data):
        batch_size = data.num_graphs
        l_size = data.l_size.sum().item()
        c_size = data.c_size.sum().item()
        l_edge_index = data.l_edge_index
        c_edge_index = data.c_edge_index

        if self.opts.init_emb == 'learned':
            l_emb = (self.l_init).repeat(l_size, 1)
            c_emb = (self.c_init).repeat(c_size, 1)
        else:
            # self.opts.init_emb == 'random'
            l_emb = torch.randn((l_size, self.opts.dim), device=self.opts.device)
            c_emb = torch.randn((c_size, self.opts.dim), device=self.opts.device)

        l_embs, c_embs = self.gnn(l_size, c_size, l_edge_index, c_edge_index, l_emb, c_emb)
        
        if self.opts.task == 'satisfiability':
            if self.opts.satisfiability_readout == 'literal':
                l_batch = data.l_batch
                g_emb = scatter_mean(l_embs[-1], l_batch, dim=0, dim_size=batch_size)
                g_logit = self.g_readout(g_emb).reshape(-1)
            else:
                # self.opts.satisfiability_readout == 'clause'
                c_batch = data.c_batch
                g_emb = scatter_mean(c_embs[-1], c_batch, dim=0, dim_size=batch_size)
                g_logit = self.g_readout(g_emb).reshape(-1)
            
            if self.training and self.opts.use_contrastive_learning:
                g_emb = self.proj(g_emb)
                h = F.normalize(g_emb)
                sim = torch.exp(torch.mm(h, h.t()) / self.tau)
                # remove the similarity measure between two same objects
                mask = (1 - torch.eye(batch_size, device=self.opts.device)) 
                sim = sim * mask
                return torch.sigmoid(g_logit), sim
            else:
                return torch.sigmoid(g_logit)

        elif self.opts.task == 'assignment':
            if not self.training and self.opts.multiple_assignments:
                v_assigns = []
                for l_emb in l_embs:
                    v_logit = self.l_readout(l_emb.reshape(-1, self.opts.dim * 2))
                    v_assigns.append(torch.sigmoid(v_logit))
                return v_assigns
            else:
                v_logit = self.l_readout(l_embs[-1].reshape(-1, self.opts.dim * 2))
                return torch.sigmoid(v_logit)


class GNN_VCG(nn.Module):
    def __init__(self, opts):
        super(GNN_VCG, self).__init__()
        if self.opts.init_emb == 'learned':
            self.v_init = nn.Parameter(torch.randn(1, self.opts.dim))
            self.c_init = nn.Parameter(torch.randn(1, self.opts.dim))

        self.p_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.n_v2c_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.p_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        self.n_c2v_msg_func = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, self.opts.dim, self.opts.activation)
        
        if self.opts.updater == 'gru':
            self.c_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)
            self.v_update = nn.GRUCell(self.opts.dim * 2, self.opts.dim)
        elif self.opts.updater == 'mlp1':
            self.c_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim, self.opts.activation)
            self.l_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 3, self.opts.dim, self.opts.dim, self.opts.activation)
        elif self.opts.updater == 'mlp2':
            self.c_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)
            self.l_update = MLP(self.opts.n_mlp_layers, self.opts.dim * 2, self.opts.dim, self.opts.dim, self.opts.activation)

        self.g_readout = MLP(self.opts.n_mlp_layers, self.opts.dim, self.opts.dim, 1, self.opts.activation)
        self.init_norm = math.sqrt(self.opts.dim)
    
    def forward(self, data):
        v_size = data.v_size.sum().item()
        c_size = data.c_size.sum().item()

        c_edge_index = data.c_edge_index
        v_edge_index = data.v_edge_index
        p_edge_index = data.p_edge_index
        n_edge_index = data.n_edge_index

        if self.opts.init_emb == 'learned':
            v_emb = (self.v_init).repeat(v_size, 1) / self.init_norm
            c_emb = (self.c_init).repeat(c_size, 1) / self.init_norm
        else:
            v_emb = torch.randn(v_size, self.opts.dim) / self.init_norm
            c_emb = torch.randn(c_size, self.opts.dim) / self.init_norm
        
        if self.opts.aggregator == 'mean':
            p_v_one = torch.ones((v_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
            p_v_deg = scatter_sum(p_v_one, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            p_v_deg[p_v_deg < 1] = 1
            n_v_one = torch.ones((v_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
            n_v_deg = scatter_sum(n_v_one, v_edge_index[n_edge_index], dim=0, dim_size=v_size)
            n_v_deg[n_v_deg < 1] = 1

            p_c_one = torch.ones((c_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
            p_c_deg = scatter_sum(p_c_one, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            p_c_deg[p_c_deg < 1] = 1
            n_c_one = torch.ones((c_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
            n_c_deg = scatter_sum(n_c_one, c_edge_index[n_edge_index], dim=0, dim_size=c_size)
            n_c_deg[n_c_deg < 1] = 1

        elif self.opts.aggregator == 'degree-norm':
            p_v_one = torch.ones((v_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
            p_v_deg = scatter_sum(p_v_one, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            p_v_deg[p_v_deg < 1] = 1
            n_v_one = torch.ones((v_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
            n_v_deg = scatter_sum(n_v_one, v_edge_index[n_edge_index], dim=0, dim_size=v_size)
            n_v_deg[n_v_deg < 1] = 1

            p_c_one = torch.ones((c_edge_index[p_edge_index].size(0), 1), device=self.opts.device)
            p_c_deg = scatter_sum(p_c_one, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            p_c_deg[p_c_deg < 1] = 1
            n_c_one = torch.ones((c_edge_index[n_edge_index].size(0), 1), device=self.opts.device)
            n_c_deg = scatter_sum(n_c_one, c_edge_index[n_edge_index], dim=0, dim_size=c_size)
            n_c_deg[n_c_deg < 1] = 1

            p_norm = p_v_deg[v_edge_index[p_edge_index]].pow(0.5) * p_c_deg[c_edge_index[p_edge_index]].pow(0.5)
            n_norm = n_v_deg[v_edge_index[n_edge_index]].pow(0.5) * n_c_deg[c_edge_index[n_edge_index]].pow(0.5)

        for i in range(self.opts.n_iterations):
            p_v2c_msg_feat = self.p_v2c_msg_func(v_emb)
            p_v2c_msg = p_v2c_msg_feat[v_edge_index[p_edge_index]]
            if self.opts.aggregator == 'sum':
                p_v2c_msg_aggr = scatter_sum(p_v2c_msg, c_edge_index[p_edge_index], dim=0, dim_size=c_size)
            elif self.opts.aggregator == 'mean':
                p_v2c_msg_aggr = scatter_sum(p_v2c_msg, c_edge_index[p_edge_index], dim=0, dim_size=c_size) / p_c_deg
            elif self.opts.aggregator == 'degree-norm':
                p_v2c_msg_aggr = scatter_sum(p_v2c_msg / p_norm, c_edge_index[p_edge_index], dim=0, dim_size=c_size)

            n_v2c_msg_feat = self.n_v2c_msg_func(v_emb)
            n_v2c_msg = n_v2c_msg_feat[v_edge_index[n_edge_index]]
            if self.opts.aggregator == 'sum':
                n_v2c_msg_aggr = scatter_sum(n_v2c_msg, c_edge_index[n_edge_index], dim=0, dim_size=c_size)
            elif self.opts.aggregator == 'mean':
                n_v2c_msg_aggr = scatter_sum(n_v2c_msg, c_edge_index[n_edge_index], dim=0, dim_size=c_size) / n_c_deg
            elif self.opts.aggregator == 'degree-norm':
                n_v2c_msg_aggr = scatter_sum(n_v2c_msg / n_norm, c_edge_index[n_edge_index], dim=0, dim_size=c_size)
            
            if self.opts.updater == 'gru':
                c_emb = self.c_update(torch.cat([p_v2c_msg_aggr, n_v2c_msg_aggr], dim=1), c_emb)
            elif self.opts.updater == 'mlp1':
                c_emb = self.c_update(torch.cat([p_v2c_msg_aggr, n_v2c_msg_aggr, c_emb], dim=1))
            elif self.opts.updater == 'mlp2':
                c_emb = self.c_update(torch.cat([p_v2c_msg_aggr, n_v2c_msg_aggr], dim=1))

            p_c2v_msg_feat = self.p_c2v_msg_func(c_emb)
            p_c2v_msg = p_c2v_msg_feat[c_edge_index[p_edge_index]]
            if self.opts.aggregator == 'sum':
                p_c2v_msg_aggr = scatter_sum(p_c2v_msg, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            elif self.opts.aggregator == 'mean':
                p_c2v_msg_aggr = scatter_sum(p_c2v_msg, v_edge_index[p_edge_index], dim=0, dim_size=v_size) / p_v_deg
            elif self.opts.aggregator == 'degree-norm':
                p_c2v_msg_aggr = scatter_sum(p_c2v_msg / p_norm, v_edge_index[p_edge_index], dim=0, dim_size=v_size)
            
            n_c2v_msg_feat = self.n_c2v_msg_func(c_emb)
            n_c2v_msg = n_c2v_msg_feat[c_edge_index[n_edge_index]]
            if self.opts.aggregator == 'sum':
                n_c2v_msg_aggr = scatter_sum(n_c2v_msg, v_edge_index[n_edge_index], dim=0, dim_size=v_size)
            elif self.opts.aggregator == 'mean':
                n_c2v_msg_aggr = scatter_sum(n_c2v_msg, v_edge_index[n_edge_index], dim=0, dim_size=v_size) / n_v_deg
            elif self.opts.aggregator == 'degree-norm':
                n_c2v_msg_aggr = scatter_sum(n_c2v_msg / n_norm, v_edge_index[n_edge_index], dim=0, dim_size=v_size)
            
            if self.opts.updater == 'gru':
                v_emb = self.v_update(torch.cat([p_c2v_msg_aggr, n_c2v_msg_aggr], dim=1), v_emb)
            elif self.opts.updater == 'mlp1':
                v_emb = self.v_update(torch.cat([p_c2v_msg_aggr, n_c2v_msg_aggr, v_emb], dim=1))
            elif self.opts.updater == 'mlp2':
                v_emb = self.v_update(torch.cat([p_c2v_msg_aggr, n_c2v_msg_aggr], dim=1))            
        
        if self.opts.task == 'satisfiability':
            v_batch = data.v_batch
            batch_size = data.num_graphs
            g_emb = scatter_mean(v_emb, v_batch, dim=0, dim_size=batch_size)
            g_logit = self.g_readout(g_emb).reshape(-1)
            return torch.sigmoid(g_logit)

        elif self.opts.task == 'assignment':
            v_logit = self.v_readout(v_emb)
            return torch.sigmoid(v_logit)


def GNN(opts):
    if opts.graph == 'lcg':
        return GNN_LCG(opts)
    else:
        # opts.graph == 'vcg'
        return GNN_VCG(opts)
