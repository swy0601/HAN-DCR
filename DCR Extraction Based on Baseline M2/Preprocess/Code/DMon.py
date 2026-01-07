import torch
from torch_geometric.nn import GCNConv, DenseGraphConv
# from dmon_pooling import DMoNPooling   # 使用本地副本
from math import ceil
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch.nn import Linear
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_undirected

# dmon_pooling.py  —— 旧版 DMoNPooling 的平替实现
import torch
from torch.nn import Linear, Parameter
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class DMoNPooling(torch.nn.Module):
    """
    与原 torch_geometric.nn.DMoNPooling 接口完全一致。
    输出：s, x, adj, spectral_loss, ortho_loss, cluster_loss
    """

    def __init__(self, channels, num_clusters, dropout=0):
        super().__init__()
        self.num_clusters = num_clusters
        self.dropout = dropout

        # 用于生成软分配矩阵 S 的 MLP
        if isinstance(channels, (list, tuple)):
            in_channels = channels[0]
        else:
            in_channels = channels
        self.mlp = torch.nn.Sequential(
            Linear(in_channels, channels[1] if isinstance(channels, (list, tuple)) else channels),
            torch.nn.ReLU(inplace=True),
            Linear(channels[1] if isinstance(channels, (list, tuple)) else channels, num_clusters)
        )

    def forward(self, x, adj, mask=None):
        """
        x   : (batch, num_nodes, feat)  —— 必须是 dense batch
        adj : (batch, num_nodes, num_nodes)
        mask: (batch, num_nodes)  bool  True 表示真实节点
        """
        B, N, _ = x.size()
        s = self.mlp(x)                                    # (B, N, K)
        s = F.dropout(s, p=self.dropout, training=self.training)
        s = softmax(s, index=None, batch=None, dim=1)      # 沿节点维归一化

        # 掩掉 padding 节点
        if mask is not None:
            s = s * mask.unsqueeze(-1)

        # 新特征  X' = S^T X
        x_pooled = torch.matmul(s.transpose(1, 2), x)      # (B, K, feat)
        # 新邻接 A' = S^T A S
        adj_pooled = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s)

        # DMoN 损失
        eye = torch.eye(self.num_clusters, device=x.device, dtype=x.dtype)
        s_gram = torch.matmul(s.transpose(1, 2), s)        # (B, K, K)
        ortho_loss = (s_gram - eye).pow(2).mean()          # 正交正则
        cluster_loss = -torch.sum(s * torch.log(s + 1e-12), dim=1).mean()  # 熵正则
        spectral_loss = 0.0                                # 示例里未用，可设 0

        # 归一化新邻接
        deg = adj_pooled.sum(dim=-1, keepdim=True).clamp_min(1)
        adj_pooled = adj_pooled / deg

        return s, x_pooled, adj_pooled, spectral_loss, ortho_loss, cluster_loss



class DMon(torch.nn.Module):

    def __init__(self, data_args, model_args):
        super().__init__()

        self.in_channels = data_args.num_node_features
        self.out_channels = data_args.num_classes
        self.avg_num_nodes = data_args.avg_num_nodes

        self.hidden_channels = model_args.hidden_channels
        self.mlp_hidden = model_args.mlp_hidden

        self.conv1 = GCNConv(self.in_channels, self.hidden_channels)
        self.conv4 = GCNConv(self.hidden_channels, self.hidden_channels)

        num_nodes = ceil(0.5 * self.avg_num_nodes)
        self.pool1 = DMoNPooling([self.hidden_channels, self.hidden_channels], num_nodes)
        self.conv2 = DenseGraphConv(self.hidden_channels, self.hidden_channels)

        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = DMoNPooling([self.hidden_channels, self.hidden_channels], num_nodes)
        self.conv3 = DenseGraphConv(self.hidden_channels, self.hidden_channels)

        num_nodes = ceil(0.5 * num_nodes)
        self.pool3 = DMoNPooling([self.hidden_channels, self.hidden_channels], num_nodes)
        self.conv5 = DenseGraphConv(self.hidden_channels, self.hidden_channels)

        # 升维层：将128维升到256维
        # self.fc_up = Linear(self.hidden_channels, self.hidden_channels * 2)  # 升维层
        self.lin1 = Linear(self.hidden_channels, self.mlp_hidden)

        # self.lin1 = Linear(self.hidden_channels, self.mlp_hidden)
        self.lin2 = Linear(self.mlp_hidden, self.out_channels)


    def forward(self, x, edge_index, batch, return_features=False):
        x = self.conv1(x, edge_index).relu()
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)
        
        _, x, adj, sp1, o1, c1 = self.pool1(x, adj, mask)
        x = self.conv2(x, adj).relu()
        _, x, adj, sp2, o2, c2 = self.pool2(x, adj)
        x = self.conv3(x, adj)
        x = x.mean(dim=1)
        feature = x
        x = self.lin1(x).relu()
        logits = self.lin2(x)
        
        if return_features:
            return feature, sp1   # 作为特征提取器使用
        else:
            return F.log_softmax(logits, dim=-1), sp1  # 作为分类器使用


    
    def update_state_dict(self, state_dict):
        original_state_dict = self.state_dict()
        loaded_state_dict = dict()
        for k, v in state_dict.items():
            if k in original_state_dict.keys():
                loaded_state_dict[k] = v
        self.load_state_dict(loaded_state_dict)

# torch.nn.Sigmoid()
# torch.nn.LeakyReLU()
# torch.nn.Relu()
# torch.nn.ELU()
# torch.nn.SELU()
