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