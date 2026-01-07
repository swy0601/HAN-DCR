import torch
import torch.nn as nn
import torch.nn.functional as F


class DMonClassifier(nn.Module):
    def __init__(self, dmon_model, hidden_dim=128, num_classes=2, dropout=0.5):
        """
        DMonClassifier 用于将 DMon 特征映射到分类结果
        Args:
            dmon_model: 已加载的 DMon 特征提取模型
            hidden_dim: 中间层维度（可根据 DMon 输出维度调整）
            num_classes: 分类类别数量
            dropout: dropout 比例
        """
        super(DMonClassifier, self).__init__()
        self.dmon = dmon_model
        self.dropout = nn.Dropout(dropout)

        # 推测 DMon 输出维度（如果 DMon 有线性层，可用它的输出特征数）
        out_dim = getattr(dmon_model, "out_channels", 256)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index=None, batch=None, return_features=False):
        """
        前向传播
        Args:
            x: [B, F] 输入特征或图节点特征
            edge_index: 图的边索引（如果 DMon 需要）
            batch: batch 索引
            return_features: 是否返回中间特征
        """
        # --- 1. 提取 DMon 特征 ---
        if edge_index is not None and batch is not None:
            features, _ = self.dmon(x, edge_index, batch, return_features=True)
        else:
            # 有些 DMon 模型 forward(x) 就能直接提取特征
            if hasattr(self.dmon, "forward"):
                try:
                    features = self.dmon(x)
                except TypeError:
                    features, _ = self.dmon(x, None, None, return_features=True)
            else:
                features = x  # 如果 DMon 已经是空模型

        # --- 2. 分类 ---
        features = self.dropout(features)
        logits = self.classifier(features)

        if return_features:
            return logits, features
        return logits
