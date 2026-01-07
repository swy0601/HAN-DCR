from han.model_hetero3 import HAN
# RF那儿最终采用的这个，最终最终的！！！
# from han.model_hetero2 import HAN
# 输入128,输出4,特征提取完输出64
# from han.model_hetero4 import HAN
# # 输入128,输出4,特征提取完输出128

# from han.model_hetero import HAN
import loss.model as loss
import torch

from torch import nn

import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, meta_paths, embedding_size, hidden_size, out_size, num_heads, dropout,backbone=None):
        super().__init__()
        if backbone==None:
            backbone = HAN(meta_paths=meta_paths,
                                    in_size=embedding_size,
                                    hidden_size=hidden_size,
                                    out_size=out_size,
                                    num_heads=num_heads,
                                    dropout=dropout)
        self.backbone=backbone
        self.loss = loss.LogLoss()


    def _forward_features(self,g,features):
        output = self.backbone(g, features)
        return output

    def _forward_train(self,features,sample: list, positive: list, negative: list):
        samples = torch.stack([features[i] for i in sample])
        positives = torch.stack([features[i] for i in positive])
        negatives = torch.stack([features[i] for i in negative])
        l1 = self.loss(samples, positives, correlation=True)
        l2 = self.loss(samples, negatives, correlation=False)
        return l1.sum() + l2.sum()

    def _forward_train_positive(self,features,sample: list, positive: list):
        # sample 列表包含的是索引，这些索引指向 features 列表或张量中的特定位置。
        # 从features中获取sample对应索引的特征向量，并重新组合
        samples = torch.stack([features[i] for i in sample])
        positives = torch.stack([features[i] for i in positive])
        l1 = self.loss(samples, positives, correlation=True)
        return l1.sum()

    def _forward_train_negative(self, features, sample: list,negative: list):
        samples = torch.stack([features[i] for i in sample])
        negatives = torch.stack([features[i] for i in negative])
        l2 = self.loss(samples, negatives, correlation=False)
        return l2.sum()

    def forward(self, g, features):
        output=self._forward_features(g,features)
        return output

    def evaluation(self, g, features):
        return self.backbone(g, features)

