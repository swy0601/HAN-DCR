import torch
from torch import nn
import torch.nn.functional as F


from typing import cast
from typing import List
from typing import Type
from typing import Tuple
from typing import Union
from typing import Optional

# class Loss(nn.Module):
#     def __init__(self, activation=torch.sigmoid):
#         super().__init__()
#         self.activation = activation
#
#     def forward(self, x1, x2, correlation=True):
#         if correlation:
#             output = torch.mul(x1, x2)
#             output = torch.sum(output, dim=1)
#             output = 2 - self.activation(output)
#             output = torch.log(output)
#         else:
#             output = torch.mul(x1, x2)
#             output = torch.sum(output, dim=1)
#             output = 1 + self.activation(output)
#             output = torch.log(output)
#         return output/x1.shape[0]

class LogLoss(nn.Module):
    def __init__(self, activation=torch.sigmoid):
        super().__init__()
        self.activation = activation

    def forward(self, x1, x2, correlation=True):
        x1 = torch.nn.functional.normalize(x1)
        x2 = torch.nn.functional.normalize(x2)
        if correlation:
            output = torch.mul(x1, x2)
            output = torch.sum(output, dim=1)
            output=(output+1)/2
            output=1-output
            # output = 2 - self.activation(output)
            # output = torch.log(output)
        else:
            output = torch.mul(x1, x2)
            output = torch.sum(output, dim=1)
            output = (output + 1) / 2
            # output = 1 + self.activation(output)
            # output = torch.log(output)
        return torch.sum(output)/output.shape[0]



class ProjectionModel(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc = nn.Linear(in_features, 64)

    def forward(self, x):
        x = self.fc(x)

        return x


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for Models. If both `labels` and `mask` are None,
		it degenerates to SimCLR unsupervised loss:
		https://arxiv.org/pdf/2002.05709.pdf
		Args:
		features: hidden vector of shape [bsz, n_views, ...].
		labels: ground truth of shape [bsz].
		mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
		has the same class as sample i. Can be asymmetric.
		Returns:
		A loss scalar.
		"""
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            # here mask is of shape [bsz, bsz] and is one for one for [i,j] where label[i]=label[j]
            mask = torch.eq(labels, labels.T).float().to(device)

        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # number of positives per sample

        # contrast_features separates the features of different views of the samples and puts them in rows,
        # so features of shape of [50, 2, 128] becomes [100, 128]. we do this to be to calculate dot-product between
        # each two views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits - calculates the dot product of every two vectors divided by temperature
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # for numerical stability  (some kind of normalization!)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask as much as number of positives per sample
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        eps = 1e-30
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eps)

        # loss
        loss = -  mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def loss_fn(self, y_pred: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if isinstance(y_pred, tuple):
            y_pred, *_ = y_pred

        if y_pred.shape == y.shape:
            loss_pred = F.binary_cross_entropy_with_logits(
                y_pred,
                y.to(dtype=y_pred.dtype, device=y_pred.device),
                reduction='sum'
            ) / y_pred.shape[0]
        else:
            loss_pred = F.cross_entropy(y_pred, y.to(y_pred.device))

        return loss_pred

    def forward(self, y_pred, y):
        return self.loss_fn(y_pred, y)


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, temperature=0.07):
        super(HybridLoss, self).__init__()
        self.contrastive_loss = SupConLoss(temperature)
        self.cross_entropy_loss=CrossEntropyLoss()
        self.alpha = alpha

    def cross_entropy_one_hot(self, input, target):
        _, labels = target.max(dim=1)
        return nn.CrossEntropyLoss()(input, labels)

    def forward(self, y_proj, y_pred, label):
        contrastiveLoss = self.contrastive_loss(y_proj.unsqueeze(1), label)
        label_vec=torch.eye(max(label)+1)[label]
        label_vec=label_vec.to(label.device)
        entropyLoss = self.cross_entropy_one_hot(y_pred, label_vec)

        return contrastiveLoss * self.alpha, entropyLoss * (1 - self.alpha)