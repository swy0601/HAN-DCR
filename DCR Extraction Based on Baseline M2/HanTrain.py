
import os
# 尽可能在导入 torch 之前设置 CUDA 确定性相关环境变量
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')  # 或 ':4096:2'

import warnings
warnings.filterwarnings('ignore')

import math
import copy
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

# =============
# 全局设备
# =============
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# =============
# 随机性控制
# =============

def set_global_seed(seed: int):
    """在同一进程内尽可能固定随机性（注意：个别 GPU 稀疏算子在不同驱动/硬件上仍可能存在微小非确定性）。"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        # 开启严格确定性（如遇不支持的算子会抛异常）
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print(f"[WARN] use_deterministic_algorithms 失败或部分算子不支持: {e}")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        dgl.random.seed(seed)
    except Exception:
        pass

# =============
# 工具函数
# =============

def compute_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics['Accuracy']  = float(accuracy_score(y_true, y_pred))
    metrics['Precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['Recall']    = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['F1']        = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        metrics['AUC']   = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics['AUC']   = float('nan')
    metrics['MCC']       = float(matthews_corrcoef(y_true, y_pred))
    return metrics


def print_metrics_table(title, metrics_dict):
    print(f"\n=== {title} ===")
    headers = ["Metric", "Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]
    values  = ["Value"] + [f"{float(metrics_dict.get(k, float('nan'))):.4f}" for k in headers[1:]]
    widths = [max(len(h), len(v)) + 2 for h, v in zip(headers, values)]
    fmt = "".join("{" + f":{w}" + "}" for w in widths)
    sep = "-" * sum(widths)
    print(fmt.format(*headers))
    print(sep)
    print(fmt.format(*values))


def print_improvement_table(metrics_fused, metrics_dmon_only):
    print("\n=== 改进（融合 - DMon-only） ===")
    rows = []
    for k in ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]:
        vf = float(metrics_fused.get(k, float('nan')))
        vd = float(metrics_dmon_only.get(k, float('nan')))
        dv = (vf - vd) if (not math.isnan(vf) and not math.isnan(vd)) else float('nan')
        rows.append([k, f"{dv:+.4f}"])
    headers = ["Metric", "Δ(Abs)"]
    w0 = max(len(headers[0]), max(len(r[0]) for r in rows)) + 2
    w1 = max(len(headers[1]), max(len(r[1]) for r in rows)) + 2
    fmt = "{" + f":{w0}" + "}" + "{" + f":{w1}" + "}"
    sep = "-" * (w0 + w1)
    print(fmt.format(*headers))
    print(sep)
    for r in rows:
        print(fmt.format(*r))


def avg_metrics(dict_list):
    """对若干指标字典取均值（跳过为 None 的条目）。"""
    if not dict_list:
        return {}
    keys = ["Accuracy", "Precision", "Recall", "F1", "AUC", "MCC"]
    out = {}
    for k in keys:
        vals = [d[k] for d in dict_list if d is not None and k in d]
        out[k] = float(np.mean(vals)) if len(vals) > 0 else float('nan')
    return out

# =============
# 图加载与检查
# =============

def load_and_check_graph(path: str):
    print("正在加载图数据...")
    graphs, _ = dgl.load_graphs(path)
    g = graphs[0]
    print("图加载成功!")
    print("节点类型:", g.ntypes)
    print("边类型:", g.etypes)

    for ntype in g.ntypes:
        print(f"{ntype} 节点数量: {g.num_nodes(ntype)}")

    assert 'features' in g.nodes['file'].data, "file 节点缺少 'features'"
    assert 'features' in g.nodes['author'].data, "author 节点缺少 'features'"

    f_file = g.nodes['file'].data['features']
    f_author = g.nodes['author'].data['features']

    n_file = g.num_nodes('file')
    n_author = g.num_nodes('author')

    print(f"file 特征形状: {tuple(f_file.shape)}")
    print(f"author 特征形状: {tuple(f_author.shape)}")
    print('file nodes / feat rows =', n_file, '/', f_file.shape[0])
    print('author nodes / feat rows =', n_author, '/', f_author.shape[0])

    if (f_file.shape[0] == n_author) and (f_author.shape[0] == n_file):
        print('[WARN] 侦测到 file/author 特征疑似放反，执行自动交换。')
        tmp = f_file.clone()
        g.nodes['file'].data['features'] = f_author
        g.nodes['author'].data['features'] = tmp

    assert g.nodes['file'].data['features'].shape[0] == n_file,   "file 特征行数与节点数不一致"
    assert g.nodes['author'].data['features'].shape[0] == n_author, "author 特征行数与节点数不一致"

    for mask_type in ['train_mask', 'val_mask', 'test_mask']:
        if mask_type in g.nodes['file'].data:
            m = g.nodes['file'].data[mask_type]
            print(f"{mask_type}: {m.sum().item()} 个节点")
        else:
            print(f"[WARN] 缺少 {mask_type}")

    assert 'label' in g.nodes['file'].data, "file 节点缺少 'label'"
    return g

# =============
# 掩码重划分（固定种子）
# =============

def repartition_masks(g, new_val_k=10, seed=3407):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    file_n = g.num_nodes('file')
    train_mask = g.nodes['file'].data.get('train_mask', torch.zeros(file_n, dtype=torch.bool))
    val_mask   = g.nodes['file'].data.get('val_mask',   torch.zeros(file_n, dtype=torch.bool))
    test_mask  = g.nodes['file'].data.get('test_mask',  torch.zeros(file_n, dtype=torch.bool))

    old_train = int(train_mask.sum().item())
    old_val   = int(val_mask.sum().item())
    old_test  = int(test_mask.sum().item())

    new_train_mask = train_mask | val_mask

    test_idx = torch.nonzero(test_mask, as_tuple=True)[0]
    assert test_idx.numel() >= new_val_k, f"测试集样本不足以划分新验证集：需要 {new_val_k}，当前 {test_idx.numel()}"

    g_cpu = torch.Generator(device='cpu')
    g_cpu.manual_seed(seed)
    perm = torch.randperm(test_idx.numel(), generator=g_cpu)
    picked = test_idx[perm[:new_val_k]]
    remain = test_idx[perm[new_val_k:]]

    new_val_mask  = torch.zeros(file_n, dtype=torch.bool)
    new_test_mask = torch.zeros(file_n, dtype=torch.bool)
    new_val_mask[picked]  = True
    new_test_mask[remain] = True

    g.nodes['file'].data['train_mask'] = new_train_mask.to(g.device)
    g.nodes['file'].data['val_mask']   = new_val_mask.to(g.device)
    g.nodes['file'].data['test_mask']  = new_test_mask.to(g.device)

    print("\n=== 掩码重划分完成（固定随机种子） ===")
    print(f"随机种子: {seed}")
    print(f"原划分  ->  train={old_train:>3}, val={old_val:>3}, test={old_test:>3}")
    print(f"新划分  ->  train={int(new_train_mask.sum().item()):>3}, val={int(new_val_mask.sum().item()):>3}, test={int(new_test_mask.sum().item()):>3}")

    return picked.detach().cpu().numpy().tolist()

# =============
# 模型定义（HAN / 对比分类器 / DMon分类器）
# =============

class HANLayer(nn.Module):
    def __init__(self, in_dim_dict, out_dim, num_heads, dropout=0.6):
        super().__init__()
        fdim, adim = in_dim_dict['file'], in_dim_dict['author']
        self.gat_layers = nn.ModuleDict({
            'fa': dglnn.GATConv((fdim, adim), out_dim, num_heads,
                                 feat_drop=dropout, attn_drop=dropout,
                                 activation=F.elu, allow_zero_in_degree=True),
            'af': dglnn.GATConv((adim, fdim), out_dim, num_heads,
                                 feat_drop=dropout, attn_drop=dropout,
                                 activation=F.elu, allow_zero_in_degree=True),
            'file_self': dglnn.GATConv(fdim, out_dim, num_heads,
                                       feat_drop=dropout, attn_drop=dropout,
                                       activation=F.elu, allow_zero_in_degree=True),
            'author_self': dglnn.GATConv(adim, out_dim, num_heads,
                                         feat_drop=dropout, attn_drop=dropout,
                                         activation=F.elu, allow_zero_in_degree=True),
        })
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, h_dict):
        assert h_dict['file'].shape[0] == g.num_nodes('file')
        assert h_dict['author'].shape[0] == g.num_nodes('author')

        h_out = {}
        if 'fa' in g.etypes:
            h = self.gat_layers['fa'](g['fa'], (h_dict['file'], h_dict['author']))
            h_out['author_fa'] = h.view(h.shape[0], -1)
        if 'af' in g.etypes:
            h = self.gat_layers['af'](g['af'], (h_dict['author'], h_dict['file']))
            h_out['file_af'] = h.view(h.shape[0], -1)
        if 'file_self' in g.etypes:
            h = self.gat_layers['file_self'](g['file_self'], h_dict['file'])
            h_out['file_self'] = h.view(h.shape[0], -1)
        if 'author_self' in g.etypes:
            h = self.gat_layers['author_self'](g['author_self'], h_dict['author'])
            h_out['author_self'] = h.view(h.shape[0], -1)
        return h_out

class HAN(nn.Module):
    def __init__(self, in_dim_dict, hidden_dim, out_dim, num_heads, dropout=0.6):
        super().__init__()
        self.layer1 = HANLayer(in_dim_dict, hidden_dim, num_heads, dropout)
        layer2_in_dim = {'file': hidden_dim * num_heads, 'author': hidden_dim * num_heads}
        self.layer2 = HANLayer(layer2_in_dim, hidden_dim, num_heads, dropout)
        layer3_in_dim = {'file': hidden_dim * num_heads, 'author': hidden_dim * num_heads}
        self.layer3 = HANLayer(layer3_in_dim, out_dim, 1, dropout)
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim * num_heads, out_dim)

    def forward(self, g, h_dict):
        h1 = self.layer1(g, h_dict)
        h1 = {k: self.dropout(v) for k, v in h1.items()}

        h2_in = {
            'file': h1.get('file_af', h1.get('file_self', h_dict['file'])),
            'author': h1.get('author_fa', h1.get('author_self', h_dict['author'])),
        }
        h2 = self.layer2(g, h2_in)
        h2 = {k: self.dropout(v) for k, v in h2.items()}

        h3_in = {
            'file': h2.get('file_af', h2.get('file_self', h2_in['file'])),
            'author': h2.get('author_fa', h2.get('author_self', h2_in['author'])),
        }
        h3 = self.layer3(g, h3_in)

        if 'file_af' in h3:
            return h3['file_af']
        if 'file_self' in h3:
            return h3['file_self']
        return self.output_proj(h_dict['file'])

class ContrastiveClassifier(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, h1, h2):
        return self.classifier(torch.cat([h1, h2], dim=1))

class DMonClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=2):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return F.log_softmax(self.lin2(F.relu(self.lin1(x))), dim=-1)

class ModelArgs:
    def __init__(self):
        self.hidden_dim = 128
        self.num_heads = 4
        self.dropout = 0.6
        self.lr = 0.001
        self.weight_decay = 1e-4
        self.batch_size = 10
        self.patience = 20
        self.mlp_hidden = 64

# =============
# 训练器（阶段一）
# =============

class HANTrainer:
    def __init__(self, g, model_args, device):
        self.g = g
        self.device = device
        self.model_args = model_args

        file_feat_dim = g.nodes['file'].data['features'].shape[1]
        author_feat_dim = g.nodes['author'].data['features'].shape[1]
        in_dim_dict = {'file': file_feat_dim, 'author': author_feat_dim}

        self.han = HAN(in_dim_dict, model_args.hidden_dim, model_args.hidden_dim,
                       model_args.num_heads, model_args.dropout).to(device)
        self.contrastive_classifier = ContrastiveClassifier(model_args.hidden_dim).to(device)
        self.feature_fusion_weight = nn.Parameter(torch.tensor(0.0))

        self.optimizer = torch.optim.Adam(
            list(self.han.parameters()) + list(self.contrastive_classifier.parameters()) + [self.feature_fusion_weight],
            lr=model_args.lr, weight_decay=model_args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.8)

    def get_fused_features(self, han_features, dmon_features):
        w = torch.sigmoid(self.feature_fusion_weight)
        fused = (1 - w) * han_features + w * dmon_features
        return fused, float(w.item())

    @staticmethod
    def _dist_loss(anchor, sample, similarity, pos=True):
        cos = F.cosine_similarity(anchor, sample, dim=1)
        if pos:
            dist = 1 - cos
            return (dist * similarity).mean()
        else:
            dist = cos
            return (dist * (1 - similarity)).mean()

    def train_epoch(self):
        self.han.train()
        self.contrastive_classifier.train()
        total_loss, total_steps = 0.0, 0

        train_mask = self.g.nodes['file'].data['train_mask']
        train_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
        np.random.shuffle(train_indices)

        batch_size = self.model_args.batch_size
        for i in range(0, len(train_indices), batch_size):
            batch = train_indices[i:i + batch_size]
            if not batch:
                continue

            h_dict = {
                'file': self.g.nodes['file'].data['features'],
                'author': self.g.nodes['author'].data['features'],
            }
            han_emb = self.han(self.g, h_dict)
            dmon_feat = self.g.nodes['file'].data['features']
            fused, w = self.get_fused_features(han_emb, dmon_feat)

            batch_loss = 0.0
            batch_nodes = 0
            for idx in batch:
                anchor = fused[idx].unsqueeze(0)

                pos_idx = self.g.nodes['file'].data['positive_indices'][idx]
                pos_sim = self.g.nodes['file'].data['positive_similarities'][idx]
                pos_len = int(self.g.nodes['file'].data['positive_lengths'][idx].item())

                neg_idx = self.g.nodes['file'].data['negative_indices'][idx]
                neg_sim = self.g.nodes['file'].data['negative_similarities'][idx]
                neg_len = int(self.g.nodes['file'].data['negative_lengths'][idx].item())

                node_loss, node_cnt = 0.0, 0
                if pos_len > 0:
                    mask = (pos_idx[:pos_len] != -1)
                    v_idx = pos_idx[:pos_len][mask]
                    v_sim = pos_sim[:pos_len][mask]
                    for pi, ps in zip(v_idx, v_sim):
                        pe = fused[pi].unsqueeze(0)
                        dist = self._dist_loss(anchor, pe, ps, pos=True)
                        pred = self.contrastive_classifier(anchor, pe)
                        cls = F.binary_cross_entropy(pred, torch.ones_like(pred))
                        node_loss += (dist + cls)
                        node_cnt += 1

                if neg_len > 0:
                    mask = (neg_idx[:neg_len] != -1)
                    v_idx = neg_idx[:neg_len][mask]
                    v_sim = neg_sim[:neg_len][mask]
                    for ni, ns in zip(v_idx, v_sim):
                        ne = fused[ni].unsqueeze(0)
                        dist = self._dist_loss(anchor, ne, ns, pos=False)
                        pred = self.contrastive_classifier(anchor, ne)
                        cls = F.binary_cross_entropy(pred, torch.zeros_like(pred))
                        node_loss += (dist + cls)
                        node_cnt += 1

                if node_cnt > 0:
                    batch_loss += node_loss / node_cnt
                    batch_nodes += 1

            if batch_nodes > 0:
                self.optimizer.zero_grad()
                loss = batch_loss / batch_nodes
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_steps += 1

        return (total_loss / total_steps if total_steps > 0 else 0.0), w

    def validate(self):
        self.han.eval()
        self.contrastive_classifier.eval()
        total_loss, total_samples = 0.0, 0
        val_mask = self.g.nodes['file'].data['val_mask']
        val_indices = val_mask.nonzero(as_tuple=True)[0].tolist()

        with torch.no_grad():
            h_dict = {
                'file': self.g.nodes['file'].data['features'],
                'author': self.g.nodes['author'].data['features'],
            }
            han_emb = self.han(self.g, h_dict)
            dmon_feat = self.g.nodes['file'].data['features']
            fused, w = self.get_fused_features(han_emb, dmon_feat)

            for idx in val_indices:
                anchor = fused[idx].unsqueeze(0)
                pos_idx = self.g.nodes['file'].data['positive_indices'][idx]
                pos_sim = self.g.nodes['file'].data['positive_similarities'][idx]
                pos_len = int(self.g.nodes['file'].data['positive_lengths'][idx].item())
                neg_idx = self.g.nodes['file'].data['negative_indices'][idx]
                neg_sim = self.g.nodes['file'].data['negative_similarities'][idx]
                neg_len = int(self.g.nodes['file'].data['negative_lengths'][idx].item())

                node_loss, node_cnt = 0.0, 0
                if pos_len > 0:
                    mask = (pos_idx[:pos_len] != -1)
                    v_idx = pos_idx[:pos_len][mask]
                    v_sim = pos_sim[:pos_len][mask]
                    for pi, ps in zip(v_idx, v_sim):
                        pe = fused[pi].unsqueeze(0)
                        dist = self._dist_loss(anchor, pe, ps, pos=True)
                        pred = self.contrastive_classifier(anchor, pe)
                        cls = F.binary_cross_entropy(pred, torch.ones_like(pred))
                        node_loss += (dist + cls)
                        node_cnt += 1
                if neg_len > 0:
                    mask = (neg_idx[:neg_len] != -1)
                    v_idx = neg_idx[:neg_len][mask]
                    v_sim = neg_sim[:neg_len][mask]
                    for ni, ns in zip(v_idx, v_sim):
                        ne = fused[ni].unsqueeze(0)
                        dist = self._dist_loss(anchor, ne, ns, pos=False)
                        pred = self.contrastive_classifier(anchor, ne)
                        cls = F.binary_cross_entropy(pred, torch.zeros_like(pred))
                        node_loss += (dist + cls)
                        node_cnt += 1

                if node_cnt > 0:
                    total_loss += float((node_loss / node_cnt).item())
                    total_samples += 1

        return (total_loss / total_samples if total_samples > 0 else float('inf')), w

    def train(self, epochs):
        print("开始训练 HAN 模型（阶段一：对比学习）...")
        best_val, patience = float('inf'), 0
        for ep in range(epochs):
            tr_loss, tr_w = self.train_epoch()
            va_loss, va_w = self.validate()
            self.scheduler.step()
            print(
                f"Epoch {ep+1:>3}/{epochs:<3} | "
                f"TrainLoss={tr_loss:7.4f} (w={tr_w:5.3f})  ||  "
                f"ValLoss={va_loss:7.4f} (w={va_w:5.3f})"
            )
            if va_loss < best_val:
                best_val, patience = va_loss, 0
                torch.save({
                    'han_state_dict': self.han.state_dict(),
                    'contrastive_classifier_state_dict': self.contrastive_classifier.state_dict(),
                    'feature_fusion_weight': self.feature_fusion_weight.detach().cpu(),
                }, 'best_han_model_tmp.pth')
                print(f"  [保存最佳模型] 验证损失: {best_val:.4f}")
            else:
                patience += 1
            if patience >= self.model_args.patience:
                print(f"早停于第 {ep+1} 轮")
                break
        print("阶段一训练完成！")

# =============
# DMon 分类器提取
# =============

def extract_dmon_classifier(dmon_model, input_dim, hidden_dim, output_dim=2):
    clf = DMonClassifier(input_dim, hidden_dim, output_dim).to(device)
    d_weights = dict(dmon_model.named_parameters())
    copied = []
    for k in ['lin1.weight', 'lin1.bias', 'lin2.weight', 'lin2.bias']:
        if k in d_weights:
            src = d_weights[k].data
            layer_name, param_name = k.split('.')
            dst = getattr(clf, layer_name)._parameters[param_name]
            if src.shape == dst.shape:
                with torch.no_grad():
                    dst.copy_(src)
                copied.append(k)
    if copied:
        print("DMon 分类器参数已复制:", copied)
    else:
        print("[WARN] DMon 分类器权重形状不匹配或未找到，使用随机初始化。")
    return clf

# =============
# 评估函数
# =============

def evaluate_pair_classifier(g, han_trainer):
    print("\n=== 使用正负样本对标签评估对比分类器（测试集） ===")
    test_mask = g.nodes['file'].data['test_mask']
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()

    with torch.no_grad():
        h_dict = {'file': g.nodes['file'].data['features'], 'author': g.nodes['author'].data['features']}
        han_emb = han_trainer.han(g, h_dict)
        dmon_feat = g.nodes['file'].data['features']
        fused, _ = han_trainer.get_fused_features(han_emb, dmon_feat)

        y_true, y_prob, y_pred = [], [], []
        total_pairs = 0
        for idx in test_indices:
            anchor = fused[idx].unsqueeze(0)
            # 正样本
            pos_idx = g.nodes['file'].data['positive_indices'][idx]
            pos_len = int(g.nodes['file'].data['positive_lengths'][idx].item())
            if pos_len > 0:
                mask = (pos_idx[:pos_len] != -1)
                v_idx = pos_idx[:pos_len][mask]
                for pi in v_idx:
                    pe = fused[int(pi.item())].unsqueeze(0)
                    p = han_trainer.contrastive_classifier(anchor, pe).item()
                    y_true.append(1); y_prob.append(p); y_pred.append(1 if p >= 0.5 else 0)
                    total_pairs += 1
            # 负样本
            neg_idx = g.nodes['file'].data['negative_indices'][idx]
            neg_len = int(g.nodes['file'].data['negative_lengths'][idx].item())
            if neg_len > 0:
                mask = (neg_idx[:neg_len] != -1)
                v_idx = neg_idx[:neg_len][mask]
                for ni in v_idx:
                    ne = fused[int(ni.item())].unsqueeze(0)
                    p = han_trainer.contrastive_classifier(anchor, ne).item()
                    y_true.append(0); y_prob.append(p); y_pred.append(1 if p >= 0.5 else 0)
                    total_pairs += 1

        if total_pairs == 0:
            print("[WARN] 测试集没有可评估的正/负样本对，跳过对比分类器评估。")
            return None

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        print_metrics_table("Pair Classifier on Test Pairs", metrics)
        return metrics


def fine_tune_with_dmon(g, han_trainer, dmon_model, model_args, epochs=50):
    print("\n开始阶段二微调（验证集：真实标签分类损失 + 样本对度量损失）...")

    clf = extract_dmon_classifier(
        dmon_model,
        input_dim=model_args.hidden_dim,
        hidden_dim=model_args.mlp_hidden,
        output_dim=2,
    )

    params = list(han_trainer.han.parameters()) + [han_trainer.feature_fusion_weight] + list(clf.parameters())
    optimizer = torch.optim.Adam(params, lr=model_args.lr, weight_decay=model_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    criterion_cls = nn.NLLLoss()

    val_mask = g.nodes['file'].data['val_mask']
    val_indices = val_mask.nonzero(as_tuple=True)[0].tolist()
    assert len(val_indices) > 0, "验证集为空，无法进行阶段二微调"

    for ep in range(epochs):
        han_trainer.han.train(); clf.train()
        total_cls, total_pair, node_cnt_sum = 0.0, 0.0, 0

        h_dict = {'file': g.nodes['file'].data['features'], 'author': g.nodes['author'].data['features']}
        han_emb = han_trainer.han(g, h_dict)
        dmon_feat = g.nodes['file'].data['features']
        fused, w = han_trainer.get_fused_features(han_emb, dmon_feat)

        loss_epoch = 0.0
        optimizer.zero_grad()
        for idx in val_indices:
            anchor = fused[idx].unsqueeze(0)

            y = g.nodes['file'].data['label'][idx].long()
            assert int(y.item()) in (0, 1), "验证集中存在无效标签（需为 0/1）"
            out = clf(anchor)
            cls_loss = criterion_cls(out, y.view(1))

            pos_idx = g.nodes['file'].data['positive_indices'][idx]
            pos_sim = g.nodes['file'].data['positive_similarities'][idx]
            pos_len = int(g.nodes['file'].data['positive_lengths'][idx].item())
            neg_idx = g.nodes['file'].data['negative_indices'][idx]
            neg_sim = g.nodes['file'].data['negative_similarities'][idx]
            neg_len = int(g.nodes['file'].data['negative_lengths'][idx].item())

            pair_loss_sum, pair_cnt = 0.0, 0
            if pos_len > 0:
                mask = (pos_idx[:pos_len] != -1)
                v_idx = pos_idx[:pos_len][mask]
                v_sim = pos_sim[:pos_len][mask]
                for pi, ps in zip(v_idx, v_sim):
                    pe = fused[pi].unsqueeze(0)
                    pair_loss_sum += han_trainer._dist_loss(anchor, pe, ps, pos=True)
                    pair_cnt += 1
            if neg_len > 0:
                mask = (neg_idx[:neg_len] != -1)
                v_idx = neg_idx[:neg_len][mask]
                v_sim = neg_sim[:neg_len][mask]
                for ni, ns in zip(v_idx, v_sim):
                    ne = fused[ni].unsqueeze(0)
                    pair_loss_sum += han_trainer._dist_loss(anchor, ne, ns, pos=False)
                    pair_cnt += 1

            pair_loss = (pair_loss_sum / pair_cnt) if pair_cnt > 0 else 0.0
            node_loss = cls_loss + pair_loss
            loss_epoch += node_loss

            total_cls  += float(cls_loss.item())
            total_pair += float(pair_loss if isinstance(pair_loss, float) else pair_loss.item())
            node_cnt_sum += 1

        if node_cnt_sum > 0:
            (loss_epoch / node_cnt_sum).backward()
            optimizer.step()
        scheduler.step()

        avg_cls  = total_cls  / max(1, node_cnt_sum)
        avg_pair = total_pair / max(1, node_cnt_sum)
        avg_tot  = avg_cls + avg_pair
        print(
            f"FT Epoch {ep+1:>3}/{epochs:<3} | "
            f"ClsLoss={avg_cls:7.4f}  PairLoss={avg_pair:7.4f}  Total={avg_tot:7.4f} (w={w:5.3f})"
        )

    return clf


def evaluate_on_test_set(g, dmon_model, han_trainer, model_args, clf=None):
    print("=== 在测试集上用 DMon 分类器评估（输入为融合特征） ===")
    test_mask = g.nodes['file'].data['test_mask']
    test_idx = test_mask.nonzero(as_tuple=True)[0].tolist()
    test_labels = g.nodes['file'].data['label'][test_idx]

    print(f"测试集节点数量: {len(test_idx)}")
    print(f"测试集标签分布: R={int((test_labels==0).sum())}, U={int((test_labels==1).sum())}")

    if clf is None:
        clf = extract_dmon_classifier(
            dmon_model,
            input_dim=model_args.hidden_dim,  # 特征维度与 HAN 输出一致
            hidden_dim=model_args.mlp_hidden,
            output_dim=2,
        )

    clf.eval()
    with torch.no_grad():
        h_dict = {
            'file': g.nodes['file'].data['features'],
            'author': g.nodes['author'].data['features'],
        }
        # 先一次性前向得到 HAN 嵌入
        han_emb = han_trainer.han(g, h_dict)
        dmon_feat = g.nodes['file'].data['features']

        # ---- Fused ----
        fused, w = han_trainer.get_fused_features(han_emb, dmon_feat)
        X_fused = fused[test_idx].to(device)
        out_f = clf(X_fused)
        prob_f = torch.exp(out_f)
        pred_f = torch.argmax(out_f, dim=1)
        prob1_f = prob_f[:, 1]

        y_true = test_labels.cpu().numpy()
        y_pred_f = pred_f.cpu().numpy()
        y_prob_f = prob1_f.cpu().numpy()
        metrics_fused = compute_metrics(y_true, y_pred_f, y_prob_f)
        print_metrics_table("DMon Classifier on Real Labels (Fused Features)", metrics_fused)
        print("详细分类报告(融合特征):", classification_report(test_labels.cpu(), pred_f.cpu(), target_names=['Readable','Unreadable']))

    # ---- DMon-only ----
    print("=== 对比：仅用 DMon 特征 ===")
    with torch.no_grad():
        Xd = g.nodes['file'].data['features'][test_idx].to(device)
        out_d = clf(Xd)
        prob_d = torch.exp(out_d)
        pred_d = torch.argmax(out_d, dim=1)
        prob1_d = prob_d[:, 1]

        y_pred_d = pred_d.cpu().numpy()
        y_prob_d = prob1_d.cpu().numpy()
        metrics_dmon_only = compute_metrics(y_true, y_pred_d, y_prob_d)
        print_metrics_table("DMon Classifier on Real Labels (DMon-only)", metrics_dmon_only)

    # ---- HAN-only ----
    print("=== 对比：仅用 HAN 特征 ===")
    with torch.no_grad():
        Xh = han_emb[test_idx].to(device)
        out_h = clf(Xh)
        prob_h = torch.exp(out_h)
        pred_h = torch.argmax(out_h, dim=1)
        prob1_h = prob_h[:, 1]

        y_pred_h = pred_h.cpu().numpy()
        y_prob_h = prob1_h.cpu().numpy()
        metrics_han_only = compute_metrics(y_true, y_pred_h, y_prob_h)
        print_metrics_table("DMon Classifier on Real Labels (HAN-only)", metrics_han_only)

    # 仅打印一次改进（Fused - DMon-only），标题与函数一致
    print_improvement_table(metrics_fused, metrics_dmon_only)
    return metrics_fused, metrics_dmon_only, metrics_han_only

# =============
# 单次完整实验（给定随机种子）
# =============

# LI
# def run_experiment(seed: int, graph_path: str, stage1_epochs=100, stage2_epochs=50):
# WYh
def run_experiment(seed: int, graph_path: str, stage1_epochs=5, stage2_epochs=5):
    print("\n" + "=="*36)
    print(f"开始运行单次实验，随机种子 = {seed}")
    print("=="*36)

    set_global_seed(seed)

    # 每个 seed 都独立加载图，避免进程内状态污染
    g = load_and_check_graph(graph_path)
    g = g.to(device)

    # 划分新掩码（10 验证，剩余为新测试）
    _ = repartition_masks(g, new_val_k=10, seed=seed)

    # 加载 DMon
    print("\n正在加载 DMon 模型...")
    from Preprocess.Code.DMon import DMon
    from Preprocess.Code.configures import data_args, model_args as dmon_model_args

    dmon_model = DMon(data_args, dmon_model_args)
    # MODEL_PATH = "Preprocess/Code/checkpoint/Models/dmon_best_3.pth"
    # LI
    # MODEL_PATH = "Preprocess/Code/checkpoint/Models/dmon_best_3.pth"
    # WYH
    MODEL_PATH = "Preprocess/Code/checkpoint/code_hh/dmon_best_3.pth"
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    dmon_model.update_state_dict(checkpoint['net'])
    dmon_model.to(device)
    dmon_model.eval()
    print("DMon 模型加载成功!")

    # 阶段一训练
    margs = ModelArgs()
    trainer = HANTrainer(g, margs, device)
    trainer.train(epochs=stage1_epochs)

    # 载入阶段一最佳（用临时文件，不跨 seed 复用）
    ckpt = torch.load('best_han_model_tmp.pth', map_location=device)
    trainer.han.load_state_dict(ckpt['han_state_dict'])
    trainer.contrastive_classifier.load_state_dict(ckpt['contrastive_classifier_state_dict'])
    trainer.feature_fusion_weight.data = ckpt['feature_fusion_weight'].to(device)

    # 评估样本对分类器
    pair_metrics = evaluate_pair_classifier(g, trainer)

    # 阶段二微调
    clf = fine_tune_with_dmon(g, trainer, dmon_model, margs, epochs=stage2_epochs)

    # 测试集评估（融合 vs DMon-only）
    metrics_fused, metrics_dmon_only, metrics_han_only = evaluate_on_test_set(g, dmon_model, trainer, margs, clf=clf)

    return pair_metrics, metrics_fused, metrics_dmon_only, metrics_han_only

# =============
# 主入口：多随机种子实验 & 汇总
# =============

def main():
    graph_path = "code_readability_graph_with_authors.bin"

    # 五个随机种子（可按需修改）
    seeds = [2023, 3407, 42, 7, 114514]

    all_pair = []
    all_fused = []
    all_dmon = []
    all_han  = []

    for sd in seeds:
        try:
            pair_m, fused_m, dmon_m, han_m = run_experiment(sd, graph_path)
            if pair_m is not None:
                all_pair.append(pair_m)
            all_fused.append(fused_m)
            all_dmon.append(dmon_m)
            all_han.append(han_m)
        except Exception as e:
            print(f"[ERROR] 种子 {sd} 运行失败: {e}")

    # 汇总均值
    avg_pair = avg_metrics(all_pair)
    avg_fused = avg_metrics(all_fused)
    avg_dmon  = avg_metrics(all_dmon)
    avg_han   = avg_metrics(all_han)

    print("\n" + "=="*40)
    print("多随机种子结果汇总（5 seeds）")
    print("=="*40)

    if all_pair:
        print_metrics_table("平均指标 - Pair Classifier (5 seeds)", avg_pair)
    else:
        print("[WARN] 无 Pair Classifier 结果可汇总（可能没有样本对）")

    print_metrics_table("平均指标 - DMon Classifier (Fused Features, 5 seeds)", avg_fused)
    print_metrics_table("平均指标 - DMon Classifier (DMon-only, 5 seeds)", avg_dmon)
    print_metrics_table("平均指标 - DMon Classifier (HAN-only, 5 seeds)", avg_han)

    # 平均改进
    if all_fused and all_dmon:
        print("\n=== 平均改进（Fused - DMon-only, 5 seeds） ===")
        avg_improve = {k: avg_fused[k] - avg_dmon[k] for k in avg_fused}
        print_improvement_table(avg_fused, avg_dmon)

    print("\n=== 全部实验完成 ===")


if __name__ == "__main__":
    main()
