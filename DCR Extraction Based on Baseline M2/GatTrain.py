
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

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout=0.6):
        super().__init__()
        self.gat_conv = dglnn.GATConv(
            in_dim,
            out_dim,
            num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            activation=F.elu,
            allow_zero_in_degree=True
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        h = self.gat_conv(g, x)
        h = h.view(h.shape[0], -1)  # [N, out_dim * num_heads]
        return self.dropout(h)


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, dropout=0.6):
        super().__init__()
        self.layer1 = GATLayer(in_dim, hidden_dim, num_heads, dropout)
        self.layer2 = GATLayer(hidden_dim * num_heads, hidden_dim, num_heads, dropout)
        self.layer3 = dglnn.GATConv(
            hidden_dim * num_heads,
            out_dim,
            num_heads=1,
            feat_drop=dropout,
            attn_drop=dropout,
            activation=None,
            allow_zero_in_degree=True
        )
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(hidden_dim * num_heads, out_dim)

    def forward(self, g, x):
        h = self.layer1(g, x)
        h = self.layer2(g, h)
        h = self.dropout(h)
        h = self.layer3(g, h).squeeze(1)  # [N, out_dim]
        return h

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GATTrainer:
    def __init__(self, g, model_args, device):
        self.g_full = g  # 保留完整异构图，用于正负样本索引
        self.device = device
        self.model_args = model_args

        # 构造 file 节点的同构图
        self.g = dgl.node_type_subgraph(g, ['file']).to(device)
        self.x = self.g.ndata['features'].to(device)
        in_dim = self.x.shape[1]

        # GAT 模型
        self.gat = GAT(
            in_dim=in_dim,
            hidden_dim=model_args.hidden_dim,
            out_dim=model_args.hidden_dim,
            num_heads=model_args.num_heads,
            dropout=model_args.dropout
        ).to(device)

        self.contrastive_classifier = ContrastiveClassifier(model_args.hidden_dim).to(device)
        self.feature_fusion_weight = nn.Parameter(torch.tensor(0.0, device=device))

        self.optimizer = torch.optim.Adam(
            list(self.gat.parameters()) +
            list(self.contrastive_classifier.parameters()) +
            [self.feature_fusion_weight],
            lr=model_args.lr,
            weight_decay=model_args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.8)

    def get_fused_features(self, gat_features, dmon_features):
        w = torch.sigmoid(self.feature_fusion_weight)
        fused = (1 - w) * gat_features + w * dmon_features
        return fused, float(w.item())

    @staticmethod
    def _dist_loss(anchor, sample, similarity, pos=True):
        cos = F.cosine_similarity(anchor, sample, dim=1)
        if pos:
            return ((1 - cos) * similarity).mean()
        else:
            return (cos * (1 - similarity)).mean()

    def train_epoch(self):
        self.gat.train()
        self.contrastive_classifier.train()
        total_loss, total_steps = 0.0, 0

        # train_mask 仍用完整图的 file 节点
        train_mask = self.g_full.nodes['file'].data['train_mask']
        train_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
        np.random.shuffle(train_indices)

        batch_size = self.model_args.batch_size
        for i in range(0, len(train_indices), batch_size):
            batch = train_indices[i:i + batch_size]
            if not batch:
                continue

            # GAT 只用同构子图
            gat_emb = self.gat(self.g, self.x)
            dmon_feat = self.x
            fused, w = self.get_fused_features(gat_emb, dmon_feat)

            batch_loss, batch_nodes = 0.0, 0
            for idx in batch:
                anchor = fused[idx].unsqueeze(0)

                pos_idx = self.g_full.nodes['file'].data['positive_indices'][idx]
                pos_sim = self.g_full.nodes['file'].data['positive_similarities'][idx]
                pos_len = int(self.g_full.nodes['file'].data['positive_lengths'][idx].item())

                neg_idx = self.g_full.nodes['file'].data['negative_indices'][idx]
                neg_sim = self.g_full.nodes['file'].data['negative_similarities'][idx]
                neg_len = int(self.g_full.nodes['file'].data['negative_lengths'][idx].item())

                node_loss, node_cnt = 0.0, 0

                if pos_len > 0:
                    mask = (pos_idx[:pos_len] != -1)
                    for pi, ps in zip(pos_idx[:pos_len][mask], pos_sim[:pos_len][mask]):
                        pe = fused[pi].unsqueeze(0)
                        dist = self._dist_loss(anchor, pe, ps, pos=True)
                        pred = self.contrastive_classifier(anchor, pe)
                        cls = F.binary_cross_entropy(pred, torch.ones_like(pred))
                        node_loss += (dist + cls)
                        node_cnt += 1

                if neg_len > 0:
                    mask = (neg_idx[:neg_len] != -1)
                    for ni, ns in zip(neg_idx[:neg_len][mask], neg_sim[:neg_len][mask]):
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
        self.gat.eval()
        self.contrastive_classifier.eval()
        total_loss, total_samples = 0.0, 0

        val_mask = self.g_full.nodes['file'].data['val_mask']
        val_indices = val_mask.nonzero(as_tuple=True)[0].tolist()

        with torch.no_grad():
            gat_emb = self.gat(self.g, self.x)
            dmon_feat = self.x
            fused, w = self.get_fused_features(gat_emb, dmon_feat)

            for idx in val_indices:
                anchor = fused[idx].unsqueeze(0)

                pos_idx = self.g_full.nodes['file'].data['positive_indices'][idx]
                pos_sim = self.g_full.nodes['file'].data['positive_similarities'][idx]
                pos_len = int(self.g_full.nodes['file'].data['positive_lengths'][idx].item())

                neg_idx = self.g_full.nodes['file'].data['negative_indices'][idx]
                neg_sim = self.g_full.nodes['file'].data['negative_similarities'][idx]
                neg_len = int(self.g_full.nodes['file'].data['negative_lengths'][idx].item())

                node_loss, node_cnt = 0.0, 0

                if pos_len > 0:
                    mask = (pos_idx[:pos_len] != -1)
                    for pi, ps in zip(pos_idx[:pos_len][mask], pos_sim[:pos_len][mask]):
                        pe = fused[pi].unsqueeze(0)
                        dist = self._dist_loss(anchor, pe, ps, pos=True)
                        pred = self.contrastive_classifier(anchor, pe)
                        cls = F.binary_cross_entropy(pred, torch.ones_like(pred))
                        node_loss += (dist + cls)
                        node_cnt += 1

                if neg_len > 0:
                    mask = (neg_idx[:neg_len] != -1)
                    for ni, ns in zip(neg_idx[:neg_len][mask], neg_sim[:neg_len][mask]):
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
                    'gat_state_dict': self.gat.state_dict(),
                    'contrastive_classifier_state_dict': self.contrastive_classifier.state_dict(),
                    'feature_fusion_weight': self.feature_fusion_weight.detach().cpu(),
                }, 'best_gat_model_tmp.pth')
                print(f"  [保存最佳模型] 验证损失: {best_val:.4f}")
            else:
                patience += 1
            if patience >= self.model_args.patience:
                print(f"早停于第 {ep+1} 轮")
                break
        print("阶段一训练完成！")

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

    # # 加载 DMon
    # print("\n正在加载 DMon 模型...")
    # from Preprocess.Code.DMon import DMon
    # from Preprocess.Code.configures import data_args, model_args as dmon_model_args

    # dmon_model = DMon(data_args, dmon_model_args)
    # MODEL_PATH = "Preprocess/Code/checkpoint/code_hh/dmon_best_3.pth"
    # checkpoint = torch.load(MODEL_PATH, map_location=device)
    # dmon_model.update_state_dict(checkpoint['net'])
    # dmon_model.to(device)
    # dmon_model.eval()
    # print("DMon 模型加载成功!")

    # 阶段一训练
    margs = ModelArgs()
    trainer = GATTrainer(g, margs, device)
    trainer.train(epochs=stage1_epochs)

    # 载入阶段一最佳（用临时文件，不跨 seed 复用）
    # ckpt = torch.load('best_han_model_tmp.pth', map_location=device)
    # trainer.han.load_state_dict(ckpt['han_state_dict'])
    # trainer.contrastive_classifier.load_state_dict(ckpt['contrastive_classifier_state_dict'])
    # trainer.feature_fusion_weight.data = ckpt['feature_fusion_weight'].to(device)

    # # 评估样本对分类器
    # pair_metrics = evaluate_pair_classifier(g, trainer)

    # # 阶段二微调
    # clf = fine_tune_with_dmon(g, trainer, dmon_model, margs, epochs=stage2_epochs)

    # # 测试集评估（融合 vs DMon-only）
    # metrics_fused, metrics_dmon_only, metrics_han_only = evaluate_on_test_set(g, dmon_model, trainer, margs, clf=clf)

    # return pair_metrics, metrics_fused, metrics_dmon_only, metrics_han_only



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



    print("\n=== 全部实验完成 ===")


if __name__ == "__main__":
    main()
