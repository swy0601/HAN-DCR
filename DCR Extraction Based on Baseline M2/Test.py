import os
import pickle
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef
from Preprocess.Code.DMon import DMon
from Preprocess.Code.DMonClassifier import DMonClassifier
from Preprocess.Code.configures import data_args, model_args
from models.model import Model  # 保留 HAN 模型
from utils import load_test_data

DATA_PATH = "mlkit.pkl"
ORIGINAL_MODEL_PATH = "best_model.pth"  # HAN 模型的保存路径
DMON_MODEL_PATH = "Preprocess/Code/checkpoint/code_hh/dmon_best_4.pth"  # DMon 模型的保存路径


class MyTestDataset(Dataset):
    """
    简单的数据集包装，将 features 和 labels 打包到一起，以便 DataLoader 批量读取。
    """
    def __init__(self, features: torch.Tensor, labels: np.ndarray):
        self.features = features
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_original_model(model_path, device):
    """
    加载原始 HAN 图模型，并用于特征提取。
    """
    args = {
        'hidden_units': 32,
        'num_heads': [8],
        'dropout': 0.5
    }
    model = Model(
        meta_paths=[['fa', 'af']],
        embedding_size=128,
        hidden_size=args['hidden_units'],
        out_size=128,
        num_heads=args['num_heads'],
        dropout=args['dropout']
    ).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"HAN模型加载成功: {model_path}")
    else:
        print(f"未找到原始HAN模型文件: {model_path}")
    model.eval()
    return model


def load_dmon_model(model_path):
    """
    加载 DMon 模型，并初始化分类器。
    """
    try:
        model = DMon(data_args, model_args)
        checkpoint = torch.load(model_path, map_location=model_args.device)
        model_state_dict = model.state_dict()
        checkpoint_state_dict = checkpoint['net']
        checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if 'lin1' in k or 'lin2' in k}
        model_state_dict.update(checkpoint_state_dict)
        model.load_state_dict(model_state_dict)
        model.to(model_args.device)
        print(f"DMon模型加载成功: {model_path}")
        return model.eval()
    except Exception as e:
        print(f"加载DMon模型失败: {str(e)}")
        return None


def main():
    device = torch.device("cpu")
    
    # -------------------- 1. 加载测试数据 --------------------
    g, features, label, _, _, _ = load_test_data(DATA_PATH)
    features = features.to(device)
    g = g.to(device)

    # 处理标签，确保为 numpy 数组
    if isinstance(label, list):
        true_labels = np.array(label)
    else:
        true_labels = label

    # -------------------- 2. 用 HAN 模型提取特征 --------------------
    original_model = load_original_model(ORIGINAL_MODEL_PATH, device)
    with torch.no_grad():
        processed_features = original_model(g, features)  # shape: [N, 128]

    # -------------------- 3. 拼接原始输入特征与 HAN 模型提取的特征 --------------------
    # 直接将原始输入特征（features）与HAN模型提取的特征（processed_features）在特征维度上拼接
    fused_features = torch.cat((features, processed_features), dim=1)  # shape: [N, feature_dim + 128]

    # -------------------- 4. 加载 DMon 模型 --------------------
    dmon_model = load_dmon_model(DMON_MODEL_PATH)
    if dmon_model is None:
        print("DMon 模型加载失败，退出。")
        return

    classifier = DMonClassifier(dmon_model)
    classifier.to(device)

    # -------------------- 5. 构建 DataLoader，分批预测 --------------------
    test_dataset = MyTestDataset(fused_features, true_labels)
    test_loader = DataLoader(test_dataset, batch_size=data_args.batch_size, shuffle=False)

    all_preds_list = []
    all_labels_list = []
    all_probs_list = []

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features = batch_features.to(device)

            # 推理
            preds = classifier(batch_features)  # shape: [batch_size, num_classes]

            # 预测类别
            predicted_labels = preds.argmax(dim=1).cpu().numpy()
            all_preds_list.append(predicted_labels)

            # 真实标签
            all_labels_list.append(batch_labels.cpu().numpy())

            # 获取概率
            probs = torch.exp(preds).cpu().numpy()
            all_probs_list.append(probs)

    # -------------------- 6. 拼接各批次结果，并计算指标 --------------------
    predicted_labels = np.concatenate(all_preds_list, axis=0)
    true_labels = np.concatenate(all_labels_list, axis=0)
    probs = np.concatenate(all_probs_list, axis=0)

    # 计算评价指标
    test_accuracy = accuracy_score(true_labels, predicted_labels)
    test_precision = precision_score(true_labels, predicted_labels, average='macro')
    test_recall = recall_score(true_labels, predicted_labels, average='macro')
    test_f1 = f1_score(true_labels, predicted_labels, average='macro')

    if probs.shape[1] == 2:
        test_auc = roc_auc_score(true_labels, probs[:, 1])
    else:
        test_auc = roc_auc_score(true_labels, probs, multi_class='ovr')

    test_mcc = matthews_corrcoef(true_labels, predicted_labels)

    # 打印评价指标
    print("预测完成。")
    print(f'Test Accuracy: {test_accuracy:.4f}')
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1 Score: {test_f1:.4f}')
    print(f'Test AUC: {test_auc:.4f}')
    print(f'Test MCC: {test_mcc:.4f}')


if __name__ == '__main__':
    main()
