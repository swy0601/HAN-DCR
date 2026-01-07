import torch
from configures import data_args, model_args
from load_dataset import get_dataloader
from DMon import DMon
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef, accuracy_score  # 确保导入所有指标
from sklearn.preprocessing import label_binarize
import os
import numpy as np
import pandas as pd

# 配置设备和路径
model_args.device = "cuda" if torch.cuda.is_available() else "cpu"
data_args.dataset_dir = "../Dataset/Packaged Pkl/input.pkl"

# 加载数据集（仅加载测试集）
try:
    # 这里假设 get_dataloader 函数已经修改为只返回测试集的数据加载器
    dataloader = get_dataloader(data_args, only_test=True)  # 传入标志位，仅加载测试集
    print("测试集样本数:", len(dataloader['test'].dataset))  # 验证数据加载
except Exception as e:
    print("数据加载失败:", str(e))
    exit()


def load_model(model_path):
    try:
        model = DMon(data_args, model_args)
        checkpoint = torch.load(model_path, map_location=model_args.device)
        model.load_state_dict(checkpoint['net'])
        model.to(model_args.device)
        print(f"模型加载成功: {model_path}")
        return model.eval()
    except Exception as e:
        print(f"加载模型 {model_path} 失败:", str(e))
        return None

def evaluate_model(model, dataloader):
    if model is None:
        return {}

    prob_all, label_all = [], []
    try:
        with torch.no_grad():
            for data in dataloader:
                data = data.to(model_args.device)
                pre, _ = model(data.x, data.edge_index, data.batch)
                prob = torch.softmax(pre, dim=-1)
                prob_all.extend(prob.cpu().numpy())
                # 强制转换 labels 为 1D
                labels = data.y.cpu().numpy().squeeze()
                label_all.extend(labels)

        # 转换预测结果
        prob_all = np.array(prob_all)
        pred_all = np.argmax(prob_all, axis=1)
        labels_unique = np.unique(label_all)

        # 计算指标
        acc = accuracy_score(label_all, pred_all)
        f1 = f1_score(label_all, pred_all, average='macro')
        mcc = matthews_corrcoef(label_all, pred_all)
        
        # 修正 AUC 计算
        if len(labels_unique) == 2:  # 二分类
            auc = roc_auc_score(label_all, prob_all[:, 1])
        else:  # 多分类
            label_bin = label_binarize(label_all, classes=labels_unique)
            auc = roc_auc_score(label_bin, prob_all, multi_class='ovr')

        print(f"评估结果: Accuracy: {acc}, F1: {f1}, AUC: {auc}, MCC: {mcc}")
        return {'accuracy': acc, 'f1_score': f1, 'auc': auc, 'mcc': mcc}
    except Exception as e:
        print("评估失败:", str(e))
        return {}


checkpoint_dir = '/root/Preprocess/Code/checkpoint/code_hh/'  # 确保这里是目录路径
model_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('dmon_best_') and f.endswith('.pth')][:5]

print("找到的模型文件:", model_files)

# 确认获取到模型文件后，继续执行评估
results = {}
for model_file in model_files:
    try:
        print(f"\nTesting model: {model_file}")
        model_path = os.path.join(checkpoint_dir, model_file)
        model = load_model(model_path)
        if model is None:
            print(f"模型加载失败: {model_file}")
            continue
        metrics = evaluate_model(model, dataloader['test'])
        if metrics:
            results[model_file] = metrics
        else:
            print(f"{model_file} 评估失败。")
    except Exception as e:
        print(f"Error: {str(e)}")

# 保存结果
if results:
    df = pd.DataFrame.from_dict(results, orient='index')
    df.to_csv("new_dataset_test_results.csv")
    print("\nCSV 文件已保存，内容如下：")
    print(df)
else:
    print("无有效结果可保存。")
