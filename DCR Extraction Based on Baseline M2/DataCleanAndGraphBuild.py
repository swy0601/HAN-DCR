
import hashlib
import argparse
import csv
import json
import pickle
import time
import os
import re
import glob
import numpy as np
import pandas as pd

import dgl
import torch
from git import Repo
from tqdm import tqdm

from torch_geometric.data import Data



GIT_PATH = "dataset/git_repo/mlkit"  # Git 仓库路径
# READABLE_DIR = "Test/Readable"
# UNREADABLE_DIR = "Test/Unreadable"
READABLE_DIR = "Test/Readable"
UNREADABLE_DIR = "Test/Unreadable"

class Author:
    def __init__(self, name, email):
        self.name = name
        self.email = email

    def __str__(self):
        return self.name + " <" + self.email + ">"

    def __repr__(self):
        return self.name + " <" + self.email + ">"

    def __eq__(self, other):
        return self.name == other.name and self.email == other.email

    def __hash__(self):
        return hash(self.name + self.email)


def get_total_file_lines(file_path):
    with open(file_path, "r") as f:
        total_line_size = len(f.readlines())
        return total_line_size


class Git_repo:
    commit_list = []
    authors = {}
    files_data = {}
    author_commit = {}
    file_contribution = {}

    def __init__(self, path):
        self.path = path
        self.repo = Repo(path)
        print("Repo init... Author")
        for i in tqdm(self.repo.iter_commits(), total=len(list(self.repo.iter_commits()))):
            self.commit_list.append(i.hexsha)
            if Author(i.author.name, i.author.email) not in self.author_commit:
                self.author_commit[Author(i.author.name, i.author.email)] = [i.hexsha]
            else:
                self.author_commit[Author(i.author.name, i.author.email)].append(i.hexsha)
        counter = 0
        for author in tqdm(self.author_commit):
            self.authors[author] = counter
            counter += 1
        # Get HEAD commit
        self.head = self.repo.head.commit
        # Get all files in the repo
        self.files = self.repo.tree()
        for file in tqdm(self.files.traverse(), total=len(list(self.files.traverse())), desc="Blame", unit="file"):
            self.files_data[file.path] = counter
            counter += 1

        print("\nRepo init Author finished")
        print("Repo init file")

        # Get HEAD commit
        self.head = self.repo.head.commit
        # Get all files in the repo
        self.files = self.repo.tree()
        # Get blame for each file
        self.blames = {}
        for file in tqdm(self.files.traverse(), total=len(list(self.files.traverse())), desc="Blame", unit="file"):
            if file.type != "tree" and str(file.path).endswith(".java"):
                self.file_contribution[file.path] = {}
                self.blames[file.path] = self.repo.blame(self.head, file.path)
                for commit in self.blames[file.path]:
                    commit_ref = commit[0]
                    lines = commit[1]
                    contibuted_line_size = len(lines)
                    author = Author(commit_ref.author.name, commit_ref.author.email)
                    if author not in self.file_contribution[file.path]:
                        self.file_contribution[file.path][author] = contibuted_line_size
                    else:
                        self.file_contribution[file.path][author] += contibuted_line_size

        print("Repo init file finished")

    def get_branch_name(self):
        return self.repo.active_branch.name

    def get_branch_list(self):
        return self.repo.branches

    def get_commit_message(self, commit_id):
        return self.repo.commit(commit_id).message

    def get_author_email(self, commit_id):
        return self.repo.commit(commit_id).author.email

    def get_contribution_by_author(self, author):
        return self.file_contribution[author]


class Node:
    def __init__(self, id_graph, name, type_data):
        self.id = id_graph
        self.name = name
        self.type = type_data

    def __str__(self):
        return str(self.id) + " " + self.name

    def __repr__(self):
        return str(self.id) + " " + self.name

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Edge:
    def __init__(self, id_graph, source, target, weight):
        self.id = id_graph
        self.source = source
        self.target = target
        self.weight = weight

    def __str__(self):
        return str(self.id) + " " + str(self.source) + " " + str(self.target) + " " + str(self.weight)

    def __repr__(self):
        return str(self.id) + " " + str(self.source) + " " + str(self.target) + " " + str(self.weight)

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)




# 加载Git仓库
git_repo = Git_repo(GIT_PATH)
print('加载 Git 仓库')
contribution = git_repo.file_contribution
authors = git_repo.authors


# 获取两个文件夹中所有 .json 和 .java 文件
readable_json_files = [f for f in os.listdir(READABLE_DIR) if f.endswith('.json')]
unreadable_json_files = [f for f in os.listdir(UNREADABLE_DIR) if f.endswith('.json')]

# readable_java_files = [f for f in os.listdir(READABLE_DIR) if f.endswith('.java')]
# unreadable_java_files = [f for f in os.listdir(UNREADABLE_DIR) if f.endswith('.java')]

print("Json Files in Readable folder:", readable_json_files)
print(" ")
print("Json Files in Unreadable folder:", unreadable_json_files)
print(" ")
# print("Java Files in Readable folder:", readable_java_files)
# print(" ")
# print("Java Files in Unreadable folder:", unreadable_java_files)


# 获取Test文件夹中所有Java文件的basename和完整路径
readable_java_files = {}
unreadable_java_files = {}

# 遍历Readable文件夹中的所有Java文件
for file in os.listdir(READABLE_DIR):
    if file.endswith('.java'):
        file_path = os.path.join(READABLE_DIR, file)
        readable_java_files[file] = file_path  # 使用文件名作为key

# 遍历Unreadable文件夹中的所有Java文件
for file in os.listdir(UNREADABLE_DIR):
    if file.endswith('.java'):
        file_path = os.path.join(UNREADABLE_DIR, file)
        unreadable_java_files[file] = file_path  # 使用文件名作为key

print(f"Readable文件夹中找到 {len(readable_java_files)} 个Java文件")
print(f"Unreadable文件夹中找到 {len(unreadable_java_files)} 个Java文件")





def compare_file_content(git_file_path, test_file_path):
    """比较两个文件的内容是否相同"""
    try:
        with open(git_file_path, 'r', encoding='utf-8', errors='ignore') as f1:
            content1 = f1.read()
        with open(test_file_path, 'r', encoding='utf-8', errors='ignore') as f2:
            content2 = f2.read()
        return content1 == content2
    except Exception as e:
        print(f"比较文件时出错 {git_file_path} vs {test_file_path}: {e}")
        return False


def build_unique_name_mapping(file_paths):
    """
    对 file_paths 中的每个 .java 文件构建唯一名称映射字典。
    如果某个文件的 basename 在多个路径中出现，则第一个保留原名，其余依次加上后缀 _1, _2, ...
    :param file_paths: list，包含多个文件的原始路径
    :return: dict，key 为原始文件路径，value 为唯一文件名（仅文件名部分）
    """
    mapping = {}
    basename_count = {}
    for f in file_paths:
        bn = os.path.basename(f)
        # print(bn)
        basename_count[bn] = basename_count.get(bn, 0) + 1
    seen = {}
    for f in file_paths:
        bn = os.path.basename(f)
        if basename_count[bn] == 1:
            mapping[f] = bn
        else:
            if bn not in seen:
                mapping[f] = bn
                seen[bn] = 1
            else:
                name, ext = os.path.splitext(bn)
                new_name = f"{name}_{seen[bn]}{ext}"
                mapping[f] = new_name
                seen[bn] += 1
    return mapping



# 仅针对 .java 文件构建唯一名称映射字典
java_files = [f for f in contribution.keys() if f.endswith('.java')]
mapping = build_unique_name_mapping(java_files)
# print(mapping)

files = []  # 保存原始文件路径（用于 contribution 统计等后续计算）
file_names = []  # 保存映射后的唯一文件名，用于构建图节点
fa = []
af = []
weight = []
label_dict = {}


matched_test_files = set()


for file in contribution.keys():
    if file.endswith('.java'):
        # 使用 mapping 得到唯一文件名；若不存在则直接使用 basename
        new_name = mapping.get(file, os.path.basename(file))
        files.append(file)
        file_names.append(new_name)
        
        current_file_index = len(file_names) - 1
        
        # 获取当前文件的basename
        file_basename = os.path.basename(file)
        
        # 构建Git仓库中的完整文件路径
        git_file_path = os.path.join(GIT_PATH, file)
        
        # 检查是否在readable文件夹中（使用文件名匹配）
        if file_basename in readable_java_files:
            test_file_path = readable_java_files[file_basename]
            # 检查这个Test文件是否已经被匹配过
            if test_file_path not in matched_test_files:
                if os.path.exists(git_file_path) and os.path.exists(test_file_path):
                    if compare_file_content(git_file_path, test_file_path):
                        label_dict[current_file_index] = 0  # 0表示readable
                        matched_test_files.add(test_file_path)  # 标记这个Test文件已匹配
                        print(f"文件 {file} 与 {test_file_path} 内容相同，添加到label_dict: {current_file_index} -> 0 (Readable)")
        
        # 检查是否在unreadable文件夹中（使用文件名匹配）
        elif file_basename in unreadable_java_files:
            test_file_path = unreadable_java_files[file_basename]
            # 检查这个Test文件是否已经被匹配过
            if test_file_path not in matched_test_files:
                if os.path.exists(git_file_path) and os.path.exists(test_file_path):
                    if compare_file_content(git_file_path, test_file_path):
                        label_dict[current_file_index] = 1  # 1表示unreadable
                        matched_test_files.add(test_file_path)  # 标记这个Test文件已匹配
                        print(f"文件 {file} 与 {test_file_path} 内容相同，添加到label_dict: {current_file_index} -> 1 (Unreadable)")
        
        for author in contribution[file].keys():
            fa.append((current_file_index, authors[author]))
            af.append((authors[author], current_file_index))
            weight.append(contribution[file][author])

print(f"找到 {len(label_dict)} 个匹配的文件添加到label_dict")
print(f"label_dict: {label_dict}")

# 打印一些调试信息，显示匹配的文件
if label_dict:
    print("\n匹配的文件详情:")
    for idx, label in label_dict.items():
        file_path = files[idx]
        file_basename = os.path.basename(file_path)
        label_name = "Readable" if label == 0 else "Unreadable"
        print(f"  - {file_path} -> {label_name}")



# print(fa)
# print(af)



# 构建异质图 - 包含所有需要的边类型
print("构建异质图...")
g = dgl.heterograph({
    ('file', 'fa', 'author'): fa, 
    ('author', 'af', 'file'): af,
    ('file', 'file_self', 'file'): ([], []),  # 先创建空的边类型
    ('author', 'author_self', 'author'): ([], [])  # 先创建空的边类型
})

# 设置边的权重
g.edges["fa"].data["weight"] = torch.tensor(weight)
g.edges["af"].data["weight"] = torch.tensor(weight)

# 现在可以添加自环边了
try:
    # 为file节点添加自环边
    file_nodes = g.num_nodes('file')
    if file_nodes > 0:
        file_self_edges = (torch.arange(file_nodes), torch.arange(file_nodes))
        g = dgl.add_edges(g, file_self_edges[0], file_self_edges[1], etype=('file', 'file_self', 'file'))
        g.edges[('file', 'file_self', 'file')].data['weight'] = torch.ones(file_nodes)
        print(f"成功添加 {file_nodes} 个 file 节点自环边")
    else:
        print("没有 file 节点可以添加自环边")
except Exception as e:
    print(f"添加 file 节点自环边失败: {e}")

try:
    # 为author节点添加自环边
    author_nodes = g.num_nodes('author')
    if author_nodes > 0:
        author_self_edges = (torch.arange(author_nodes), torch.arange(author_nodes))
        g = dgl.add_edges(g, author_self_edges[0], author_self_edges[1], etype=('author', 'author_self', 'author'))
        g.edges[('author', 'author_self', 'author')].data['weight'] = torch.ones(author_nodes)
        print(f"成功添加 {author_nodes} 个 author 节点自环边")
    else:
        print("没有 author 节点可以添加自环边")
except Exception as e:
    print(f"添加 author 节点自环边失败: {e}")

print("添加自环后的边类型:", g.canonical_etypes)
print("各边类型的边数量:")
for etype in g.canonical_etypes:
    print(f"  {etype}: {g.number_of_edges(etype)}")

# 在构建完异质图后，打印总的节点数量
total_nodes = sum(g.num_nodes(ntype) for ntype in g.ntypes)
print(f"总的节点数量: {total_nodes}")







# 设置文件节点的标签
num_file_nodes = g.num_nodes('file')
labels = torch.full((num_file_nodes,), -1, dtype=torch.long)  # 用-1初始化表示无标签
print(num_file_nodes)


# 将有标签的节点填入
for idx, label in label_dict.items():
    # print(idx)
    labels[idx] = label

# 将标签添加到图的文件节点数据中
g.nodes['file'].data['label'] = labels




def create_unsupervised_masks(labels, label_dict, train_ratio=5/6):
    """
    创建半监督学习掩码
    - 无标签节点按5:1划分为训练集和验证集
    - 有标签节点作为测试集
    
    参数:
    - labels: 所有节点的标签张量
    - label_dict: 有标签节点的字典 {节点索引: 标签}
    - train_ratio: 无标签节点中训练集的比例 (5/6 表示 5:1 划分)
    """
    num_nodes = len(labels)
    
    # 获取无标签节点的索引 (标签为-1)
    unlabeled_indices = (labels == -1).nonzero(as_tuple=True)[0].tolist()
    
    # 获取有标签节点的索引 (测试集)
    labeled_indices = list(label_dict.keys())
    
    print(f"总节点数: {num_nodes}")
    print(f"无标签节点数: {len(unlabeled_indices)}")
    print(f"有标签节点数: {len(labeled_indices)}")
    
    # 随机打乱无标签节点
    np.random.shuffle(unlabeled_indices)
    
    # 划分无标签节点为训练集和验证集 (5:1)
    n_unlabeled = len(unlabeled_indices)
    n_train = int(n_unlabeled * train_ratio)
    
    train_indices = unlabeled_indices[:n_train]
    val_indices = unlabeled_indices[n_train:]
    
    print(f"训练集节点数 (无标签): {len(train_indices)}")
    print(f"验证集节点数 (无标签): {len(val_indices)}")
    print(f"测试集节点数 (有标签): {len(labeled_indices)}")
    
    # 创建掩码张量
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    # 设置掩码
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[labeled_indices] = True
    
    return train_mask, val_mask, test_mask




# 假设我们已经有了图g和label_dict
labels = g.nodes['file'].data['label']
train_mask, val_mask, test_mask = create_unsupervised_masks(labels, label_dict)

# 将掩码添加到图中
g.nodes['file'].data['train_mask'] = train_mask
g.nodes['file'].data['val_mask'] = val_mask
g.nodes['file'].data['test_mask'] = test_mask

# 验证划分结果
print("\n掩码统计:")
print(f"训练集: {train_mask.sum().item()} 个节点")
print(f"验证集: {val_mask.sum().item()} 个节点")
print(f"测试集: {test_mask.sum().item()} 个节点")

# 检查划分是否互斥且覆盖所有节点
all_masks = train_mask | val_mask | test_mask
print(f"所有节点是否都被分配: {all_masks.all().item()}")
print(f"掩码是否有重叠: {((train_mask & val_mask) | (train_mask & test_mask) | (val_mask & test_mask)).any().item()}")





# 设置正负样本对计算的阈值和最大样本数
alpha = 0.1  # 您可以根据需要调整这个阈值
K = 50  # 每个节点最多保留的正负样本数量

def create_masked_positive_negative_samples_with_similarity(g, contribution, files, authors, alpha=0.5, k=50):
    """
    为每个掩码划分（训练集、验证集、测试集）分别构建正负样本对
    每个节点最多保留k个正样本和k个负样本，并存储相似度alpha1
    """
    
    # 获取掩码
    train_mask = g.nodes['file'].data['train_mask']
    val_mask = g.nodes['file'].data['val_mask']
    test_mask = g.nodes['file'].data['test_mask']
    
    # 获取每个掩码的节点索引
    train_indices = train_mask.nonzero(as_tuple=True)[0].tolist()
    val_indices = val_mask.nonzero(as_tuple=True)[0].tolist()
    test_indices = test_mask.nonzero(as_tuple=True)[0].tolist()
    
    print(f"训练集节点数: {len(train_indices)}")
    print(f"验证集节点数: {len(val_indices)}")
    print(f"测试集节点数: {len(test_indices)}")
    
    # 为每个掩码划分分别构建正负样本对
    train_positive, train_negative = _create_samples_with_similarity(
        train_indices, contribution, files, authors, alpha, k, "训练集")
    val_positive, val_negative = _create_samples_with_similarity(
        val_indices, contribution, files, authors, alpha, k, "验证集")
    test_positive, test_negative = _create_samples_with_similarity(
        test_indices, contribution, files, authors, alpha, k, "测试集")
    
    # 合并所有掩码的正负样本对
    all_positive_samples = {}
    all_negative_samples = {}
    
    for idx, samples in train_positive.items():
        all_positive_samples[idx] = samples
    for idx, samples in train_negative.items():
        all_negative_samples[idx] = samples
        
    for idx, samples in val_positive.items():
        all_positive_samples[idx] = samples
    for idx, samples in val_negative.items():
        all_negative_samples[idx] = samples
        
    for idx, samples in test_positive.items():
        all_positive_samples[idx] = samples
    for idx, samples in test_negative.items():
        all_negative_samples[idx] = samples
    
    return all_positive_samples, all_negative_samples

def _create_samples_with_similarity(indices, contribution, files, authors, alpha, k, mask_name):
    """
    为特定掩码划分内的节点构建正负样本对，存储相似度alpha1
    每个节点最多保留k个正样本和k个负样本
    """
    print(f"\n开始为{mask_name}构建正负样本对(含相似度)...")
    
    positive_samples = {}  # 格式: {node_idx: [(target_idx, similarity), ...]}
    negative_samples = {}  # 格式: {node_idx: [(target_idx, similarity), ...]}
    
    # 只在当前掩码的节点之间构建样本对
    for i, idx1 in enumerate(tqdm(indices, desc=f"{mask_name}样本对")):
        orig_f1 = files[idx1]
        pos_samples = []  # 存储 (target_idx, similarity)
        neg_samples = []  # 存储 (target_idx, similarity)
        
        for idx2 in indices:
            if idx1 == idx2:
                continue
                
            orig_f2 = files[idx2]
            
            # 获取两个文件的贡献信息
            contrib1 = contribution.get(orig_f1, {})
            contrib2 = contribution.get(orig_f2, {})
            
            sum1 = sum(contrib1.values())
            sum2 = sum(contrib2.values())
            
            # 如果任一文件没有贡献信息，跳过
            if sum1 == 0 or sum2 == 0:
                continue
                
            set1 = set(contrib1.keys())
            set2 = set(contrib2.keys())
            s = set1.intersection(set2)
            
            o1, o2 = 0, 0
            for author in s:
                if contrib1[author] < contrib2[author]:
                    o1 += contrib1[author]
                else:
                    o2 += contrib2[author]
            rate1 = o1 / sum1
            rate2 = o2 / sum2
            if rate1 > rate2:
                alpha1 = rate2
            else:
                alpha1 = rate1
            
            if alpha1 > alpha:
                pos_samples.append((idx2, alpha1))
            else:
                neg_samples.append((idx2, alpha1))
        
        # 按相似度排序并采样
        # 正样本：按相似度降序排列（相似度越高越好）
        if len(pos_samples) > k:
            pos_samples = sorted(pos_samples, key=lambda x: x[1], reverse=True)[:k]
        
        # 负样本：按相似度升序排列（相似度越低越典型）
        if len(neg_samples) > k:
            neg_samples = sorted(neg_samples, key=lambda x: x[1])[:k]
        
        positive_samples[idx1] = pos_samples
        negative_samples[idx1] = neg_samples
    
    # 统计信息
    total_pos = sum(len(samples) for samples in positive_samples.values())
    total_neg = sum(len(samples) for samples in negative_samples.values())
    avg_pos = total_pos / len(positive_samples) if positive_samples else 0
    avg_neg = total_neg / len(negative_samples) if negative_samples else 0
    
    # 计算平均相似度
    avg_pos_sim = np.mean([sim for samples in positive_samples.values() for _, sim in samples]) if total_pos > 0 else 0
    avg_neg_sim = np.mean([sim for samples in negative_samples.values() for _, sim in samples]) if total_neg > 0 else 0
    
    print(f"{mask_name} - 总正样本对: {total_pos}, 平均每节点: {avg_pos:.2f}, 平均相似度: {avg_pos_sim:.4f}")
    print(f"{mask_name} - 总负样本对: {total_neg}, 平均每节点: {avg_neg:.2f}, 平均相似度: {avg_neg_sim:.4f}")
    
    return positive_samples, negative_samples

def store_samples_with_similarity_as_tensors(g, positive_samples, negative_samples, k):
    """
    将带有相似度的正负样本对存储为张量
    """
    print(f"\n开始存储带有相似度的正负样本对(最多{k}个)...")
    
    num_nodes = g.num_nodes('file')
    
    # 创建固定大小的张量，用-1填充节点索引，用0填充相似度
    positive_indices = torch.full((num_nodes, k), -1, dtype=torch.long)
    positive_similarities = torch.zeros((num_nodes, k), dtype=torch.float)
    negative_indices = torch.full((num_nodes, k), -1, dtype=torch.long)
    negative_similarities = torch.zeros((num_nodes, k), dtype=torch.float)
    
    # 存储每个节点的实际样本数量
    positive_lengths = torch.zeros(num_nodes, dtype=torch.long)
    negative_lengths = torch.zeros(num_nodes, dtype=torch.long)
    
    # 填充张量
    for node_idx in tqdm(range(num_nodes), desc="存储样本对"):
        if node_idx in positive_samples:
            pos_samples = positive_samples[node_idx]  # [(target_idx, similarity), ...]
            neg_samples = negative_samples[node_idx]  # [(target_idx, similarity), ...]
            
            # 存储正样本
            if len(pos_samples) > 0:
                for i, (target_idx, similarity) in enumerate(pos_samples):
                    positive_indices[node_idx, i] = target_idx
                    positive_similarities[node_idx, i] = similarity
                positive_lengths[node_idx] = len(pos_samples)
            
            # 存储负样本
            if len(neg_samples) > 0:
                for i, (target_idx, similarity) in enumerate(neg_samples):
                    negative_indices[node_idx, i] = target_idx
                    negative_similarities[node_idx, i] = similarity
                negative_lengths[node_idx] = len(neg_samples)
    
    # 添加到图数据中
    g.nodes['file'].data['positive_indices'] = positive_indices
    g.nodes['file'].data['positive_similarities'] = positive_similarities
    g.nodes['file'].data['negative_indices'] = negative_indices
    g.nodes['file'].data['negative_similarities'] = negative_similarities
    g.nodes['file'].data['positive_lengths'] = positive_lengths
    g.nodes['file'].data['negative_lengths'] = negative_lengths
    
    return positive_indices, positive_similarities, negative_indices, negative_similarities



# 执行正负样本对构建（带相似度）
print("开始构建按掩码划分的带相似度正负样本对...")
all_positive_samples, all_negative_samples = create_masked_positive_negative_samples_with_similarity(
    g, contribution, files, authors, alpha, K
)

# 存储为张量
positive_indices, positive_similarities, negative_indices, negative_similarities = store_samples_with_similarity_as_tensors(
    g, all_positive_samples, all_negative_samples, K
)

# 统计信息
print("\n=== 正负样本对统计 (带相似度) ===")
total_pos_pairs = sum(len(samples) for samples in all_positive_samples.values())
total_neg_pairs = sum(len(samples) for samples in all_negative_samples.values())
avg_pos_per_node = total_pos_pairs / len(all_positive_samples) if all_positive_samples else 0
avg_neg_per_node = total_neg_pairs / len(all_negative_samples) if all_negative_samples else 0

# 计算总体平均相似度
all_pos_similarities = [sim for samples in all_positive_samples.values() for _, sim in samples]
all_neg_similarities = [sim for samples in all_negative_samples.values() for _, sim in samples]
avg_pos_sim = np.mean(all_pos_similarities) if all_pos_similarities else 0
avg_neg_sim = np.mean(all_neg_similarities) if all_neg_similarities else 0

print(f"总正样本对数量: {total_pos_pairs}")
print(f"总负样本对数量: {total_neg_pairs}")
print(f"平均每节点正样本对: {avg_pos_per_node:.2f}")
print(f"平均每节点负样本对: {avg_neg_per_node:.2f}")
print(f"正样本对平均相似度: {avg_pos_sim:.4f}")
print(f"负样本对平均相似度: {avg_neg_sim:.4f}")

# 按掩码统计
train_mask = g.nodes['file'].data['train_mask']
val_mask = g.nodes['file'].data['val_mask']
test_mask = g.nodes['file'].data['test_mask']

train_pos = sum(len(all_positive_samples[i]) for i in range(len(files)) if train_mask[i])
train_neg = sum(len(all_negative_samples[i]) for i in range(len(files)) if train_mask[i])
val_pos = sum(len(all_positive_samples[i]) for i in range(len(files)) if val_mask[i])
val_neg = sum(len(all_negative_samples[i]) for i in range(len(files)) if val_mask[i])
test_pos = sum(len(all_positive_samples[i]) for i in range(len(files)) if test_mask[i])
test_neg = sum(len(all_negative_samples[i]) for i in range(len(files)) if test_mask[i])

train_nodes = train_mask.sum().item()
val_nodes = val_mask.sum().item()
test_nodes = test_mask.sum().item()

print(f"\n按掩码划分统计:")
print(f"训练集 - {train_nodes}节点: 正样本对 {train_pos}, 负样本对 {train_neg}")
print(f"验证集 - {val_nodes}节点: 正样本对 {val_pos}, 负样本对 {val_neg}")
print(f"测试集 - {test_nodes}节点: 正样本对 {test_pos}, 负样本对 {test_neg}")

# 打印一些示例，展示不同节点的样本对数量和相似度
print(f"\n示例节点样本对详情 (最多{K}个):")
for i in range(min(5, len(files))):
    pos_samples = all_positive_samples.get(i, [])
    neg_samples = all_negative_samples.get(i, [])
    mask_type = "训练集" if train_mask[i] else "验证集" if val_mask[i] else "测试集"
    
    print(f"节点 {i}: {mask_type}")
    if pos_samples:
        avg_pos_sim_node = np.mean([sim for _, sim in pos_samples])
        print(f"  正样本对: {len(pos_samples)}个, 平均相似度: {avg_pos_sim_node:.4f}")
        # 显示前3个正样本对的相似度
        for j, (target_idx, sim) in enumerate(pos_samples[:3]):
            print(f"    - 与节点{target_idx}: 相似度={sim:.4f}")
    if neg_samples:
        avg_neg_sim_node = np.mean([sim for _, sim in neg_samples])
        print(f"  负样本对: {len(neg_samples)}个, 平均相似度: {avg_neg_sim_node:.4f}")
        # 显示前3个负样本对的相似度
        for j, (target_idx, sim) in enumerate(neg_samples[:3]):
            print(f"    - 与节点{target_idx}: 相似度={sim:.4f}")

print("\n带相似度的正负样本对构建完成！")


# 引入 DMon 和相关配置
from Preprocess.Code.DMon import DMon
from Preprocess.Code.configures import data_args, train_args, model_args
from code_dataset import ConvertToGraph, graph_to_input  # 引入处理函数
from transformers import AutoTokenizer, AutoModel
from Preprocess.Code.load_dataset import load_ex, extra


def load_model(model_path):
    try:
        model = DMon(data_args, model_args)

        # 加载模型的所有权重
        checkpoint = torch.load(model_path, map_location=model_args.device)

        # 获取当前模型的权重字典
        model_state_dict = model.state_dict()

        # 从保存的模型中获取权重
        checkpoint_state_dict = checkpoint['net']

        # 过滤掉不需要的层（比如 lin1、lin2 和 fc_up）
        checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items()
                                 if 'lin1' not in k and 'lin2' not in k and 'fc_up' not in k}

        # 更新模型的状态字典，只加载需要的部分
        model_state_dict.update(checkpoint_state_dict)

        # 将更新后的字典加载到模型中
        model.load_state_dict(model_state_dict)

        model.to(model_args.device)
        print(f"模型加载成功: {model_path}")
        return model.eval()
    except Exception as e:
        print(f"加载模型 {model_path} 失败:", str(e))
        return None



# 初始化 tokenizer 与 embedding 模型
# MODEL_PATH = "Preprocess/Code/checkpoint/Models/dmon_best_3.pth"
# WYH
MODEL_PATH = "Preprocess/Code/checkpoint/code_hh/dmon_best_3.pth"
model_path = r"Preprocess/Code/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
embedding = AutoModel.from_pretrained(model_path)

# 加载 DMon 模型
dmon = load_model(MODEL_PATH)



JSONPATH = "json/mlkit/out"
features = []
json_files = []
# 处理 JSON 文件时同样依据 skip_files 过滤（假设 JSON 文件名与唯一 Java 文件名对应，
# 即将 .json 替换成 .java 后为映射后的名称）
for file_name in os.listdir(JSONPATH):
    if file_name.endswith('.json'):
        json_files.append(file_name)
print("JSON 文件夹中的文件:", json_files)
print("数量:", len(json_files))


graph_list = []
target_list = []
code_filename_list = []
# 遍历 JSON 文件（假设 JSON 文件名与映射后的 Java 文件名对应）
# tqdm(os.listdir(JSONPATH))
# print(file_names)
for file_name in file_names:
    if file_name.endswith('.java'):
        json_file = file_name.replace('.java', '.json')
        json_file_path = os.path.join(JSONPATH, json_file)
        with open(json_file_path, encoding="utf-8") as f:
            graph = ConvertToGraph(json.load(f))
            graph_list.append(graph)
            target_list.append(-1)
            java_file_path = os.path.join(JSONPATH, file_name)
            code_filename_list.append(java_file_path)
    else:
        pass

print(f"共有 {len(graph_list)} 个图待处理。")
# print(graph_list)




# 进行特征提取并统一节点维度
all_node_embeddings = []
all_edge_indices = []
batch_list = []

graph_input = []
target_input = []

print(len(graph_list))

empty_graph_count = 0
empty_graph_files = []

# print(code_filename_list)
for i in range(len(graph_list)):
    node_type, raw_code_list, node_embedding_list, edge_list, edge_types, target, node_one_hot_list = graph_to_input(
        graph_list[i], code_filename_list[i], target_list[i], tokenizer, embedding)

    # 如果节点嵌入为空，则填充默认值（例如全零向量），并记录该图
    if len(node_embedding_list) == 0:
        print(f"Preprocess {code_filename_list[i]} has no node embeddings, filling with zeros.")
        empty_graph_count += 1
        empty_graph_files.append(code_filename_list[i])
        # 假设节点特征维度为 (840,)（根据后面拼接操作），这里可以根据实际情况修改
        default_node_embedding = np.zeros(840)
        node_embeddings = [default_node_embedding]
        # 如果需要构造 edge_index，也可以设为一个空图，或者其他默认值
        edge_list = np.empty((2, 0), dtype=int)  # 无边
    else:
        node_embeddings = []
        for j in range(len(node_embedding_list)):
            node_embedding = np.array(node_embedding_list[j])
            node_embedding = np.mean(node_embedding_list[j], axis=0)  # 对每个节点的嵌入进行平均
            node_info = np.concatenate((node_embedding.tolist(), node_one_hot_list[j]), axis=0)
            node_embeddings.append(node_info)

    x = torch.tensor(node_embeddings)
    x = x.to(torch.float32)
    # 填充每个图的节点特征，以保证所有图的节点数一致
    max_nodes = max(1000, x.size(0))  # 保留最小保障1000，但根据实际情况扩展
    x_zero = torch.zeros(max_nodes, 840).float()
    x_zero[:x.size(0), :] = x

    edge_index = torch.tensor(edge_list)
    # 在此处添加print语句来检查edge_index的形状
    print("Edge index shape:", edge_index.shape)

    y = torch.tensor([target]).float()
    graph_data = Data(x=x, edge_index=edge_index, y=target)

    target_input.append(target)
    # node_type #edge_type
    graph_input.append(graph_data)

# 输出空图统计信息
print(f"总共有 {empty_graph_count} 个空图。")
print("空图对应的文件列表：", empty_graph_files)

dataloader = extra(data_args, graph_input)
# 特征提取
with torch.no_grad():
    for data in dataloader['extra']:
        data = data.to(model_args.device)
        feature, loss = dmon(data.x, data.edge_index, data.batch, return_features=True)  # 用DMon提取特征
        features.append(feature)
        print("Feature shape:", feature.shape)

features_tensor = torch.cat(features, dim=0)




def validate_order_consistency_improved(g, file_names, code_filename_list, features_tensor):
    """
    改进的顺序一致性验证
    """
    print("\n=== 顺序一致性验证（改进版） ===")
    
    # 基本数量检查
    num_file_nodes = g.num_nodes('file')
    num_features = features_tensor.shape[0] if features_tensor is not None else 0
    num_code_files = len(code_filename_list)
    
    print(f"图中文件节点数量: {num_file_nodes}")
    print(f"特征数量: {num_features}")
    print(f"代码文件列表数量: {num_code_files}")
    
    if num_file_nodes != num_code_files or num_features != num_code_files:
        print(f"❌ 数量不匹配")
        return False
    
    # 检查文件名称对应关系
    print("\n检查文件名称对应关系:")
    mismatch_count = 0
    exact_match_count = 0
    mapping_consistent_count = 0
    
    # 分析映射模式
    mapping_patterns = {}
    
    for i in range(min(len(file_names), len(code_filename_list))):
        graph_filename = file_names[i]
        feature_filename = os.path.basename(code_filename_list[i])
        
        # 1. 检查精确匹配
        if graph_filename == feature_filename:
            exact_match_count += 1
            print(f"✅ 索引 {i}: 精确匹配 '{graph_filename}'")
        else:
            mismatch_count += 1
            print(f"❌ 索引 {i}: 图='{graph_filename}', 特征='{feature_filename}'")
            
            # 2. 分析映射模式是否一致
            # 提取基础文件名（去掉后缀数字）
            graph_base = re.sub(r'_?\d*\.java$', '.java', graph_filename)
            feature_base = re.sub(r'_?\d*\.java$', '.java', feature_filename)
            
            if graph_base == feature_base:
                mapping_consistent_count += 1
                # 记录映射模式
                pattern = f"{graph_base} -> {graph_filename} vs {feature_filename}"
                if pattern not in mapping_patterns:
                    mapping_patterns[pattern] = 0
                mapping_patterns[pattern] += 1
    
    # 输出统计结果
    print(f"\n=== 验证结果统计 ===")
    print(f"总检查文件数: {min(len(file_names), len(code_filename_list))}")
    print(f"精确匹配: {exact_match_count}")
    print(f"名称不匹配: {mismatch_count}")
    print(f"映射模式一致: {mapping_consistent_count}")
    
    # 分析映射模式
    if mapping_patterns:
        print(f"\n映射模式分析:")
        for pattern, count in mapping_patterns.items():
            print(f"  {pattern}: {count}次")
    
    # 判断标准：
    # 1. 如果所有文件都精确匹配，完美
    # 2. 如果映射模式一致，也可以接受（只是命名细节不同）
    # 3. 如果映射模式不一致，有问题
    
    if mismatch_count == 0:
        print("✅ 所有文件名精确匹配，顺序完全一致")
        return True
    elif mapping_consistent_count == mismatch_count:
        print("✅ 文件名映射模式一致，顺序一致（只是命名细节不同）")
        return True
    else:
        print("❌ 文件名映射模式不一致，顺序可能有问题")
        return False

# 运行改进的验证
is_consistent = validate_order_consistency_improved(g, file_names, code_filename_list, features_tensor)





# 直接将特征嵌入到图节点中
print("开始嵌入特征到图节点...")

# 将图转移到与特征张量相同的设备
g = g.to(features_tensor.device)

# 现在可以安全地分配特征了
g.nodes['file'].data['features'] = features_tensor

# 验证嵌入结果
print("特征嵌入完成!")
print(f"嵌入特征形状: {g.nodes['file'].data['features'].shape}")
print(f"图当前设备: {g.device}")
print(f"特征设备: {features_tensor.device}")


# 检查特征质量
feature_norms = torch.norm(g.nodes['file'].data['features'], dim=1)
zero_features = (feature_norms == 0).sum().item()
print(f"非零特征节点: {100 - zero_features}/100")



import torch
import numpy as np

# 填充开发者节点属性
print("开始填充开发者节点属性...")

# 获取图的设备信息
device = g.device
print(f"图当前设备: {device}")

# 方法1: 基于开发者的贡献统计信息
def create_author_features(git_repo, authors, g):
    """
    为开发者节点创建特征
    基于开发者的贡献统计信息
    """
    # 统计每个作者的贡献信息
    author_total_commits = {}
    author_total_files = {}
    author_total_lines = {}
    
    # 初始化
    for author in authors.keys():
        author_total_commits[author] = 0
        author_total_files[author] = 0
        author_total_lines[author] = 0
    
    # 统计每个作者的贡献
    for file_path, contributions in git_repo.file_contribution.items():
        if file_path.endswith('.java'):
            for author, lines in contributions.items():
                if author in author_total_commits:
                    author_total_files[author] += 1
                    author_total_lines[author] += lines
    
    # 统计每个作者的提交次数
    for author, commits in git_repo.author_commit.items():
        if author in author_total_commits:
            author_total_commits[author] = len(commits)
    
    # 创建特征向量
    author_features = []
    for author_id in range(g.num_nodes('author')):
        # 找到对应的Author对象
        author_obj = None
        for author, aid in authors.items():
            if aid == author_id:
                author_obj = author
                break
        
        if author_obj and author_obj in author_total_commits:
            # 创建特征向量
            commits = author_total_commits[author_obj]
            files = author_total_files[author_obj]
            lines = author_total_lines[author_obj]
            
            # 归一化特征（避免数值过大）
            max_commits = max(author_total_commits.values()) if author_total_commits else 1
            max_files = max(author_total_files.values()) if author_total_files else 1
            max_lines = max(author_total_lines.values()) if author_total_lines else 1
            
            feature = [
                commits / max_commits,  # 归一化的提交次数
                files / max_files,      # 归一化的文件数量
                lines / max_lines,      # 归一化的代码行数
                np.log1p(commits),      # 提交次数的对数
                np.log1p(files),        # 文件数量的对数
                np.log1p(lines)         # 代码行数的对数
            ]
        else:
            # 如果没有找到作者信息，使用默认特征
            feature = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        author_features.append(feature)
    
    return torch.tensor(author_features, dtype=torch.float, device=device)

# 方法2: 基于作者名的简单特征
def create_simple_author_features(authors, g):
    """
    基于作者名创建简单特征
    """
    author_features = []
    
    for author_id in range(g.num_nodes('author')):
        # 找到对应的Author对象
        author_obj = None
        for author, aid in authors.items():
            if aid == author_id:
                author_obj = author
                break
        
        if author_obj:
            # 使用作者名和邮箱的哈希值作为特征
            author_str = str(author_obj)
            # 创建多个哈希特征
            hash1 = int(hashlib.md5(author_str.encode()).hexdigest()[:8], 16) % 1000 / 1000
            hash2 = int(hashlib.md5(author_str.encode()).hexdigest()[8:16], 16) % 1000 / 1000
            hash3 = int(hashlib.md5(author_str.encode()).hexdigest()[16:24], 16) % 1000 / 1000
            
            # 作者名字符长度特征
            name_len = len(author_obj.name) / 50  # 假设最大长度为50
            
            # 邮箱特征（是否包含常见域名）
            email = author_obj.email.lower()
            common_domains = ['gmail.com', 'hotmail.com', 'yahoo.com', 'outlook.com']
            domain_feature = 0.0
            for domain in common_domains:
                if domain in email:
                    domain_feature = 1.0
                    break
            
            feature = [hash1, hash2, hash3, name_len, domain_feature]
        else:
            feature = [0.0, 0.0, 0.0, 0.0, 0.0]
        
        author_features.append(feature)
    
    return torch.tensor(author_features, dtype=torch.float, device=device)

# 选择方法一和方法二组合特征
print("创建开发者节点特征...")

# 使用基于贡献统计的特征
author_features1 = create_author_features(git_repo, authors, g)
print(f"基于贡献统计的特征形状: {author_features1.shape}")

# 使用基于作者名的简单特征
author_features2 = create_simple_author_features(authors, g)
print(f"基于作者名的特征形状: {author_features2.shape}")

# 组合所有特征
author_features_combined = torch.cat([author_features1, author_features2], dim=1)
print(f"组合特征形状: {author_features_combined.shape}")

# 将特征添加到图中
g.nodes['author'].data['features'] = author_features_combined

# 添加其他有用的作者属性
print("添加其他作者属性...")

# 添加作者标识符
author_identifiers = []
for author_id in range(g.num_nodes('author')):
    author_obj = None
    for author, aid in authors.items():
        if aid == author_id:
            author_obj = author
            break
    author_identifiers.append(str(author_obj) if author_obj else f"unknown_{author_id}")

# 注意：DGL不能直接存储字符串列表，我们可以存储为数字编码
# 或者我们可以将标识符保存为图的属性
g.author_identifiers = author_identifiers

# 添加作者的提交次数
author_commit_counts = []
for author_id in range(g.num_nodes('author')):
    author_obj = None
    for author, aid in authors.items():
        if aid == author_id:
            author_obj = author
            break
    
    if author_obj and author_obj in git_repo.author_commit:
        commit_count = len(git_repo.author_commit[author_obj])
    else:
        commit_count = 0
    
    author_commit_counts.append(commit_count)

g.nodes['author'].data['commit_count'] = torch.tensor(author_commit_counts, dtype=torch.float, device=device)

# 验证开发者节点属性
print("\n开发者节点属性填充完成!")
print(f"作者节点特征形状: {g.nodes['author'].data['features'].shape}")
print(f"作者节点提交次数形状: {g.nodes['author'].data['commit_count'].shape}")

# 检查特征质量
if 'features' in g.nodes['author'].data:
    author_feature_norms = torch.norm(g.nodes['author'].data['features'], dim=1)
    zero_author_features = (author_feature_norms == 0).sum().item()
    print(f"作者节点非零特征比例: {(g.num_nodes('author') - zero_author_features) / g.num_nodes('author') * 100:.2f}%")

# 打印前几个作者的信息
print("\n前5个作者节点信息:")
for i in range(min(5, g.num_nodes('author'))):
    print(f"作者 {i}: {g.author_identifiers[i] if i < len(g.author_identifiers) else 'unknown'}")
    print(f"  提交次数: {g.nodes['author'].data['commit_count'][i].item()}")
    if 'features' in g.nodes['author'].data:
        print(f"  特征范数: {torch.norm(g.nodes['author'].data['features'][i]).item():.4f}")
    print()

# 保存更新后的图
print("保存包含开发者节点属性的图...")
dgl.save_graphs("code_readability_graph_with_authors1.bin", [g])
print("图已保存到 code_readability_graph_with_authors1.bin")













