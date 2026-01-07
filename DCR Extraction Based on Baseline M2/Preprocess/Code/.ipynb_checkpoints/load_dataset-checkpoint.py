from torch.utils.data import Dataset as TorchDataset
from torch_geometric.loader import DataLoader
import torch
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np
from torch.utils.data import random_split


class InputDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.iloc[index].input


class CodeDataset(TorchDataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def balance_dataset(dataset):
    unread = []
    read = []

    for data in dataset:
        if data.y == 0:
            read.append(data)
        if data.y == 1:
            unread.append(data)

    return CodeDataset(read), CodeDataset(unread)




def load_ex(data_args, graph_input):
    read, unread = balance_dataset(graph_input)
    batch_size = data_args.batch_size
    extra = ConcatDataset([read, unread])
    dataloader = dict()
    dataloader['extra'] = DataLoader(extra, batch_size=batch_size, shuffle=False)
    return dataloader

def extra(data_args, graph_input):
    batch_size = data_args.batch_size
    dataloader = dict()
    dataloader['extra'] = DataLoader(graph_input, batch_size=batch_size, shuffle=False)
    return dataloader



def no_label(data_args, graph_input):
    """
    对无标签数据按照 5:1 的比例划分成训练集和验证集

    Args:
        data_args: 包含 batch_size 和 seed 等配置信息的参数对象
        graph_input: 无标签数据集，通常为一个列表或 Dataset 对象

    Returns:
        一个包含 'train' 和 'valid' 两个 DataLoader 的字典
    """
    total_len = len(graph_input)
    # 计算训练集和验证集的长度
    train_len = int(total_len * 5 / 6)
    valid_len = total_len - train_len

    # 使用固定的随机种子确保每次划分一致
    train_dataset, valid_dataset = random_split(
        graph_input, [train_len, valid_len],
        generator=torch.Generator().manual_seed(data_args.seed)
    )

    # 构建 DataLoader
    dataloader = dict()
    dataloader['train'] = DataLoader(train_dataset, batch_size=data_args.batch_size, shuffle=True)
    dataloader['valid'] = DataLoader(valid_dataset, batch_size=data_args.batch_size, shuffle=True)
    
    return dataloader
    






def get_dataloader(data_args, only_test=False):
    """
    Args:
        dataset:
        batch_size: int
        only_test: boolean, if True, only returns the test dataset.

    Returns:
        a dictionary with the test dataLoader
    """

    dataset = pd.read_pickle(data_args.dataset_dir)
    dataset = InputDataset(dataset)
    read, unread = balance_dataset(dataset)

    batch_size = data_args.batch_size

    if only_test:
        # 只加载测试集
        test_r = read
        test_u = unread

        test = ConcatDataset([test_r, test_u])

        dataloader = dict()
        dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=False)  # shuffle=False 保证顺序一致
        return dataloader

    # 如果 not only_test，默认会继续返回训练集、验证集和测试集（保持原功能）
    train_r, valid_r, test_r = random_split(read, lengths=[int(data_args.data_split_ratio[0] * len(read)),
                                                           int(data_args.data_split_ratio[1] * len(read)),
                                                           len(read) - int(data_args.data_split_ratio[0] * len(read)) - int(
                                                               data_args.data_split_ratio[1] * len(read))],
                                            generator=torch.Generator().manual_seed(data_args.seed))

    train_u, valid_u, test_u = random_split(unread, lengths=[int(data_args.data_split_ratio[0] * len(unread)),
                                                             int(data_args.data_split_ratio[1] * len(unread)),
                                                             len(unread) - int(data_args.data_split_ratio[0] * len(unread)) - int(
                                                                 data_args.data_split_ratio[1] * len(unread))],
                                            generator=torch.Generator().manual_seed(data_args.seed))

    train = ConcatDataset([train_r, train_u])
    valid = ConcatDataset([valid_r, valid_u])
    test = ConcatDataset([test_r, test_u])

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['valid'] = DataLoader(valid, batch_size=batch_size, shuffle=True)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=True)

    return dataloader
