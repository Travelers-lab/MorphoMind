import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast
import numpy as np
from sklearn.model_selection import train_test_split
import os


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        feature1 = torch.tensor(ast.literal_eval(self.df['Input1'][idx]), dtype=torch.float32)
        feature2 = torch.tensor(ast.literal_eval(self.df['Input2'][idx]), dtype=torch.float32)
        output = torch.tensor(ast.literal_eval(self.df['Output'][idx]), dtype=torch.float32)
        return feature1, feature2, output


def split_large_csv(csv_file_path, output_dir="./split_data"):
    """
    分割大型CSV文件为训练集、验证集和测试集
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 由于文件很大，我们使用分块读取的方式
    chunk_size = 10000  # 根据内存调整这个值
    chunks = []

    print("开始读取CSV文件...")
    for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
        chunks.append(chunk)

    # 合并所有块
    df = pd.concat(chunks, ignore_index=True)
    print(f"数据集总大小: {len(df)} 行")

    # 首先分割出测试集和验证集（各1%）
    train_val_df, test_df = train_test_split(
        df, test_size=0.01, random_state=42, shuffle=True
    )

    # 从剩余数据中分割出验证集（占原始数据的1%）
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.01 / 0.99, random_state=42, shuffle=True
    )

    print(f"训练集大小: {len(train_df)} 行 ({len(train_df) / len(df) * 100:.2f}%)")
    print(f"验证集大小: {len(val_df)} 行 ({len(val_df) / len(df) * 100:.2f}%)")
    print(f"测试集大小: {len(test_df)} 行 ({len(test_df) / len(df) * 100:.2f}%)")

    # 保存分割后的数据集
    train_df.to_csv(f"{output_dir}/train.csv", index=False)
    val_df.to_csv(f"{output_dir}/val.csv", index=False)
    test_df.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"分割完成！文件保存在: {output_dir}")

    return train_df, val_df, test_df


def create_data_loaders(batch_size=32):
    """
    创建数据加载器
    """
    # 加载分割后的数据
    train_df = pd.read_csv("./split_data/train.csv")
    val_df = pd.read_csv("./split_data/val.csv")
    test_df = pd.read_csv("./split_data/test.csv")

    # 创建数据集实例
    train_dataset = CustomDataset(train_df)
    val_dataset = CustomDataset(val_df)
    test_dataset = CustomDataset(test_df)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# 更高效的内存版本（适用于非常大的数据集）
def split_large_csv_memory_efficient(csv_file_path, output_dir="./split_data"):
    """
    内存高效版本 - 逐行处理大型CSV文件
    """
    os.makedirs(output_dir, exist_ok=True)

    # 首先统计总行数
    total_rows = 0
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        total_rows = sum(1 for _ in f) - 1  # 减去标题行

    print(f"数据集总行数: {total_rows}")

    # 计算各集合的大小
    test_size = int(total_rows * 0.01)
    val_size = int(total_rows * 0.01)
    train_size = total_rows - test_size - val_size

    print(f"训练集: {train_size} 行, 验证集: {val_size} 行, 测试集: {test_size} 行")

    # 生成随机索引
    indices = np.random.permutation(total_rows)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # 创建集合索引的集合（用于快速查找）
    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)

    # 打开输出文件
    train_file = open(f"{output_dir}/train.csv", 'w', encoding='utf-8')
    val_file = open(f"{output_dir}/val.csv", 'w', encoding='utf-8')
    test_file = open(f"{output_dir}/test.csv", 'w', encoding='utf-8')

    # 写入标题
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        train_file.write(header)
        val_file.write(header)
        test_file.write(header)

    # 逐行处理数据
    print("开始分割数据...")
    with open(csv_file_path, 'r', encoding='utf-8') as f:
        next(f)  # 跳过标题行
        for i, line in enumerate(f):
            if i in train_set:
                train_file.write(line)
            elif i in val_set:
                val_file.write(line)
            elif i in test_set:
                test_file.write(line)

            if i % 100000 == 0:
                print(f"已处理 {i} 行...")

    # 关闭文件
    train_file.close()
    val_file.close()
    test_file.close()

    print("分割完成！")


if __name__ == "__main__":
    # 使用方法
    csv_file_path = '/mnt/data/morpho_mind/datasets/training_datasets_DAI.csv'  # 替换为你的CSV文件路径

    output_dir = "/mnt/data/morpho_mind/datasets/"

    # 方法1：如果内存足够
    # train_df, val_df, test_df = split_large_csv(csv_file_path)

    # 方法2：内存高效版本（推荐用于51GB的大文件）
    split_large_csv_memory_efficient(csv_file_path, output_dir=output_dir)

    # 创建数据加载器
    # train_loader, val_loader, test_loader = create_data_loaders(batch_size=32)

    # 示例：如何使用数据加载器
    # for batch_idx, (feature1, feature2, output) in enumerate(train_loader):
    #     # 在这里进行训练
    #     print(f"Batch {batch_idx}: Feature1 shape: {feature1.shape}, Feature2 shape: {feature2.shape}, Output shape: {output.shape}")
    #     break  # 只显示第一个batch