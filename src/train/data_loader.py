import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import ast
import yaml
import os
from os.path import dirname, join, abspath
import threading
import numpy as np
import math

class CSVDataset(Dataset):
    def __init__(self, chunk):
        self.df = chunk
        self.len = len(self.df)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        feature1 = torch.tensor(ast.literal_eval(self.df['Input1'][idx]), dtype=torch.float32)
        feature2 = torch.tensor(ast.literal_eval(self.df['Input2'][idx]), dtype=torch.float32)
        output = torch.tensor(ast.literal_eval(self.df['Output'][idx]), dtype=torch.float32)
        if feature2[0][0] == 1:
            if len(feature1[0]) == 25:
                feature2[:, 4] = feature2[:, 4] - 1.2465
                output[0:12] = output[0:12] / math.pi
                output[12:24] = output[12:24] / 2
                output[24:] = output[24:] / 12
            elif len(feature1[0]) == 13:
                output[0:6] = output[0:6] / math.pi
                output[6:12] = output[6:12] / 2
                output[12:] = output[12:] / 50
            elif len(feature1[0]) == 6:
                output[0:5] = output[0:5] / math.pi
                output[5:10] = output[5:10] / 2
                output[10:] = output[10:] / math.pi

        elif feature2[0][0] == 0:
            if len(feature1[0]) == 25:
                if feature2[0][2] == 1:
                    feature2[:, 3:15] = feature2[:, 2:14] / 50
                    output[:, 3] = feature2[:, 3] - 1.2465
                elif feature2[0][2] == 2:
                    feature2[:, 3:15] = feature2[:, 2:14] / 250
                    output[:, 3] = feature2[:, 3] - 1.2465
            elif len(feature1[0]) == 13:
                if feature2[0][2] == 1:
                    feature2[:, 3:9] = feature2[:, 2:8] / 50
                    output[:, 3] = feature2[:, 3] - 1.2465
                elif feature2[0][2] == 2:
                    feature2[:, 3:9] = feature2[:, 2:8] / 250
                    output[:, 3] = feature2[:, 3] - 1.2465

        return  feature1, feature2, output