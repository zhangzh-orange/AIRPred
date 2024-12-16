#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------
# @File  : dataset.py
# @Time  : 2024/03/13 14:22:36
# @Author: Zhang, Zehong 
# @Email : zhang@fmp-berlin.de
# @Desc  : None
# -----------------------------------

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from torch.utils import data
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', 1000)

# peptide process
seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
# v是氨基酸，i+1是序号
seq_dict_len = len(seq_dict)

# modification process
mod_voc = "MC"
mod_dict = {v: (i + 1) for i, v in enumerate(mod_voc)}
mod_dict['-'] = 0 # 添加占位符，补齐没有modification的位置
mod_dict_len = len(mod_dict)

# n-gram处理
tri_gram_list = []
for x in seq_voc:
    for y in seq_voc:
        for z in seq_voc:
            tri_gram = x + y + z
            tri_gram_list.append(tri_gram)
tri_gram_dict = {v: (i + 1) for i, v in enumerate(tri_gram_list)}

def peptide_mod_to_num(prot, max_seq_len, mod_list=None):
    """Convert peptide into numerical representations, with modification
    mod_list: a list of modification. For example:[M6, C2]"""

    prot = prot.strip()
    prot = prot.replace(" ", "")
    protein_length = len(prot) + 1

    mod_seq = "-" * len(prot)
    if type(mod_list) is not list:
        mod_seq = mod_seq
    else:
        try:
            mod_site_type_dict = {int(mod[1:]): mod[0] for mod in mod_list}
            for site, mod_type in mod_site_type_dict.items():
                mod_seq = mod_seq[:site] + mod_type + mod_seq[site+1:]
        except AttributeError:
            mod_seq = mod_seq

    prot_num_seq = np.zeros(max_seq_len, dtype=int)
    mod_num_seq = np.zeros(max_seq_len, dtype=int)
    if protein_length > max_seq_len:
        # 如果蛋白序列大于列表，取蛋白N端一段等于列表长度的序列
        for i, ch in enumerate(prot[:max_seq_len]):
            prot_num_seq[i] = seq_dict[ch]
        for i, ch in enumerate(mod_seq[:max_seq_len]):
            mod_num_seq[i] = mod_dict[ch]
    else:
        # 如果蛋白序列小于等于列表，将蛋白序列放到列表中间
        list_block = math.ceil((max_seq_len - protein_length) / 2)
        for i, ch in enumerate(prot[:max_seq_len]):
            prot_num_seq[i + list_block] = seq_dict[ch]
        for i, ch in enumerate(mod_seq[:max_seq_len]):
            mod_num_seq[i + list_block] = mod_dict[ch]

    return prot_num_seq,mod_num_seq

def create_mod_list(mod_in_df):
    """Convert Alpha/Beta modification(s) columns into Alpha/Beta mod list"""
    if pd.isna(mod_in_df):  # Check if seq is NaN
        mod_list = None
    else:
        try:
            mod_list = [mod.split(' (')[0] for mod in mod_in_df.split('; ')]
        except AttributeError:
            mod_list = None
    return mod_list


class ModelDataset_old(data.Dataset):
    def __init__(self, data_df, max_peptide_len):

        print("Dataset preparing")
        # Create column of id
        data_df.reset_index(inplace=True)
        data_df.rename(columns={'index': 'id'}, inplace=True)

        # Create two columns: 'Alpha mod list' and 'Beta mod list'
        data_df['Alpha mod list'] = data_df['Alpha modification(s)'].apply(create_mod_list)
        data_df['Beta mod list'] = data_df['Beta modification(s)'].apply(create_mod_list)

        # Intensity(alpha/beta)字典，ratio是已经log2计算转换过
        intensity_ratio = data_df[["id", "Mean log2 Intensity(alpha/beta)"]].set_index("id")["Mean log2 Intensity(alpha/beta)"].to_dict()
        self.intensity_ratio = intensity_ratio

        # alpha_peptide and alpha_mod dictionary
        alpha_peptide = {}
        alpha_mod = {}
        for index, row in data_df[["id", "Alpha peptide", "Alpha mod list"]].iterrows():
            alpha_peptide[row.iloc[0]],alpha_mod[row.iloc[0]] = peptide_mod_to_num(row.iloc[1], max_peptide_len, row.iloc[2])
        self.alpha_peptide = alpha_peptide
        self.alpha_mod = alpha_mod
        self.max_alpha_peptide_len = max_peptide_len

        # beta_peptide and beta_mod dictionary
        beta_peptide = {}
        beta_mod = {}
        for index, row in data_df[["id", "Beta peptide", "Beta mod list"]].iterrows():
            beta_peptide[row.iloc[0]],beta_mod[row.iloc[0]] = peptide_mod_to_num(row.iloc[1], max_peptide_len, row.iloc[2])
        self.beta_peptide = beta_peptide
        self.beta_mod = beta_mod
        self.max_beta_peptide_len = max_peptide_len

        self.length = len(self.intensity_ratio)

    def __getitem__(self, index):
        label = list(self.intensity_ratio.values())[index]
        data_id = list(self.intensity_ratio.keys())[index]

        alpha_peptide_feature = self.alpha_peptide[data_id].astype(np.float32)
        alpha_mod_feature = self.alpha_mod[data_id].astype(np.float32)

        beta_peptide_feature = self.beta_peptide[data_id].astype(np.float32)
        beta_mod_feature = self.beta_mod[data_id].astype(np.float32)

        return alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, label, data_id

    def __len__(self):
        return self.length
    

class ModelDataset(data.Dataset):
    def __init__(self, data_df, max_peptide_len):
        print("Dataset preparing")
        # Create column of id
        data_df.reset_index(inplace=True)
        data_df.rename(columns={'index': 'id'}, inplace=True)

        self.data_df = data_df
        self.max_peptide_len = max_peptide_len

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        label = row["Mean log2 Intensity(alpha/beta)"]
        data_id = row["id"]

        # Lazy processing
        alpha_peptide_feature, alpha_mod_feature = peptide_mod_to_num(
            row["Alpha peptide"], self.max_peptide_len, create_mod_list(row["Alpha modification(s)"])
        )
        beta_peptide_feature, beta_mod_feature = peptide_mod_to_num(
            row["Beta peptide"], self.max_peptide_len, create_mod_list(row["Beta modification(s)"])
        )

        return (
            alpha_peptide_feature.astype(np.float32),
            beta_peptide_feature.astype(np.float32),
            alpha_mod_feature.astype(np.float32),
            beta_mod_feature.astype(np.float32),
            label,
            data_id
        )

    def __len__(self):
        return len(self.data_df)

    

if __name__ == "__main__":
    # open file with relative path
    # 获取当前脚本所在目录
    current_directory = r"M:\FB2\AG FLiu\Users\ZZH\MyProjects\XLProteomicDB\IntensityRatioPredictor\Model"
    csv_filename = 'DatasetBatch1_with_mean_intensity_ratio_240312.csv'
    csv_path = os.path.join(current_directory, '..', 'Dataset', csv_filename)

    # 测试Dataset
    data_df = pd.read_csv(csv_path)
    max_peptide_len = 30
    print(data_df.head())
    dataset = ModelDataset(data_df, max_peptide_len)
    alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, label, data_id = dataset[10]
    print('alpha peptide feature:', alpha_peptide_feature)
    print('alpha_mod_feature:',alpha_mod_feature)
    print('label', label)

