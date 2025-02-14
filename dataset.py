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
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Set pandas options for better display
pd.set_option('display.max_columns', 1000)

# Constants for peptide processing
SEQ_VOCAB = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
SEQ_DICT = {v: (i + 1) for i, v in enumerate(SEQ_VOCAB)}
SEQ_DICT_LEN = len(SEQ_DICT)

# Constants for modification processing
MOD_VOCAB = "MC"
MOD_DICT = {v: (i + 1) for i, v in enumerate(MOD_VOCAB)}
MOD_DICT['-'] = 0  # Placeholder for positions without modifications
MOD_DICT_LEN = len(MOD_DICT)

# Generate all possible tri-grams from the sequence vocabulary
TRI_GRAM_LIST = [x + y + z for x in SEQ_VOCAB for y in SEQ_VOCAB for z in SEQ_VOCAB]
TRI_GRAM_DICT = {v: (i + 1) for i, v in enumerate(TRI_GRAM_LIST)}

def peptide_mod_to_num(
    prot: str, max_seq_len: int, mod_list: list = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert peptide sequence into numerical representations with modifications.
    
    Args:
        prot (str): Peptide sequence.
        max_seq_len (int): Maximum length of the sequence.
        mod_list (list): List of modifications (e.g., ['M6', 'C2']).
        
    Returns:
        tuple: Arrays representing the peptide sequence and modifications.
    """
    prot = prot.strip().replace(" ", "")
    protein_length = len(prot) + 1

    # Handle modifications
    mod_seq = "-" * len(prot)
    if isinstance(mod_list, list):
        try:
            mod_site_type_dict = {int(mod[1:]): mod[0] for mod in mod_list}
            for site, mod_type in mod_site_type_dict.items():
                mod_seq = mod_seq[:site] + mod_type + mod_seq[site + 1:]
        except (AttributeError, ValueError):
            pass

    prot_num_seq = np.zeros(max_seq_len, dtype=int)
    mod_num_seq = np.zeros(max_seq_len, dtype=int)

    if protein_length > max_seq_len:
        # If sequence is longer than max length, truncate from N-terminus
        prot_num_seq[:max_seq_len] = [SEQ_DICT[ch] for ch in prot[:max_seq_len]]
        mod_num_seq[:max_seq_len] = [MOD_DICT[ch] for ch in mod_seq[:max_seq_len]]
    else:
        # If sequence fits, center-align it in the array
        list_block = math.ceil((max_seq_len - protein_length) / 2)
        for i, ch in enumerate(prot):
            prot_num_seq[i + list_block] = SEQ_DICT[ch]
        for i, ch in enumerate(mod_seq):
            mod_num_seq[i + list_block] = MOD_DICT[ch]

    return prot_num_seq, mod_num_seq

def create_mod_list(mod_in_df: str) -> list | None:
    """
    Convert modification columns into a list of modifications.

    Args:
        mod_in_df (str): String of modifications from the dataset.

    Returns:
        list | None: List of modifications or None if no modifications.
    """
    if pd.isna(mod_in_df):
        return None
    try:
        return [mod.split(" (")[0] for mod in mod_in_df.split("; ")]
    except AttributeError:
        return None

class ModelDataset(Dataset):
    """
    PyTorch Dataset for processing peptide data.
    """
    def __init__(self, data_df: pd.DataFrame, max_peptide_len: int):
        """
        Initialize the dataset.

        Args:
            data_df (pd.DataFrame): Input data as a pandas DataFrame.
            max_peptide_len (int): Maximum peptide sequence length.
        """
        print("Dataset preparing...")
        data_df = data_df.reset_index().rename(columns={"index": "id"})
        self.data_df = data_df
        self.max_peptide_len = max_peptide_len

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve a single data sample.

        Args:
            index (int): Index of the data sample.

        Returns:
            tuple: Features and label for the sample.
        """
        row = self.data_df.iloc[index]
        label = row["Mean log2 Intensity(alpha/beta)"]
        data_id = row["id"]

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
            data_id,
        )

    def __len__(self) -> int:
        """
        Return the size of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data_df)

if __name__ == "__main__":
    pass
