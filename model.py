#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------
# @File  : model.py
# @Time  : 2024/03/13 17:59:40
# @Author: Zhang, Zehong 
# @Email : zhang@fmp-berlin.de
# @Desc  : None
# -----------------------------------

import torch
import torch.nn as nn
import math
import numpy as np
import os
from datetime import datetime
import torch.nn.functional as F


class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()

class DeepIntRa(nn.Module):
    def __init__(self, conv_batch_1, conv_batch_2, hidden_layer_dim, num_heads, dropout1, dropout2):
        super(DeepIntRa, self).__init__()

        seq_embed_size = 60
        seq_feature_len = 25  
        mod_embed_size = 4
        mod_feature_len = 3  

        self.pep_embed = nn.Embedding(seq_feature_len + 1, seq_embed_size)
        self.mod_embed = nn.Embedding(mod_feature_len + 1, mod_embed_size)

        conv_peptide = []
        ic = seq_embed_size + mod_embed_size
        for oc in [conv_batch_1, conv_batch_2, 64]:
            conv_peptide.append(nn.Conv1d(ic, oc, 3))
            conv_peptide.append(nn.BatchNorm1d(oc))
            conv_peptide.append(nn.PReLU())
            ic = oc
        
        conv_peptide.append(nn.AdaptiveMaxPool1d(1))
        conv_peptide.append(Squeeze())
        self.conv_pep = nn.Sequential(*conv_peptide)

        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, dropout=dropout1)

        self.fc = nn.Sequential(nn.Linear(128, hidden_layer_dim),
                                nn.Tanh(),
                                nn.Dropout(dropout2),
                                nn.Linear(hidden_layer_dim, 1))

    def forward(self, alpha_peptide, beta_peptide, alpha_mod, beta_mod):
        alpha_pep_embed = self.pep_embed(alpha_peptide)  
        alpha_mod_embed = self.mod_embed(alpha_mod)
        alpha_pep_embed = torch.cat((alpha_pep_embed, alpha_mod_embed), dim=2)
        alpha_pep_embed = alpha_pep_embed.permute(0, 2, 1)  
        pep_a_conv = self.conv_pep(alpha_pep_embed) 


        beta_pep_embed = self.pep_embed(beta_peptide)  
        beta_mod_embed = self.mod_embed(beta_mod)
        beta_pep_embed = torch.cat((beta_pep_embed, beta_mod_embed), dim=2)
        beta_pep_embed = beta_pep_embed.permute(0, 2, 1) 
        pep_b_conv = self.conv_pep(beta_pep_embed) 

        ab_output, _ = self.attention(pep_a_conv.unsqueeze(0), pep_b_conv.unsqueeze(0), pep_b_conv.unsqueeze(0))
        ba_output, _ = self.attention(pep_b_conv.unsqueeze(0), pep_a_conv.unsqueeze(0), pep_a_conv.unsqueeze(0))

        ab_output = torch.reshape(ab_output,(1,-1,64))
        ba_output = torch.reshape(ba_output,(1,-1,64))

        output = torch.cat([ab_output.squeeze(0), ba_output.squeeze(0)], dim=1)

        output = self.fc(output)
        return output
    

def save_experiment_info(model, optimizer, criterion, epochs, batch_size, train_dataset_name, result_dict,model_name, Remarks = None):
    """Create a new folder to save model experiment results,
    return the new folder's path"""
    # Create a directory to save experiment information
    current_directory = os.path.dirname(__file__)
    experiment_dir = os.path.join(current_directory, '..', 'Result', f'{datetime.now().strftime("%Y%m%d_%H%M")}_experiment_results_{model_name}_{Remarks}')
    os.makedirs(experiment_dir, exist_ok=True)

    # Save experiment information to a text file
    experiment_info_path = os.path.join(experiment_dir, 'experiment_info.txt')
    with open(experiment_info_path, 'w') as f:
        f.write("="*50+"\n")
        f.write("Author: Zhang, Zehong\n")
        f.write(f"Email : zhang@fmp-berlin.de\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write(f"Other Information: {Remarks}\n")
        f.write("="*50+"\n")
        f.write("Experiment Information:\n")
        f.write(f"Train Dataset: {train_dataset_name}\n\n")
        f.write(f"Model Architecture: \n{model}\n\n")
        f.write(f"Optimizer: \n{optimizer}\n\n")
        f.write(f"Criterion: {criterion}\n\n")
        f.write(f"Number of Epochs: {epochs}\n\n")
        f.write(f"Batch Size: {batch_size}\n\n")
        f.write("-"*50+"\n")
        f.write("Result: \n")
        f.write(f"Val result: {result_dict}\n\n")
        

    print("Experiment information saved successfully!")
    return experiment_dir

def save_para_search_info(hyperparameter, train_dataset_name, result_dict, model_name, best_hyperparameters,best_val_result):
    # Create a directory to save experiment information
    current_directory = os.path.dirname(__file__)
    experiment_dir = os.path.join(current_directory, '..', 'Result', f'para_search_results_{model_name}')
    os.makedirs(experiment_dir, exist_ok=True)

    # Save experiment information to a text file
    experiment_info_path = os.path.join(experiment_dir, 'para_search_info.txt')
    with open(experiment_info_path, 'a') as f:
        f.write("-"*50+"\n")
        f.write(f"Time: {datetime.now()}\n")
        f.write("-"*50+"\n")
        f.write("Experiment Information:\n")
        f.write(f"Train Dataset: {train_dataset_name}\n")
        f.write(f"Model Hyperparameters: \n{hyperparameter}\n")
        f.write("-"*50+"\n")
        f.write("Result: \n")
        f.write(f"Val result: {result_dict}\n")
        f.write("-"*50+"\n")
        f.write(f"Best parameter:\n{best_hyperparameters}\n")
        f.write(f"Best val result:\n{best_val_result}\n")
        f.write("="*100+"\n\n")
        

    print("Experiment information saved successfully!")
    return experiment_dir

print("Model has been loaded!")
if __name__ == "__main__":
    pass