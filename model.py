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
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()

# 需改动模型
class CNNmodel(nn.Module):
    """初始模型，没有将alpha beta网络参数共享"""
    def __init__(self):
        super(CNNmodel, self).__init__()

        seq_embed_size = 60
        seq_feature_len = 25  # 序列转化为数字的总数
        mod_embed_size = 4
        mod_feature_len = 3  # mod转化为数字的总数

        self.alpha_pep_embed = nn.Embedding(seq_feature_len + 1, seq_embed_size)  # output:[batch_size,seq_len,embedding_size]
        self.beta_pep_embed = nn.Embedding(seq_feature_len + 1, seq_embed_size)
        self.alpha_mod_embed = nn.Embedding(mod_feature_len + 1, mod_embed_size)
        self.beta_mod_embed = nn.Embedding(mod_feature_len + 1, mod_embed_size)

        # embed后进行transpose [seq_len,batch_size,embedding_size]
        # [32,128,270]
        conv_peptide_alpha = []
        ic = seq_embed_size + mod_embed_size
        for oc in [32, 64, 128]:
            conv_peptide_alpha.append(nn.Conv1d(ic, oc, 3,dilation=2))
            conv_peptide_alpha.append(nn.BatchNorm1d(oc))
            conv_peptide_alpha.append(nn.PReLU())
            ic = oc
            # 1: [32,128,270-2]
            # 2: [32,128,270-4]
            # 3: [32,128,270-6]
        conv_peptide_alpha.append(nn.AdaptiveMaxPool1d(1))  # [32,128,1]
        conv_peptide_alpha.append(Squeeze())  # [32,128]
        self.conv_pep_alpha = nn.Sequential(*conv_peptide_alpha)

        # [32,128,270]
        conv_peptide_beta = []
        ic = seq_embed_size + mod_embed_size
        for oc in [32, 64, 128]:
            conv_peptide_beta.append(nn.Conv1d(ic, oc, 3,dilation=2))
            conv_peptide_beta.append(nn.BatchNorm1d(oc))
            conv_peptide_beta.append(nn.PReLU())
            ic = oc
            # 1: [32,128,270-2]
            # 2: [32,128,270-4]
            # 3: [32,128,270-6]
        conv_peptide_beta.append(nn.AdaptiveMaxPool1d(1))  # [32,128,1]
        conv_peptide_beta.append(Squeeze())  # [32,128]
        self.conv_pep_beta = nn.Sequential(*conv_peptide_beta)

        self.cat_dropout = nn.Dropout(0.2)

        self.fc_layers = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.Dropout(0.2),
            nn.PReLU(),
            nn.Linear(64, 1),
            nn.PReLU())

    def forward(self, alpha_peptide, beta_peptide, alpha_mod, beta_mod):
        # input shape [batch_size,input_len]
        alpha_pep_embed = self.alpha_pep_embed(alpha_peptide)  # [32,270,128]
        alpha_mod_embed = self.alpha_mod_embed(alpha_mod)
        alpha_pep_embed = torch.cat((alpha_pep_embed, alpha_mod_embed), dim=2)
        alpha_pep_embed = torch.transpose(alpha_pep_embed, 1, 2)  # [32,128,270]
        pep_a_conv = self.conv_pep_alpha(alpha_pep_embed)  # [32,128]


        beta_pep_embed = self.beta_pep_embed(beta_peptide)  # [32,270,128]
        beta_mod_embed = self.beta_mod_embed(beta_mod)
        beta_pep_embed = torch.cat((beta_pep_embed, beta_mod_embed), dim=2)
        beta_pep_embed = torch.transpose(beta_pep_embed, 1, 2)  # [32,128,270]
        pep_b_conv = self.conv_pep_beta(beta_pep_embed)  # [32,128]

        cat = torch.cat([pep_a_conv, pep_b_conv], dim=1)  # [32,128*3]
        cat = self.cat_dropout(cat)

        output = self.fc_layers(cat) # [32,1]
        return output
    
class ParaSharedCNNmodel(nn.Module):
    """将alpha beta网络参数共享"""
    def __init__(self):
        super(ParaSharedCNNmodel, self).__init__()

        seq_embed_size = 60
        seq_feature_len = 25  # 序列转化为数字的总数
        mod_embed_size = 4
        mod_feature_len = 3  # mod转化为数字的总数

        self.pep_embed = nn.Embedding(seq_feature_len + 1, seq_embed_size)  # output:[batch_size,seq_len,embedding_size]
        self.mod_embed = nn.Embedding(mod_feature_len + 1, mod_embed_size)

        # embed后进行transpose [seq_len,batch_size,embedding_size]
        # [32,128,270]
        conv_peptide = []
        ic = seq_embed_size + mod_embed_size
        for oc in [32, 64, 128]:
            conv_peptide.append(nn.Conv1d(ic, oc, 3,dilation=2))
            conv_peptide.append(nn.BatchNorm1d(oc))
            conv_peptide.append(nn.PReLU())
            ic = oc
            # 1: [32,128,270-2]
            # 2: [32,128,270-4]
            # 3: [32,128,270-6]
        conv_peptide.append(nn.AdaptiveMaxPool1d(1))  # [32,128,1]
        conv_peptide.append(Squeeze())  # [32,128]
        self.conv_pep = nn.Sequential(*conv_peptide)

        self.cat_dropout = nn.Dropout(0.2)

        self.fc_layers = nn.Sequential(
            nn.Linear(128 + 128, 64),
            nn.Dropout(0.2),
            nn.PReLU(),
            nn.Linear(64, 1),
            nn.PReLU())

    def forward(self, alpha_peptide, beta_peptide, alpha_mod, beta_mod):
        # input shape [batch_size,input_len]
        alpha_pep_embed = self.pep_embed(alpha_peptide)  # [32,270,128]
        alpha_mod_embed = self.mod_embed(alpha_mod)
        alpha_pep_embed = torch.cat((alpha_pep_embed, alpha_mod_embed), dim=2)
        alpha_pep_embed = torch.transpose(alpha_pep_embed, 1, 2)  # [32,128,270]
        pep_a_conv = self.conv_pep(alpha_pep_embed)  # [32,128]


        beta_pep_embed = self.pep_embed(beta_peptide)  # [32,270,128]
        beta_mod_embed = self.mod_embed(beta_mod)
        beta_pep_embed = torch.cat((beta_pep_embed, beta_mod_embed), dim=2)
        beta_pep_embed = torch.transpose(beta_pep_embed, 1, 2)  # [32,128,270]
        pep_b_conv = self.conv_pep(beta_pep_embed)  # [32,128]

        cat = torch.cat([pep_a_conv, pep_b_conv], dim=1)  # [32,128*3]
        cat = self.cat_dropout(cat)

        output = self.fc_layers(cat) # [32,1]
        return output
    

class Transformer(nn.Module):
    """需要重新检查，这个模型性能不好原因可能是中间将batch维度作为单独数据维度计算，后续检查各输出维度变化 """
    def __init__(self):
        super(Transformer, self).__init__()

        seq_embed_size = 60
        seq_feature_len = 25  # 序列转化为数字的总数
        mod_embed_size = 4
        mod_feature_len = 3  # mod转化为数字的总数

        self.pep_embed = nn.Embedding(seq_feature_len + 1, seq_embed_size)
        self.mod_embed = nn.Embedding(mod_feature_len + 1, mod_embed_size)

        self.transformer_layer = nn.Transformer(d_model=seq_embed_size + mod_embed_size, 
                                                nhead=8, 
                                                dim_feedforward=64,
                                                batch_first=True)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, alpha_peptide, beta_peptide, alpha_mod, beta_mod):
        alpha_pep_embed = self.pep_embed(alpha_peptide)
        alpha_mod_embed = self.mod_embed(alpha_mod)
        alpha_pep_embed = torch.cat((alpha_pep_embed, alpha_mod_embed), dim=2)
        alpha_pep_embed = alpha_pep_embed.permute(1, 0, 2)
        print(alpha_pep_embed.shape)

        beta_pep_embed = self.pep_embed(beta_peptide)
        beta_mod_embed = self.mod_embed(beta_mod)
        beta_pep_embed = torch.cat((beta_pep_embed, beta_mod_embed), dim=2)
        beta_pep_embed = beta_pep_embed.permute(1, 0, 2)

        ab_output = self.transformer_layer(alpha_pep_embed,beta_pep_embed)
        ba_output = self.transformer_layer(beta_pep_embed,alpha_pep_embed)

        ab_output = ab_output.mean(dim=0)
        ba_output = ba_output.mean(dim=0)

        cat = torch.cat([ab_output, ba_output], dim=1)
        cat = self.dropout(cat)

        output = self.fc2(self.dropout(self.fc1(cat)))
        return output
    
class CNNTransformer(nn.Module):
    
    def __init__(self):
        super(CNNTransformer, self).__init__()

        seq_embed_size = 60
        seq_feature_len = 25  # 序列转化为数字的总数
        mod_embed_size = 4
        mod_feature_len = 3  # mod转化为数字的总数

        self.pep_embed = nn.Embedding(seq_feature_len + 1, seq_embed_size)
        self.mod_embed = nn.Embedding(mod_feature_len + 1, mod_embed_size)

        conv_peptide = []
        ic = seq_embed_size + mod_embed_size
        for oc in [16, 32, 64]:
            conv_peptide.append(nn.Conv1d(ic, oc, 3))
            conv_peptide.append(nn.BatchNorm1d(oc))
            conv_peptide.append(nn.PReLU())
            ic = oc
        conv_peptide.append(nn.AdaptiveMaxPool1d(1))
        conv_peptide.append(Squeeze())
        self.conv_pep = nn.Sequential(*conv_peptide)

        self.transformer_layer = nn.Transformer(d_model=64, 
                                                nhead=4, 
                                                dim_feedforward=64,
                                                batch_first=True,
                                                num_encoder_layers=1,
                                                num_decoder_layers=1,
                                                dropout=0.2)

        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.2)

    def forward(self, alpha_peptide, beta_peptide, alpha_mod, beta_mod):
        alpha_pep_embed = self.pep_embed(alpha_peptide)  # [32,270,128]
        alpha_mod_embed = self.mod_embed(alpha_mod)
        alpha_pep_embed = torch.cat((alpha_pep_embed, alpha_mod_embed), dim=2)
        alpha_pep_embed = torch.transpose(alpha_pep_embed, 1, 2)  # [32,128,270]
        pep_a_conv = self.conv_pep(alpha_pep_embed)  # [32,128]
        

        beta_pep_embed = self.pep_embed(beta_peptide)  # [32,270,128]
        beta_mod_embed = self.mod_embed(beta_mod)
        beta_pep_embed = torch.cat((beta_pep_embed, beta_mod_embed), dim=2)
        beta_pep_embed = torch.transpose(beta_pep_embed, 1, 2)  # [32,128,270]
        pep_b_conv = self.conv_pep(beta_pep_embed)  # [32,128]


        ab_output = self.transformer_layer(pep_a_conv,pep_b_conv)
        ba_output = self.transformer_layer(pep_b_conv,pep_a_conv)
        output = torch.cat([ab_output, ba_output], dim=1)

        output = self.fc2(self.dropout(self.fc1(output)))
        return output
    
class CNNAttention(nn.Module):
    
    def __init__(self):
        super(CNNAttention, self).__init__()

        seq_embed_size = 60
        seq_feature_len = 25  # Total number of features for the sequence
        mod_embed_size = 4
        mod_feature_len = 3  # Total number of features for mod

        self.pep_embed = nn.Embedding(seq_feature_len + 1, seq_embed_size)
        self.mod_embed = nn.Embedding(mod_feature_len + 1, mod_embed_size)

        conv_peptide = []
        ic = seq_embed_size + mod_embed_size
        for oc in [16, 32, 64]:
            conv_peptide.append(nn.Conv1d(ic, oc, 3))
            conv_peptide.append(nn.BatchNorm1d(oc))
            conv_peptide.append(nn.PReLU())
            ic = oc
        # conv_peptide.append(nn.AdaptiveMaxPool1d(1))
        conv_peptide.append(nn.AdaptiveAvgPool1d(1))
        conv_peptide.append(Squeeze())
        self.conv_pep = nn.Sequential(*conv_peptide)

        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, dropout=0.5)

        self.fc = nn.Sequential(nn.Linear(128, 64),
                                nn.Tanh(),
                                nn.Dropout(0.2),
                                nn.Linear(64, 1))

    def forward(self, alpha_peptide, beta_peptide, alpha_mod, beta_mod):
        alpha_pep_embed = self.pep_embed(alpha_peptide)  # [batch_size, seq_len, seq_embed_size]
        alpha_mod_embed = self.mod_embed(alpha_mod)
        alpha_pep_embed = torch.cat((alpha_pep_embed, alpha_mod_embed), dim=2)
        alpha_pep_embed = alpha_pep_embed.permute(0, 2, 1)  # [batch_size, seq_embed_size + mod_embed_size, seq_len]
        pep_a_conv = self.conv_pep(alpha_pep_embed)  # [batch_size, 128]


        beta_pep_embed = self.pep_embed(beta_peptide)  # [batch_size, seq_len, seq_embed_size]
        beta_mod_embed = self.mod_embed(beta_mod)
        beta_pep_embed = torch.cat((beta_pep_embed, beta_mod_embed), dim=2)
        beta_pep_embed = beta_pep_embed.permute(0, 2, 1)  # [batch_size, seq_embed_size + mod_embed_size, seq_len]
        pep_b_conv = self.conv_pep(beta_pep_embed)  # [batch_size, 128]

        # Multihead attention
        ab_output, _ = self.attention(pep_a_conv.unsqueeze(0), pep_b_conv.unsqueeze(0), pep_b_conv.unsqueeze(0))
        ba_output, _ = self.attention(pep_b_conv.unsqueeze(0), pep_a_conv.unsqueeze(0), pep_a_conv.unsqueeze(0))
        output = torch.cat([ab_output.squeeze(0), ba_output.squeeze(0)], dim=1)

        output = self.fc(output)
        return output

class DeepIntRa(nn.Module):
    def __init__(self, conv_batch_1, conv_batch_2, hidden_layer_dim, num_heads, dropout1, dropout2):
        super(DeepIntRa, self).__init__()

        seq_embed_size = 60
        seq_feature_len = 25  # Total number of features for the sequence
        mod_embed_size = 4
        mod_feature_len = 3  # Total number of features for mod

        self.pep_embed = nn.Embedding(seq_feature_len + 1, seq_embed_size)
        self.mod_embed = nn.Embedding(mod_feature_len + 1, mod_embed_size)

        conv_peptide = []
        ic = seq_embed_size + mod_embed_size
        for oc in [conv_batch_1, conv_batch_2, 64]:
            conv_peptide.append(nn.Conv1d(ic, oc, 3))
            conv_peptide.append(nn.BatchNorm1d(oc))
            conv_peptide.append(nn.PReLU())
            ic = oc
        # conv_peptide.append(nn.AdaptiveMaxPool1d(1))
        conv_peptide.append(nn.AdaptiveMaxPool1d(1))
        conv_peptide.append(Squeeze())
        self.conv_pep = nn.Sequential(*conv_peptide)

        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=num_heads, dropout=dropout1)

        self.fc = nn.Sequential(nn.Linear(128, hidden_layer_dim),
                                nn.Tanh(),
                                nn.Dropout(dropout2),
                                nn.Linear(hidden_layer_dim, 1))

    def forward(self, alpha_peptide, beta_peptide, alpha_mod, beta_mod):
        alpha_pep_embed = self.pep_embed(alpha_peptide)  # [batch_size, seq_len, seq_embed_size]
        alpha_mod_embed = self.mod_embed(alpha_mod)
        alpha_pep_embed = torch.cat((alpha_pep_embed, alpha_mod_embed), dim=2)
        alpha_pep_embed = alpha_pep_embed.permute(0, 2, 1)  # [batch_size, seq_embed_size + mod_embed_size, seq_len]
        pep_a_conv = self.conv_pep(alpha_pep_embed)  # [batch_size, 128]


        beta_pep_embed = self.pep_embed(beta_peptide)  # [batch_size, seq_len, seq_embed_size]
        beta_mod_embed = self.mod_embed(beta_mod)
        beta_pep_embed = torch.cat((beta_pep_embed, beta_mod_embed), dim=2)
        beta_pep_embed = beta_pep_embed.permute(0, 2, 1)  # [batch_size, seq_embed_size + mod_embed_size, seq_len]
        pep_b_conv = self.conv_pep(beta_pep_embed)  # [batch_size, 128]

        # Multihead attention
        ab_output, _ = self.attention(pep_a_conv.unsqueeze(0), pep_b_conv.unsqueeze(0), pep_b_conv.unsqueeze(0))
        ba_output, _ = self.attention(pep_b_conv.unsqueeze(0), pep_a_conv.unsqueeze(0), pep_a_conv.unsqueeze(0))

        # 防止最后一批数据只有一个，导致shape为1，64而不是1，1，64而出错
        ab_output = torch.reshape(ab_output,(1,-1,64))
        ba_output = torch.reshape(ba_output,(1,-1,64))
        # print(ab_output.shape)
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
    
def generate_data(batch_size, seq_len):
    """generate data for testing model structure"""
    peptide_alpha = torch.randint(0, 20, size=(batch_size, seq_len))  # Random antibody_a data (0 to 48)
    peptide_beta = torch.randint(0, 20, size=(batch_size, seq_len))  # Random antibody_b data (0 to 48)
    alpha_mod = torch.randint(0, 3, size=(batch_size, seq_len))  # Random mod_a data (0 to 3)
    beta_mod = torch.randint(0, 3, size=(batch_size, seq_len))  # Random mod_b data (0 to 3)
    return peptide_alpha, peptide_beta, alpha_mod, beta_mod

print("Model has been loaded!")
if __name__ == "__main__":
    # 测试模型是否能跑通
    # Create an instance of the CNNmodel
    model = CNNAttention()

    # Generate random input data
    batch_size = 2
    seq_len = 48
    peptide_alpha, peptide_beta, alpha_mod, beta_mod = generate_data(batch_size, seq_len)
    print(len(peptide_alpha))
    print(torch.cat([peptide_alpha, peptide_beta, alpha_mod, beta_mod],dim=1).shape)

    # Pass the input data through the model
    output = model(peptide_alpha, peptide_beta, alpha_mod, beta_mod)
    print("type:",type(peptide_alpha))
    print("peptide_alpha", peptide_alpha)
    print("peptide_alpha shape:", peptide_alpha.shape)
    print( "alpha_mod shape:", alpha_mod.shape)
    print("Output:", output)