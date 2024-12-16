#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------
# @File  : main.py
# @Time  : 2024/03/14 15:48:58
# @Author: Zhang, Zehong 
# @Email : zhang@fmp-berlin.de
# @Desc  : None
# -----------------------------------

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.functional as F
import math
from torch.utils import data
import dataset
import torch.optim as optim
from torch.utils.data import DataLoader
import sklearn.metrics as m
import matplotlib.pyplot as plt
from metrics_utils import *
from dataset import ModelDataset
from model import CNNmodel,ParaSharedCNNmodel,Transformer,CNNTransformer,CNNAttention,DeepIntRa,save_experiment_info
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import StepLR

# ================================================================
# Set display option to show all rows
pd.set_option('display.max_rows', 100)

# Fix random seeds
RANDOM_SEED= 9999
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
# ================================================================
# Hyperparameters
EPOCHS = 150
BATCH_SIZE = 32
LR = 5e-04
# LR = 0.1
GAMMA = 0.96  # 学习率衰减因子,目前暂时不用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = CNNmodel()
# model = ParaSharedCNNmodel()
# model = CNNTransformer()
model = DeepIntRa(conv_batch_1=16, conv_batch_2=64, hidden_layer_dim=64, num_heads=8, dropout1=0.3, dropout2=0.2)
model_name = "DeepIntRa"
# criterion = nn.MSELoss()
# criterion = nn.L1Loss()
criterion = nn.SmoothL1Loss() # Loss function
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
# ================================================================

# Loaddataset by relative path
current_directory = os.path.dirname(__file__)
# csv_filename = './Train/total_data.csv'
csv_filename = './Train/total_data_duplicate_more_than3.csv'
csv_path = os.path.join(current_directory, '..', 'Dataset', csv_filename)
data_df = pd.read_csv(csv_path)
max_peptide_len = 48  # You may adjust this according to your data

# Split data into training and validation sets
train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=RANDOM_SEED)

# Create datasets and dataloaders
train_dataset = ModelDataset(train_df, max_peptide_len)
val_dataset = ModelDataset(val_df, max_peptide_len)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize lists to store metrics
train_losses = []
val_losses = []
val_RMSEs = []
val_MAEs = []
val_CORRs = []
val_SPEARMANs = []

# Train the model
for epoch in tqdm(range(EPOCHS)): 
    model.to(device)
    model.train()
    running_loss = 0.0
    for batch_idx, batch_data in enumerate(train_dataloader):
        alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, labels, data_ids = batch_data
        # Forward pass
        outputs = model(torch.IntTensor(alpha_peptide_feature.int()).to(device), 
                            torch.IntTensor(beta_peptide_feature.int()).to(device), 
                            torch.IntTensor(alpha_mod_feature.int()).to(device), 
                            torch.IntTensor(beta_mod_feature.int()).to(device))
        loss = criterion(outputs.float(), labels.float().unsqueeze(1).to(device))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Validate the model
    model.eval()
    val_loss = 0.0
    preds = []
    truths = []
    with torch.no_grad():
        for batch_data in val_dataloader:
            alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, labels, data_ids = batch_data

            outputs = model(torch.IntTensor(alpha_peptide_feature.int()).to(device), 
                            torch.IntTensor(beta_peptide_feature.int()).to(device), 
                            torch.IntTensor(alpha_mod_feature.int()).to(device), 
                            torch.IntTensor(beta_mod_feature.int()).to(device)).to("cpu")
            loss = criterion(outputs.float(), labels.float().unsqueeze(1))
            val_loss += loss.item()
            
            preds.extend(outputs.detach().numpy())
            truths.extend(labels.numpy())

    # Calculate validation metrics
    val_RMSE = RMSE(truths, preds)
    val_MAE = MAE(truths, preds)
    val_CORR = CORR(np.array(truths), np.array(preds).flatten())
    val_SPEARMAN = SPEARMAN(np.array(truths), np.array(preds).flatten())

    train_losses.append(running_loss/len(train_dataloader))
    val_losses.append(val_loss/len(val_dataloader))
    val_RMSEs.append(val_RMSE)
    val_MAEs.append(val_MAE)
    val_CORRs.append(val_CORR)
    val_SPEARMANs.append(val_SPEARMAN)

    print(f"\nEpoch {epoch+1}/{EPOCHS}:\n Training Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]},\n Validation RMSE: {val_RMSE}, Validation MAE: {val_MAE},\n Validation PEARSON: {val_CORR}, Validation SPEARMAN: {val_SPEARMAN}")

    # 更新学习率
    # print(optimizer)
    lr_scheduler.step()
    # if epoch == epochs - 1:
    #     # Create DataFrame from actual and predicted labels for the last epoch
    #     pred_result = pd.DataFrame({'Actual Labels': truths, 'Predicted Labels': preds})
    #     print(pred_result)


# ========================================================
# 保存模型，模型预测结果以及模型训练过程图
val_result_dict = {"Validation RMSE": val_RMSE, "Validation MAE": val_MAE,
                   "Validation Pearson":val_CORR,"Validation Spearman":val_SPEARMAN}
save_folder_path = save_experiment_info(model, optimizer, criterion,
                                        epochs = EPOCHS, batch_size = BATCH_SIZE,
                                        train_dataset_name = csv_path, result_dict=val_result_dict,
                                        model_name = model_name,Remarks=f"SchedulerUsed_seed{RANDOM_SEED}")

pred_result = pd.DataFrame({'Actual Labels': truths, 'Predicted Labels': [pred[0] for pred in preds]})
# print(pred_result.head(30))

# Plotting the metrics
epochs_range = range(1, EPOCHS + 1)

# Create a figure with subplots
plt.figure(figsize=(15, 6))

# First subplot: Training and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss',color = '#3b6291')
plt.plot(epochs_range, val_losses, label='Validation Loss', color = '#943c39')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Second subplot: Validation Metrics
plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_RMSEs, label='Validation RMSE',color = '#388498')
plt.plot(epochs_range, val_MAEs, label='Validation MAE',color = '#bf7334')
# plt.plot(epochs_range, val_CORRs, label='Validation CORR')
# plt.plot(epochs_range, val_SDs, label='Validation SD')
plt.xlabel('Epochs')
plt.title('Validation Metrics')
plt.legend()

# Adjust layout to prevent overlap of subplots
plt.tight_layout()

# Save the model, predict result and plots into the result folder
pred_result.to_csv(os.path.join(save_folder_path, "Predictor_Val_Result.csv"),index = False)
torch.save(model.state_dict(), os.path.join(save_folder_path, f"{model_name}_Params.pt"))
plt.savefig(os.path.join(save_folder_path, f"{model_name}_Val_Result.svg"))
plt.savefig(os.path.join(save_folder_path, f"{model_name}_Val_Result.pdf"))
print("Experiment result saved successfully!")

# Show the plot
# plt.show()


