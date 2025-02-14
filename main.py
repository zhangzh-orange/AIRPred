#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------
# @File  : main.py
# @Time  : 2024/03/14 15:48:58
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
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

from dataset import ModelDataset
from model import DeepIntRa, save_experiment_info
from metrics_utils import RMSE, MAE, CORR, SPEARMAN

# ================================================================
# Fix random seeds for reproducibility
RANDOM_SEED = 9999
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Hyperparameters
EPOCHS = 150
BATCH_SIZE = 32
LR = 5e-4
GAMMA = 0.96
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, optimizer, and scheduler
model = DeepIntRa(
    conv_batch_1=16,
    conv_batch_2=64,
    hidden_layer_dim=64,
    num_heads=8,
    dropout1=0.3,
    dropout2=0.2
)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)

# ================================================================
# Load dataset
current_directory = os.path.dirname(__file__)
csv_filename = './Train/total_data_duplicate_more_than3.csv'
csv_path = os.path.join(current_directory, '..', 'Dataset', csv_filename)
data_df = pd.read_csv(csv_path)
max_peptide_len = 48

# Split data into training and validation sets
train_df, val_df = train_test_split(data_df, test_size=0.2, random_state=RANDOM_SEED)

# Create datasets and dataloaders
train_dataset = ModelDataset(train_df, max_peptide_len)
val_dataset = ModelDataset(val_df, max_peptide_len)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Initialize lists to store metrics
train_losses, val_losses = [], []
val_RMSEs, val_MAEs, val_CORRs, val_SPEARMANs = [], [], [], []

# ================================================================
# Training and validation loop
for epoch in tqdm(range(EPOCHS)):
    model.to(device)
    model.train()
    running_loss = 0.0

    for batch_data in train_dataloader:
        alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, labels, _ = batch_data
        
        # Forward pass
        outputs = model(
            torch.IntTensor(alpha_peptide_feature.int()).to(device),
            torch.IntTensor(beta_peptide_feature.int()).to(device),
            torch.IntTensor(alpha_mod_feature.int()).to(device),
            torch.IntTensor(beta_mod_feature.int()).to(device)
        )
        loss = criterion(outputs.float(), labels.float().unsqueeze(1).to(device))
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss, preds, truths = 0.0, [], []
    with torch.no_grad():
        for batch_data in val_dataloader:
            alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, labels, _ = batch_data
            
            outputs = model(
                torch.IntTensor(alpha_peptide_feature.int()).to(device),
                torch.IntTensor(beta_peptide_feature.int()).to(device),
                torch.IntTensor(alpha_mod_feature.int()).to(device),
                torch.IntTensor(beta_mod_feature.int()).to(device)
            ).cpu()
            loss = criterion(outputs.float(), labels.float().unsqueeze(1))
            val_loss += loss.item()
            
            preds.extend(outputs.numpy())
            truths.extend(labels.numpy())

    # Compute validation metrics
    val_RMSE = RMSE(truths, preds)
    val_MAE = MAE(truths, preds)
    val_CORR = CORR(np.array(truths), np.array(preds).flatten())
    val_SPEARMAN = SPEARMAN(np.array(truths), np.array(preds).flatten())

    # Log metrics
    train_losses.append(running_loss / len(train_dataloader))
    val_losses.append(val_loss / len(val_dataloader))
    val_RMSEs.append(val_RMSE)
    val_MAEs.append(val_MAE)
    val_CORRs.append(val_CORR)
    val_SPEARMANs.append(val_SPEARMAN)

    print(f"\nEpoch {epoch+1}/{EPOCHS}:\n"
          f" Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}\n"
          f" Validation RMSE: {val_RMSE:.4f}, Validation MAE: {val_MAE:.4f}\n"
          f" Validation Pearson: {val_CORR:.4f}, Validation Spearman: {val_SPEARMAN:.4f}")

    lr_scheduler.step()

# ================================================================
# Save model and results
val_result_dict = {
    "Validation RMSE": val_RMSE,
    "Validation MAE": val_MAE,
    "Validation Pearson": val_CORR,
    "Validation Spearman": val_SPEARMAN
}
save_experiment_info(
    model,
    optimizer,
    criterion,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    train_dataset_name=csv_path,
    result_dict=val_result_dict,
    model_name="DeepIntRa",
    Remarks=f"SchedulerUsed_seed{RANDOM_SEED}"
)




