#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   test_model.py
# Time    :   2024/06/04 15:32:49
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
from metrics_utils import *
from dataset import ModelDataset
from model import CNNmodel,ParaSharedCNNmodel,Transformer,CNNTransformer,CNNAttention,DeepIntRa,save_experiment_info

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    preds = []
    truths = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, labels, data_ids = batch_data
            # Move tensors to device
            alpha_peptide_feature = alpha_peptide_feature.to(device)
            beta_peptide_feature = beta_peptide_feature.to(device)
            alpha_mod_feature = alpha_mod_feature.to(device)
            beta_mod_feature = beta_mod_feature.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(alpha_peptide_feature.int(),
                            beta_peptide_feature.int(),
                            alpha_mod_feature.int(),
                            beta_mod_feature.int())
            loss = criterion(outputs.float(), labels.float().unsqueeze(1))
            total_loss += loss.item()

            preds.extend(outputs.cpu().detach().numpy())
            truths.extend(labels.cpu().numpy())

    average_loss = total_loss / len(dataloader)
    # Calculate validation metrics
    test_RMSE = RMSE(truths, preds)
    test_MAE = MAE(truths, preds)
    test_CORR = CORR(np.array(truths), np.array(preds).flatten())
    test_SPEARMAN = SPEARMAN(np.array(truths), np.array(preds).flatten())
    return [test_RMSE, test_MAE, test_CORR, test_SPEARMAN]

def predict_result(model, dataloader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    preds = []
    truths = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, labels, data_ids = batch_data
            # Move tensors to device
            alpha_peptide_feature = alpha_peptide_feature.to(device)
            beta_peptide_feature = beta_peptide_feature.to(device)
            alpha_mod_feature = alpha_mod_feature.to(device)
            beta_mod_feature = beta_mod_feature.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(alpha_peptide_feature.int(),
                            beta_peptide_feature.int(),
                            alpha_mod_feature.int(),
                            beta_mod_feature.int())
            loss = criterion(outputs.float(), labels.float().unsqueeze(1))
            total_loss += loss.item()

            preds.extend(outputs.cpu().detach().numpy())
            truths.extend(labels.cpu().numpy())
    return preds, truths


# Example usage
if __name__ == "__main__":
    BATCH_SIZE = 1024
    max_peptide_len = 48
    # Load test data
    csv_path = r"N:\AIRPred\Additional_test\Cong_HEK\model_evaluation_WC_HEK.csv"
    test_df = pd.read_csv(csv_path)
    test_dataset = ModelDataset(test_df, max_peptide_len)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_name = "DeepIntRa"
    models = ['fine_tuned_model_DSBSO_CW_1.pt',
              'fine_tuned_model_DSBSO_CW_2.pt']
    model_paths = [r"N:\AIRPred\Additional_test\Cong_HEK\finetuned_models\{}".format(i) for i in models]
    model_backbone = DeepIntRa(conv_batch_1=16, conv_batch_2=64, hidden_layer_dim=64, num_heads=8, dropout1=0.3, dropout2=0.2).to(device)

    essemble_results = []
    for model_path in model_paths:
        model = load_model(model_path, model_backbone)
        preds, truths = predict_result(model, test_dataloader)
        essemble_results.append(preds)

    preds_final = np.mean(essemble_results, axis=0)

    test_RMSE = RMSE(truths, preds_final)
    test_MAE = MAE(truths, preds_final)
    test_CORR = CORR(np.array(truths), np.array(preds_final).flatten())
    test_SPEARMAN = SPEARMAN(np.array(truths), np.array(preds_final).flatten())
    print([test_RMSE, test_MAE, test_CORR, test_SPEARMAN])

    test_df['PredIR'] = preds_final
    test_df.to_csv(r"N:\AIRPred\Additional_test\Cong_HEK\model_evaluation_CW_HEK_predResult.csv")
