#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   test_model.py
# Time    :   2024/06/04 15:32:49
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics_utils import RMSE, MAE, CORR, SPEARMAN
from dataset import ModelDataset
from model import DeepIntRa

# ================================================================
# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, model):
    """Loads a trained model's state and sets it to evaluation mode."""
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, dataloader):
    """Evaluates the model and calculates key metrics."""
    model.eval()
    criterion = nn.MSELoss()

    total_loss, preds, truths = 0.0, [], []
    with torch.no_grad():
        for batch_data in dataloader:
            # Unpack batch data
            alpha_peptide, beta_peptide, alpha_mod, beta_mod, labels, _ = batch_data
            
            # Move tensors to the device
            alpha_peptide, beta_peptide = alpha_peptide.to(device), beta_peptide.to(device)
            alpha_mod, beta_mod = alpha_mod.to(device), beta_mod.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(alpha_peptide.int(), beta_peptide.int(), alpha_mod.int(), beta_mod.int())
            loss = criterion(outputs.float(), labels.float().unsqueeze(1))
            total_loss += loss.item()

            preds.extend(outputs.cpu().detach().numpy())
            truths.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(dataloader)
    test_RMSE = RMSE(truths, preds)
    test_MAE = MAE(truths, preds)
    test_CORR = CORR(np.array(truths), np.array(preds).flatten())
    test_SPEARMAN = SPEARMAN(np.array(truths), np.array(preds).flatten())

    return avg_loss, test_RMSE, test_MAE, test_CORR, test_SPEARMAN

def predict_result(model, dataloader):
    """Generates predictions using the model."""
    model.eval()
    preds, truths = [], []
    with torch.no_grad():
        for batch_data in dataloader:
            # Unpack and move data to the device
            alpha_peptide, beta_peptide, alpha_mod, beta_mod, labels, _ = batch_data
            alpha_peptide, beta_peptide = alpha_peptide.to(device), beta_peptide.to(device)
            alpha_mod, beta_mod = alpha_mod.to(device), beta_mod.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(alpha_peptide.int(), beta_peptide.int(), alpha_mod.int(), beta_mod.int())
            preds.extend(outputs.cpu().detach().numpy())
            truths.extend(labels.cpu().numpy())

    return preds, truths

# ================================================================
# Main execution block
if __name__ == "__main__":
    # Hyperparameters and paths
    BATCH_SIZE = 1024
    MAX_PEPTIDE_LEN = 48
    CSV_PATH = r"N:\AIRPred\Additional_test\model_evaluation_WC_HEK.csv"
    MODEL_DIR = r"N:\AIRPred\Additional_test\finetuned_models"
    OUTPUT_CSV_PATH = r"N:\AIRPred\Additional_test\model_evaluation_CW_HEK_predResult.csv"

    # Load the test dataset
    test_df = pd.read_csv(CSV_PATH)
    test_dataset = ModelDataset(test_df, MAX_PEPTIDE_LEN)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model_name = "DeepIntRa"
    model_backbone = DeepIntRa(
        conv_batch_1=16,
        conv_batch_2=64,
        hidden_layer_dim=64,
        num_heads=8,
        dropout1=0.3,
        dropout2=0.2
    ).to(device)

    # Load models for ensembling
    model_files = [
        'fine_tuned_model_DSBSO_CW.pt',
    ]
    model_paths = [os.path.join(MODEL_DIR, model_file) for model_file in model_files]

    # Ensemble predictions
    ensemble_results = []
    for model_path in model_paths:
        model = load_model(model_path, model_backbone)
        preds, _ = predict_result(model, test_dataloader)
        ensemble_results.append(preds)

    # Average predictions across models
    preds_final = np.mean(ensemble_results, axis=0)

    # Evaluate ensemble predictions
    _, truths = predict_result(model_backbone, test_dataloader)  # Truths remain consistent
    test_RMSE = RMSE(truths, preds_final)
    test_MAE = MAE(truths, preds_final)
    test_CORR = CORR(np.array(truths), np.array(preds_final).flatten())
    test_SPEARMAN = SPEARMAN(np.array(truths), np.array(preds_final).flatten())
    print(f"Metrics - RMSE: {test_RMSE:.4f}, MAE: {test_MAE:.4f}, CORR: {test_CORR:.4f}, SPEARMAN: {test_SPEARMAN:.4f}")

    # Save predictions to CSV
    test_df['PredIR'] = preds_final
    test_df.to_csv(OUTPUT_CSV_PATH, index=False)
