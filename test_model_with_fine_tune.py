#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   test_model_with_fine_tune.py
# Time    :   2024/06/05 12:19:59
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader, Dataset, random_split
from metrics_utils import RMSE, MAE, CORR, SPEARMAN
from dataset import ModelDataset
from model import DeepIntRa, save_experiment_info


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_model(model_path, model):
    """Load pre-trained model weights."""
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_result(model, dataloader):
    """Generate predictions and compute loss."""
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0
    preds, truths, alpha_pep, beta_pep = [], [], [], []

    with torch.no_grad():
        for batch_data in dataloader:
            alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, labels, _ = batch_data

            # Forward pass
            outputs = model(
                torch.IntTensor(alpha_peptide_feature.int()),
                torch.IntTensor(beta_peptide_feature.int()),
                torch.IntTensor(alpha_mod_feature.int()),
                torch.IntTensor(beta_mod_feature.int())
            )
            loss = criterion(outputs.float(), labels.float().unsqueeze(1))
            total_loss += loss.item()

            # Collect results
            alpha_pep.extend(alpha_peptide_feature.numpy())
            beta_pep.extend(beta_peptide_feature.numpy())
            preds.extend(outputs.numpy())
            truths.extend(labels.numpy())
    
    return alpha_pep, beta_pep, preds, truths


def fine_tune_model(model, dataloader, num_epochs=30, learning_rate=5e-5):
    """Fine-tune the model using a small dataset."""
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_data in dataloader:
            alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, labels, _ = batch_data

            # Forward pass
            outputs = model(
                torch.IntTensor(alpha_peptide_feature.int()),
                torch.IntTensor(beta_peptide_feature.int()),
                torch.IntTensor(alpha_mod_feature.int()),
                torch.IntTensor(beta_mod_feature.int())
            )
            loss = criterion(outputs.float(), labels.float().unsqueeze(1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def append_results_to_file(file_path, fine_tune_percentage, fine_tune_epoch, fine_tune_lr, results):
    """Log experimental results to a file."""
    with open(file_path, 'a') as file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Time: {current_time}\n")
        file.write(f"FINE_TUNE_PERCENTAGE: {fine_tune_percentage}, EPOCHS: {fine_tune_epoch}, LR: {fine_tune_lr}\n")
        file.write("Model Performance:\n")
        file.write(f"RMSE: {results[0]}, MAE: {results[1]}, CORR: {results[2]}, SPEARMAN: {results[3]}\n")
        file.write('-' * 50 + '\n')


if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    MAX_PEPTIDE_LEN = 48
    FINE_TUNE_PERCENTAGE = 0.3
    FINE_TUNE_EPOCH = 60
    FINE_TUNE_LEARNING_RATE = 5e-5
    EXPERIMENT_RECORD_FILE = "./finetune_results.txt"

    # Set random seed for reproducibility
    set_random_seed(1005)

    # Load test data
    test_data_path = os.path.join(
        os.path.dirname(__file__), '..', 'Dataset', 'Test', 'test_data.csv'
    )
    test_df = pd.read_csv(test_data_path)
    test_dataset = ModelDataset(test_df, MAX_PEPTIDE_LEN)

    # Split test data into fine-tuning and evaluation sets
    fine_tune_size = int(len(test_dataset) * FINE_TUNE_PERCENTAGE)
    eval_size = len(test_dataset) - fine_tune_size
    fine_tune_dataset, eval_dataset = random_split(test_dataset, [fine_tune_size, eval_size])

    fine_tune_loader = DataLoader(fine_tune_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Load model configurations
    model_name = "DeepIntRa"
    model_paths = [
        "./models/model_1.pt",
        "./models/model_2.pt",
        "./models/model_3.pt",
        "./models/model_4.pt",
        "./models/model_5.pt" # Or you can add more if you want to essemble
    ]
    model_backbone = DeepIntRa(conv_batch_1=16, conv_batch_2=64, hidden_layer_dim=64, num_heads=8, dropout1=0.3, dropout2=0.2)

    # Ensemble predictions
    ensemble_results = []
    for model_path in model_paths:
        model = load_model(model_path, model_backbone)

        # Fine-tune the model
        fine_tune_model(model, fine_tune_loader, num_epochs=FINE_TUNE_EPOCH, learning_rate=FINE_TUNE_LEARNING_RATE)

        # Generate predictions
        _, _, preds, truths = predict_result(model, eval_loader)
        ensemble_results.append(preds)

    # Average predictions from the ensemble
    final_preds = np.mean(ensemble_results, axis=0)

    # Save results
    results_df = pd.DataFrame({
        "Truths": truths,
        "Predictions": final_preds
    })
    results_df.to_csv("./results/predictions.csv", index=False)

    # Calculate metrics
    test_rmse = RMSE(truths, final_preds)
    test_mae = MAE(truths, final_preds)
    test_corr = CORR(np.array(truths), np.array(final_preds).flatten())
    test_spearman = SPEARMAN(np.array(truths), np.array(final_preds).flatten())

    results = [test_rmse, test_mae, test_corr, test_spearman]
    print(f"Performance: RMSE={test_rmse}, MAE={test_mae}, CORR={test_corr}, SPEARMAN={test_spearman}")

    # Append results to file
    append_results_to_file(EXPERIMENT_RECORD_FILE, FINE_TUNE_PERCENTAGE, FINE_TUNE_EPOCH, FINE_TUNE_LEARNING_RATE, results)
