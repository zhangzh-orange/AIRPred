#! python3
# -*- encoding: utf-8 -*-
# ===================================
# File    :   test_model_with_fine_tune.py
# Time    :   2024/06/05 12:19:59
# Author  :   Zehong Zhang
# Contact :   zhang@fmp-berlin.de
# ===================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import os
import pandas as pd
import numpy as np
from datetime import datetime
from metrics_utils import *
from dataset import ModelDataset
from model import CNNmodel,ParaSharedCNNmodel,Transformer,CNNTransformer,CNNAttention,DeepIntRa,save_experiment_info


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(model_path, model):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_result(model, dataloader):
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()

    preds = []
    truths = []
    alpha_pep = []
    beta_pep = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, labels, data_ids = batch_data
            # Forward pass
            outputs = model(torch.IntTensor(alpha_peptide_feature.int()), 
                                torch.IntTensor(beta_peptide_feature.int()), 
                                torch.IntTensor(alpha_mod_feature.int()), 
                                torch.IntTensor(beta_mod_feature.int()))
            loss = criterion(outputs.float(), labels.float().unsqueeze(1))
            total_loss += loss.item()
            
            alpha_pep.extend(alpha_peptide_feature.detach().numpy())
            beta_pep.extend(beta_peptide_feature.detach().numpy())
            preds.extend(outputs.detach().numpy())
            truths.extend(labels.numpy())
    return alpha_pep, beta_pep, preds, truths

def fine_tune_model(model, dataloader, num_epochs=30, learning_rate=5e-5):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for batch_idx, batch_data in enumerate(dataloader):
            alpha_peptide_feature, beta_peptide_feature, alpha_mod_feature, beta_mod_feature, labels, data_ids = batch_data
            # Forward pass
            outputs = model(torch.IntTensor(alpha_peptide_feature.int()), 
                                torch.IntTensor(beta_peptide_feature.int()), 
                                torch.IntTensor(alpha_mod_feature.int()), 
                                torch.IntTensor(beta_mod_feature.int()))
            loss = criterion(outputs.float(), labels.float().unsqueeze(1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def append_results_to_file(file_path, fine_tune_persontage, fine_tune_epoch, fine_tune_learn_rate, results):
    with open(file_path, 'a') as file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Time: {current_time}\n")
        file.write(f"FINE_TUNE_PERCENTAGE:{fine_tune_persontage}, FINE_TUNE_EPOCH: {fine_tune_epoch}, FINE_TUNE_LEARN_RATE: {fine_tune_learn_rate}\n")
        file.write("Model Performance:\n")
        file.write(f"RMSE: {results[0]}, MAE: {results[1]}, CORR: {results[2]}, SPEARMAN: {results[3]}\n")
        file.write('-' * 50 + '\n')

# Example usage
if __name__ == "__main__":
    BATCH_SIZE = 32
    max_peptide_len = 48
    FINE_TUNE_PERCENTAGE = 0.3
    FINE_TUNE_EPOCH = 60
    FINE_TUNE_LEARN_RATE = 5e-5
    EXPERIMENT_RECORD_FILE = r"M:\FB2\AG FLiu\Users\ZZH\MyProjects\XLProteomicDB\IntensityRatioPredictor\Result\0-Finetune_parameter_test_result.txt"
    # Set random seed for reproducibility
    set_random_seed(1005)

    # Load test data
    current_directory = os.path.dirname(__file__)
    csv_path = os.path.join(current_directory, '..', 'Dataset', "./Test/test_MxR_HEK_DSSO_HCD21to30_dropduplicates.csv")  # HCD 21-30
    test_df = pd.read_csv(csv_path)
    test_dataset = ModelDataset(test_df, max_peptide_len)

    # Split test data into fine-tuning and evaluation sets
    total_size = len(test_dataset)
    fine_tune_size = int(total_size * FINE_TUNE_PERCENTAGE)
    eval_size = total_size - fine_tune_size
    fine_tune_dataset, eval_dataset = random_split(test_dataset, [fine_tune_size, eval_size])

    fine_tune_dataloader = DataLoader(fine_tune_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model_name = "DeepIntRa"
    models = ['20240604_1724_experiment_results_DeepIntRa_SchedulerUsed_seed9999',
              '20240604_1718_experiment_results_DeepIntRa_SchedulerUsed_seed10000',
              '20240604_1710_experiment_results_DeepIntRa_SchedulerUsed_seed123123',
              '20240604_1656_experiment_results_DeepIntRa_SchedulerUsed_seed1234',
              '20240604_1507_experiment_results_DeepIntRa_SchedulerUsed_seed1005']
    model_paths = [r"M:\FB2\AG FLiu\Users\ZZH\MyProjects\XLProteomicDB\IntensityRatioPredictor\Result\{}\DeepIntRa_Params.pt".format(i) for i in models]
    model_backbone = DeepIntRa(conv_batch_1=16, conv_batch_2=64, hidden_layer_dim=64, num_heads=8, dropout1=0.3, dropout2=0.2)
    
    essemble_results = []
    for model_path in model_paths:
        model = load_model(model_path, model_backbone)
        
        # Fine-tune the model with 10% of the test data
        fine_tune_model(model, fine_tune_dataloader, num_epochs=FINE_TUNE_EPOCH, learning_rate=FINE_TUNE_LEARN_RATE)
        
        alpha_pep, beta_pep, preds, truths = predict_result(model, eval_dataloader)
        essemble_results.append(preds)

    preds_final = np.mean(essemble_results, axis=0)

    # 输出预测结果
    print(alpha_pep)
    df_pred_result = pd.DataFrame({'alpha_pep':[]})
    df_pred_result['alpha_pep'] = alpha_pep
    df_pred_result['beta_pep'] = beta_pep
    df_pred_result['Truths'] = truths
    df_pred_result['Preds'] = preds_final
    df_pred_result.to_csv(r"M:\FB2\AG FLiu\Users\ZZH\MyProjects\XLProteomicDB\IntensityRatioPredictor\Paper_graph\pred_result_percentage0.3_21to30_240607new.csv")

    test_RMSE = RMSE(truths, preds_final)
    test_MAE = MAE(truths, preds_final)
    test_CORR = CORR(np.array(truths), np.array(preds_final).flatten())
    test_SPEARMAN = SPEARMAN(np.array(truths), np.array(preds_final).flatten())
    
    results = [test_RMSE, test_MAE, test_CORR, test_SPEARMAN]
    print(f"FINE_TUNE_PERCENTAGE:{FINE_TUNE_PERCENTAGE},FINE_TUNE_EPOCH: {FINE_TUNE_EPOCH}, FINE_TUNE_LEARN_RATE: {FINE_TUNE_LEARN_RATE}")
    print(results)
    # Append results to the output file
    append_results_to_file(EXPERIMENT_RECORD_FILE, FINE_TUNE_PERCENTAGE, FINE_TUNE_EPOCH, FINE_TUNE_LEARN_RATE, results)