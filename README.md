# AIRPred: A Deep Learning Model for Predicting Peptide Intensity Ratios in Cross-Linking Mass Spectrometry

## Overview
**AIRPred** is a deep learning-based tool designed to enhance Cross-Linking Mass Spectrometry (XL-MS) analysis by predicting intensity ratios of cross-linked peptide pairs. By leveraging Convolutional Neural Networks (CNNs) and attention mechanisms, AIRPred improves Cross-Linked Spectrum Match (CSM) identification, outperforming traditional scoring methods that primarily rely on mass-to-charge ratio (m/z) comparisons.

---

## Features
- Predicts intensity ratios (log2-transformed) for cross-linked peptide pairs.
- Employs CNN blocks to analyze peptide fragmentation patterns.
- Utilizes an attention mechanism to capture peptide interactions.
- Provides insights into amino acid contributions using SHapley Additive exPlanations (SHAP).
- Validated on external datasets, demonstrating high accuracy and reliability.

---

## Requirements
### Hardware:
- **GPU**: NVIDIA GeForce RTX 4090 (or equivalent for optimal performance).
- **CPU**: Intel(R) Core(TM) i9-14900K (or equivalent).

### Software:
- **Python**: Version 3.11.5
- **Machine Learning Libraries**:
  - PyTorch
  - Scikit-learn
  - XGBoost
  - LightGBM
- **Data Processing and Visualization**:
  - NumPy
  - pandas
  - SciPy
  - Biopython
  - Matplotlib
  - Seaborn
- **SHAP for Model Interpretation**:
  - SHAP library ([GitHub](https://github.com/slundberg/shap))

### Additional Tools:
- **Mass Spectrometry Data Processing**:
  - Scout software (version 1.5.0)

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/AIRPred.git
   cd AIRPred
   ```

2. Set up a virtual environment:
   ```bash
   python3 -m venv airpred_env
   source airpred_env/bin/activate  # On Windows: airpred_env\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Verify installation:
   ```bash
   python -m unittest discover tests
   ```

---

## Usage
### 1. Preparing the Dataset
- **Input**: Mass spectrometry data processed with Scout software.
- **Format**: Tables containing peptide sequences, modifications, and intensity ratios.
- **Dataset Splitting**:
  - Training and validation sets: Use the Max dataset (20,961 entries).
  - Testing set: Use the MxR_HEK_21to30 dataset (975 entries).
  - Exclude low-quality data (CSMs with fewer than three occurrences).

### 2. Training the Model
To train AIRPred on your dataset:
```bash
python train.py --data_path /path/to/dataset --output_dir /path/to/output
```
- Adjust hyperparameters and model configurations in `config.json`.

### 3. Making Predictions
To predict intensity ratios for a new dataset:
```bash
python predict.py --model_path /path/to/trained_model --data_path /path/to/new_dataset --output_path /path/to/results
```

### 4. Model Interpretation
Generate SHAP explanations for the model:
```bash
python interpret.py --model_path /path/to/trained_model --data_path /path/to/analysis_set
```
- The output includes SHAP values highlighting amino acid contributions to intensity ratios.

---

## Example Workflow
1. Preprocess your raw MS data with Scout.
2. Train the AIRPred model:
   ```bash
   python train.py --data_path ./data/max_dataset.csv --output_dir ./models
   ```
3. Evaluate the model on the test dataset:
   ```bash
   python evaluate.py --model_path ./models/best_model.pth --data_path ./data/test_dataset.csv
   ```
4. Interpret model predictions using SHAP:
   ```bash
   python interpret.py --model_path ./models/best_model.pth --data_path ./data/validation_set.csv
   ```

---

## Results
- **Performance Metrics**:
  - Improved true CSM identification by >80%.
  - Robust intensity ratio predictions across diverse datasets.
- **Model Insights**:
  - SHAP analysis reveals key amino acid properties influencing intensity ratios.

---

## Dataset Description
1. **Training and Validation**: Max dataset (20,961 entries).
   - Log2-transformed intensity ratios.
   - Average intensity ratio for multiple occurrences of the same CSM.
2. **Testing**: MxR_HEK_21to30 dataset (975 entries).
   - Includes species and MS parameter variations (e.g., HCD percentage).
3. **Additional Analytical Dataset**: Ying_mito datasets (three replicates).
   - Excluded from training due to distinct fragmentation types.

---

## Citation
If you use AIRPred in your research, please cite:
```
Zhang, Z., Ruwolt, M., Zhu, Y., Jiang, P.L., Lima, D.B., & Liu, F. (2024). AIRPred: A Deep Learning Model for Predicting Peptide Intensity Ratios in Cross-Linking Mass Spectrometry. Leibniz-Forschungsinstitut für Molekulare Pharmakologie (FMP), Berlin, Germany.
```

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact
For questions or collaboration inquiries, please contact:
- Zehong Zhang: zhang.zehong@fmp-berlin.de
- Fan Liu: fan.liu@fmp-berlin.de

