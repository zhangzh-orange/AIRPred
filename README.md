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

---

## Example Workflow
1. Preprocess your raw MS data with Scout.
2. Model Training: Train AIRPred using curated datasets (e.g., Ground-truth dataset).
3. Evaluation: Test model performance on external datasets (e.g., Two-interactome or HEK293T datasets).
4. Interpretation: Use SHAP values to analyze amino acid contributions to intensity ratios.

---

## Results
- **Performance Metrics**:
  - Improved true CSM identification.
  - Robust intensity ratio predictions across diverse datasets.
- **Model Insights**:
  - SHAP analysis reveals key amino acid properties influencing intensity ratios.

---

## Citation
If you use AIRPred in your research, please cite:
```
Zehong Zhang et al., "AIRPred: A Deep Learning Model for Predicting Peptide Intensity Ratios in Cross-Linking Mass Spectrometry", [Journal Name], 2025.
```

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Contact
For questions or collaboration inquiries, please contact:
- Zehong Zhang: zhang.zehong@fmp-berlin.de
- Fan Liu: fan.liu@fmp-berlin.de

