#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------
# @File  : metrics_utils.py
# @Time  : 2024/03/13 14:21:22
# @Author: Zhang, Zehong 
# @Email : zhang@fmp-berlin.de
# @Desc  : None
# -----------------------------------
import numpy as np
import sklearn.metrics as m
from scipy.stats import pearsonr, spearmanr
from numba import njit
from sklearn.linear_model import LinearRegression
# ================================
# ======= Regression Metrics ======
# ================================

@njit
def c_index(y_true, y_pred):
    """Predictive accuracy of survival models or regression models in the context of ranking."""
    summ = 0
    pair = 0

    for i in range(1, len(y_true)):
        for j in range(0, i):
            pair += 1
            if y_true[i] > y_true[j]:
                summ += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            elif y_true[i] < y_true[j]:
                summ += 1 * (y_pred[i] < y_pred[j]) + 0.5 * (y_pred[i] == y_pred[j])
            else:
                pair -= 1

    return summ / pair if pair != 0 else 0


def RMSE(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(m.mean_squared_error(y_true, y_pred))


def MAE(y_true, y_pred):
    """Mean Absolute Error."""
    return m.mean_absolute_error(y_true, y_pred)


def CORR(y_true, y_pred):
    """Pearson correlation coefficient."""
    return pearsonr(y_true, y_pred)[0]


def SPEARMAN(y_true, y_pred):
    """Spearman correlation coefficient."""
    return spearmanr(y_true, y_pred)[0]


def SD(y_true, y_pred):
    """Standard deviation of residuals from linear regression."""
    y_pred = y_pred.reshape((-1, 1))
    lr = LinearRegression().fit(y_pred, y_true)
    y_ = lr.predict(y_pred)
    return np.sqrt(np.square(y_true - y_).sum() / (len(y_pred) - 1))


# ================================
#  Binary Classification Metrics =
# ================================

def accuracy(y_true, y_pred):
    """Accuracy: The proportion of correctly classified instances."""
    return m.accuracy_score(y_true, y_pred)


def precision(y_true, y_pred):
    """Precision: Proportion of true positives among predicted positives."""
    return m.precision_score(y_true, y_pred, zero_division=0)


def recall(y_true, y_pred):
    """Recall: Proportion of true positives among actual positives."""
    return m.recall_score(y_true, y_pred)


def f1_score(y_true, y_pred):
    """F1 Score: Harmonic mean of precision and recall."""
    return m.f1_score(y_true, y_pred)


def specificity(y_true, y_pred):
    """Specificity: Proportion of true negatives among actual negatives."""
    tn, fp, fn, tp = m.confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def roc_auc(y_true, y_pred_proba):
    """ROC-AUC: Area under the ROC curve."""
    return m.roc_auc_score(y_true, y_pred_proba)


def log_loss_metric(y_true, y_pred_proba):
    """Log Loss: Performance measure for classification models."""
    return m.log_loss(y_true, y_pred_proba)


def mcc(y_true, y_pred):
    """Matthews Correlation Coefficient (MCC): Balanced measure of prediction quality."""
    return m.matthews_corrcoef(y_true, y_pred)


def cohen_kappa(y_true, y_pred):
    """Cohen's Kappa: Agreement measure between two raters."""
    return m.cohen_kappa_score(y_true, y_pred)



def balanced_accuracy(y_true, y_pred):
    """Balanced Accuracy: Average recall for each class, useful for imbalanced datasets."""
    return m.balanced_accuracy_score(y_true, y_pred)


def g_mean(y_true, y_pred):
    """G-Mean: Geometric mean of recall and specificity."""
    tn, fp, fn, tp = m.confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return np.sqrt(sensitivity * specificity)


# ================================
# = Multi Classification Metrics =
# ================================

def precision_multi(y_true, y_pred):
    """Precision: Proportion of true positives among predicted positives."""
    return m.precision_score(y_true, y_pred, average='weighted', zero_division=0)

def recall_multi(y_true, y_pred):
    """Recall: Proportion of true positives among actual positives."""
    return m.recall_score(y_true, y_pred, average='weighted')

def f1_score_multi(y_true, y_pred):
    """F1 Score: Harmonic mean of precision and recall."""
    return m.f1_score(y_true, y_pred, average='weighted')

def specificity_multi(y_true, y_pred):
    """Specificity: Proportion of true negatives among actual negatives."""
    cm = m.confusion_matrix(y_true, y_pred)
    # For multiclass, specificity is computed for each class
    specificity_per_class = []
    for i in range(cm.shape[0]):
        tn = np.sum(cm) - np.sum(cm[:, i]) - np.sum(cm[i, :]) + cm[i, i]
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity_per_class.append(tn / (tn + fp) if tn + fp != 0 else 0)
    return np.mean(specificity_per_class)

def roc_auc_multi(y_true, y_pred_proba):
    """ROC-AUC: Area under the ROC curve."""
    return m.roc_auc_score(y_true, y_pred_proba, average='weighted', multi_class='ovr')

print("Metrics loading done.")

if __name__=="__main__":
    pass