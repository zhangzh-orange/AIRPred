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

if __name__=="__main__":
    pass