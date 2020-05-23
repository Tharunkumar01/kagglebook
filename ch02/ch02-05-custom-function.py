import numpy as np
import pandas as pd


# -----------------------------------
# Optimizing MAE by approximating metric with a custom objective function
# -----------------------------------

# Fair function
def fair(preds, dtrain):
    x = preds - dtrain.get_labels()  # Get residual
    c = 1.0  # Parameter of fair function
    den = abs(x) + c  # Calculate denominator of gradient formula
    grad = c * x / den  # Gradient
    hess = c * c / den ** 2  # Second derivative
    return grad, hess


# Pseudo-Huber function
def psuedo_huber(preds, dtrain):
    d = preds - dtrain.get_labels()  # Get residual
    delta = 1.0  # Parameter of Pseudo-Huber function
    scale = 1 + (d / delta) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt  # Gradient
    hess = 1 / scale / scale_sqrt  # Second derivative
    return grad, hess
