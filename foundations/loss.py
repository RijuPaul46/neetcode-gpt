import numpy as np
from numpy.typing import NDArray
from math import log  as ln

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon=1e-7
        n=len(y_true)
        loss=0
        for i in range (n):
            yi=y_true[i]
            pi=y_pred[i]+epsilon
            loss+=(yi*ln(pi) + (1-yi)*ln(1-pi))
        return round(-loss/n,4)
        pass

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n_samples,n_classes=np.shape(y_true)
        loss=0
        es=1e-7
        for i in range(n_samples):
            for j in range(n_classes):
                y=y_true[i][j]+es
                loss+=(-y*ln(y_pred[i][j]))
        avg_loss=round(loss/n_samples  , 4)
        return avg_loss
        pass
