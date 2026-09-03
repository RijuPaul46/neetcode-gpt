import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        n,m=np.shape(X)
        predictions=[[] for _ in range(n)]
        for i in range(n):
            prediction=np.dot(X[i],weights)
            predictions[i]=round(prediction,5)
        return predictions
        pass

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        mse=np.mean((model_prediction-ground_truth)**2)
        return round(mse,5)
        pass
