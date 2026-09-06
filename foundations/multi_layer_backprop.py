import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x=np.array(x)
        w1=np.array(W1)
        b1=np.array(b1)
        w2=np.array(W2)
        b2=np.array(b2)
        y_true=np.array(y_true)
        z1=w1 @x +b1
        a1=np.maximum(0,z1)
        # mask=np.zeros(len(a1))
        # for i in range(len(a1)):
        #     if a1[i]>0:
        #         mask[i]=1
        #     else:
        #         mask[i]=0
        
        mask=(a1>0).astype(int)
        z2=w2 @ a1 +b2
        y_hat=z2
        loss=np.mean((y_true-y_hat)**2)
        dz2=2*np.mean(z2-y_true)
        dw2=np.outer(dz2,a1)
        db2=dz2
        da1=dz2*w2[0]
        dz1=da1 * mask
        dw1=np.outer(dz1,x)
        db1=dz1
        return {
        'loss': round(float(loss), 4),
        'dW1': np.round(dw1, 4).tolist(),
        'db1': np.round(db1, 4).tolist(),
        'dW2': np.round(dw2, 4).tolist(),
        'db2': np.round(np.atleast_1d(db2), 4).tolist()
         }
        pass
