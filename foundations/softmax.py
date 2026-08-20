import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        mx=np.max(z)
        total=0
        for elm in z:
            total+=math.exp(elm-mx)
        ans=np.exp(z-mx)/total
        return np.round(ans,4)
        pass
