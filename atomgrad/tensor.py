import numpy as np

class Tensor:
    def __init__(self, data, dtype='float32'):
        self.data = np.array(data, dtype=dtype)

    def __repr__(self):
       return f'Tensor: tensor({self.data})'