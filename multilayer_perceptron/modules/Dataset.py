import numpy as np

class Dataset:
    def __init__(self, x, y, x_valid=None, y_valid=None):
        self.x = x
        self.y = y
        self.x_valid = x_valid
        self.y_valid = y_valid

    def batchiterator(self, batch_size):
        indices = np.arange(self.x.shape[0])
        np.random.shuffle(indices)
        for start_idx in range(0, len(indices) - batch_size + 1, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            yield self.x[batch_indices], self.y[batch_indices]