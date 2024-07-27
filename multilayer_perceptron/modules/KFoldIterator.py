import numpy as np
from modules.Dataset import Dataset

class KFoldIterator:
    def __init__(self, x, y, k):
        self.k = k
        self.x = x
        self.y = y
        self.size = x.shape[0]
        self.idx = np.arange(self.size)
        np.random.shuffle(self.idx)
        self.current_k = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_k >= self.k:
            raise StopIteration
        test_mask = np.zeros(self.size, dtype=bool)
        test_idx = np.arange(int(self.current_k * self.size / self.k), int((self.current_k + 1) * self.size / self.k))
        test_mask[test_idx] = True
        train_mask = ~test_mask
        self.current_k += 1
        return Dataset(self.x[train_mask], self.y[train_mask]), Dataset(self.x[test_mask], self.y[test_mask])