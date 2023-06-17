import numpy as np
from scipy.io import loadmat

class Dataset():
    def __init__(self, fn_mat: str) -> None:
        """
        Args:
            feature: file mat name, file must have format {'feature': Array[m, n], 'label': Array[1, m]}
        """
        self.fn_mat = fn_mat

        data = loadmat(self.fn_mat, variable_names=['feature', 'label'])
        self.len = data['label'].shape[-1]
        self.std = np.std(data['feature'], axis=0, keepdims=True)
        self.mean = np.mean(data['feature'], axis=0, keepdims=True)
        del data

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx) -> tuple:
        data = loadmat(self.fn_mat, variable_names=['feature', 'label'])
        X = (data['feature'][idx]-self.mean) / self.std
        y = data['label'][0][idx]
        del data
        return X, y

class _BaseIterator:
    def __init__(self, data, idxs) -> None:
        self.data = data
        self.idxs = idxs

    def __next__(self):
        try:
            idxs = next(self.idxs)
            return self.data[idxs]
        except StopIteration:
            raise StopIteration


class DataLoader:
    def __init__(self, dataset: Dataset, batch_size: int=32, shuffle: bool=True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.len = int(np.ceil(len(dataset)/batch_size))

    def __iter__(self):
        lst = np.arange(len(self.dataset), dtype=int)
        if self.shuffle:
            np.random.shuffle(lst)

        lst = np.split(lst[:(self.len-1)*self.batch_size], self.len-1) + [lst[(self.len-1)*self.batch_size:]]

        return _BaseIterator(self.dataset, iter(lst))

    def __len__(self):
        return self.len
