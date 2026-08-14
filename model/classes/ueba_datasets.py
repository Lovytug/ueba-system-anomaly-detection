import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class UEBADataset(Dataset):

    def __init__(
        self,
        data: pd.DataFrame | np.ndarray,
        dtype=torch.float32,
    ):

        if isinstance(data, pd.DataFrame):
            data = data.to_numpy(dtype=np.float32)

        elif isinstance(data, np.ndarray):
            data = data.astype(np.float32)

        else:
            raise TypeError(
                f"Датасет имеет не подходящий тип - pandas or np.ndarray. Пришло - {type(data)}"
            )

        self.data = torch.as_tensor(data, dtype=dtype)


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        x = self.data[index]

        return x