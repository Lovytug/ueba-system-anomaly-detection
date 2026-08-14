import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class DataCleaner(BaseEstimator, TransformerMixin):

    def __init__(self, len_window: int = 5, max_gap: int = 3):
        super().__init__()

        self.window = len_window
        self.gap = max_gap


    def fit(self, X, y=None):

        return self


    def transform(self, X: pd.DataFrame):

        X = X.copy()
        X = self._fix_negativ_delta(X)
        X = self._remove_nan(X)

        return X


    def _fix_negativ_delta(self, data: pd.DataFrame):

        cols = ['bytes_sent_delta', 'bytes_recv_delta', 
            'packets_sent_delta', 'packets_recv_delta']
        
        data[cols] = data[cols].clip(lower=0)

        return data


    def _remove_nan(self, data: pd.DataFrame):

        data = data.copy()

        rows_to_drop = set()

        numeric_cols = data.select_dtypes(include=np.number).columns

        for col in numeric_cols:

            mask = data[col].isna()

            if not mask.any():
                continue

            idx = np.where(mask)[0]

            groups = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

            for group in groups:

                if len(group) >= self.gap:
                    rows_to_drop.update(group)
                    continue

                left = max(0, group[0] - self.window)
                right = min(len(data), group[-1] + self.window + 1)

                neighbours = data.iloc[left:right][col].dropna()

                if len(neighbours) == 0:
                    continue

                median = neighbours.median()

                data.loc[data.index[group], col] = median

        if rows_to_drop:
            data = data.drop(index=data.index[list(rows_to_drop)])

        data.reset_index(drop=True, inplace=True)

        return data

