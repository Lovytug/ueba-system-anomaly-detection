import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler


class ScalerManager(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.percent_columns = [
            "cpu_percent",
            "memory_percent",
            "swap_percent",
        ]

        self.robust_columns = [
            "disk_read_bytes_delta",
            "disk_write_bytes_delta",
            "disk_read_count_delta",
            "disk_write_count_delta",

            "bytes_sent_delta",
            "bytes_recv_delta",

            "process_count",
            "tcp_listen",
        ]

        self.robust_scaler = RobustScaler()


    def fit(self, X, y=None):

        X = X.copy()

        cols = [
            c for c in self.robust_columns
            if c in X.columns
        ]

        self.robust_columns_ = cols

        self.robust_scaler.fit(X[self.robust_columns_])

        return self


    def transform(self, X):

        X = X.copy()

        X = self._scale_percent(X)
        X = self._scale_robust(X)

        return X


    def _scale_percent(self, X: pd.DataFrame):

        cols = [
            c for c in self.percent_columns
            if c in X.columns
        ]

        X[cols] = X[cols] / 100.0

        return X


    def _scale_robust(self, X: pd.DataFrame):

        X[self.robust_columns_] = self.robust_scaler.transform(
            X[self.robust_columns_]
        )

        return X


    def get_scaler(self):

        return self.robust_scaler

    def set_scaler(self, scaler):

        self.robust_scaler = scaler