import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder


class FeatureTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self


    def transform(self, X):

        X = X.copy()

        X = self._log_network(X)
        X = self._log_disk(X)

        return X

    
    def _to_numeric(
        self,
        X: pd.DataFrame,
        cols,
    ):
        for col in cols:
            if col in X.columns:
                X[col] = pd.to_numeric(
                    X[col],
                    errors="coerce",
                ).fillna(0.0).astype(np.float64)

        return X


    def _log_network(self, X):

        cols = [
            "bytes_sent_delta",
            "bytes_recv_delta",
        ]

        X[cols] = np.log1p( X[cols].to_numpy(dtype=np.float64) )

        return X


    def _log_disk(self, X):

        cols = [
            "disk_read_bytes_delta",
            "disk_write_bytes_delta",
            "disk_read_count_delta",
            "disk_write_count_delta",
        ]

        X[cols] = np.log1p( X[cols].to_numpy(dtype=np.float64) )

        return X



class FeatureCreator(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.weekday_encoder = OneHotEncoder(
            categories=[range(7)],
            sparse_output=False,
            handle_unknown="ignore"
        )

    def fit(self, X, y=None):

        self.weekday_encoder.fit(X[["weekday"]])

        return self


    def transform(self, X):

        X = X.copy()

        X = self._encode_hour(X)
        X = self._encode_weekday(X)
        X = self._create_process_flags(X)

        return X
    

    def _encode_hour(self, X: pd.DataFrame):

        if "hour" not in X.columns:
            return X

        hour = pd.to_numeric(
            X["hour"],
            errors="coerce",
        ).fillna(0.0).to_numpy(dtype=np.float64)
        
        X["hour_sin"] = np.sin(
            2 * np.pi * hour  / 24
        )

        X["hour_cos"] = np.cos(
            2 * np.pi * hour  / 24
        )

        X.drop(columns="hour", inplace=True)

        return X


    def _encode_weekday(self, X: pd.DataFrame):

        if "weekday" not in X.columns:
            return X
        
        X["weekday"] = pd.to_numeric(
            X["weekday"],
            errors="coerce",
        ).fillna(0).astype(int)
        
        encoded = self.weekday_encoder.transform(
            X[["weekday"]]
        )

        encoded = pd.DataFrame(
            encoded,
            columns=self.weekday_encoder.get_feature_names_out(
                ["weekday"]
            ),
            index=X.index
        )

        X = pd.concat(
            [
                X.drop(columns="weekday"),
                encoded
            ],
            axis=1
        )

        return X
    

    def _create_process_flags(self, X: pd.DataFrame):

        if "new_processes" in X.columns:

            X["has_new_processes"] = (
                X["new_processes"] > 0
            ).astype(np.uint8)

        if "new_unique_processes" in X.columns:

            X["has_new_unique_processes"] = (
                X["new_unique_processes"] > 0
            ).astype(np.uint8)

        return X



class FeatureSelector(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.columns_to_drop = [
            "timestamp",
            "disk_usage_percent",
            "tcp_total",
            "packets_sent_delta",
            "packets_recv_delta",
        ]


    def fit(self, X, y=None):

        return self


    def transform(self, X):

        return X.drop(columns=self.columns_to_drop)



class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):

        self.creator = FeatureCreator()
        self.transformer = FeatureTransformer()
        self.selector = FeatureSelector()

        self.feature_names_ = None


    def fit(self, X, y=None):

        self.creator.fit(X)

        transformed = self.creator.transform(X)
        transformed = self.transformer.transform(transformed)
        transformed = self.selector.transform(transformed)

        self.feature_names_ = transformed.columns.tolist()

        return self


    def transform(self, X):

        X = X.copy()

        X = self.creator.transform(X)
        X = self.transformer.transform(X)
        X = self.selector.transform(X)

        return X


    def get_feature_names(self):

        if self.feature_names_ is None:
            raise RuntimeError(
                "Класс - инженеринг признков не был обучен еще"
            )
        return self.feature_names_