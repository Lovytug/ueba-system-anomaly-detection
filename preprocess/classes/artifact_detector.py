import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor


class TrainingArtifactDetector:

    def __init__(
        self,
        n_neighbors: int = 10,
        contamination: float = 0.001,
        metric: str = "euclidean",
    ):

        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            metric=metric,
            novelty=True,
        )

        self._n_removed: int = None
        self.feature_columns = None


    def fit(self, X: pd.DataFrame):

        self.feature_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        self.model.fit(X[self.feature_columns])

        return self


    def predict(self, X: pd.DataFrame):
        """
        Возращает предсказание как
        1 = норм объект
        -1 = артефакт
        """
        return self.model.predict(X[self.feature_columns])


    def get_mask(self, X: pd.DataFrame):

        return self.predict(X) == 1


    def remove(self, X: pd.DataFrame):

        mask = self.get_mask(X)

        removed = (~mask).sum()
        self._n_removed = removed

        return X.loc[mask].reset_index(drop=True)