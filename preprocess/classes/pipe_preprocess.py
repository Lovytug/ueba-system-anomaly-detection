from preprocess.classes.artifact_detector import TrainingArtifactDetector
from preprocess.classes.cleaner import DataCleaner
from preprocess.classes.features_enginear import FeatureEngineer
from preprocess.classes.scaler_manager import ScalerManager


class PreprocessingPipeline:

    def __init__(self):

        self.cleaner = DataCleaner()

        self.artifact_detector = TrainingArtifactDetector()

        self.feature_transformer = FeatureEngineer()

        self.scaler = ScalerManager()


    def fit_transform(self, df):

        df = self.cleaner.fit_transform(df)

        self.artifact_detector.fit(df)
        df = self.artifact_detector.remove(df)

        df = self.feature_transformer.fit_transform(df)

        df = self.scaler.fit_transform(df)

        return df


    def transform(self, df):

        df = self.cleaner.transform(df)

        df = self.feature_transformer.transform(df)

        df = self.scaler.transform(df)

        return df


    def get_feature_names(self):

        return self.feature_transformer.get_feature_names()