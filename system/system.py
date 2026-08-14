import joblib
import torch
import pandas as pd

from preprocess.classes.pipe_preprocess import PreprocessingPipeline

from model.classes.builder_dataset import DatasetBuilder
from model.classes.ueba_datasets import UEBADataset
from model.classes.model import AutoEncoder
from model.classes.trainer import Trainer
from model.classes.evaluator import Evaluator

from threshold.classes.compute_params import RiskParameterEstimator
from threshold.classes.detector import ExponentialRiskDetector


class UEBATrainer:

    def __init__(
        self,
        hidden_dims=(64, 16),
        batch_size=256,
        train_ratio=0.8,
        lr=1e-3,
        device="cpu",
    ):

        self.device = device

        self.pipeline = PreprocessingPipeline()

        self.dataset_builder = DatasetBuilder(
            pipeline=self.pipeline,
            dataset_cls=UEBADataset,
            batch_size=batch_size,
            train_ratio=train_ratio,
        )

        self.input_dim = None

        self.hidden_dims = hidden_dims
        self.lr = lr

        self.model = None
        self.detector_params = None


    def fit(self, df, epochs=50, ):
        
        data = self.dataset_builder.build(df)
        
        self.input_dim = len(self.pipeline.get_feature_names())

        self.model = AutoEncoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )

        trainer = Trainer(
            self.model,
            optimizer,
            device=self.device,
        )

        trainer.fit(
            train_loader=data["train_loader"],
            val_loader=data["val_loader"],
            epochs=epochs,
        )

        evaluator = Evaluator(
            self.model,
            self.device,
        )

        estimator = RiskParameterEstimator(
            evaluator
        )

        self.detector_params = estimator.fit(
            data["val_loader"]
        )

        return self


    def save(
        self,
        path,
    ):

        joblib.dump(
            {
                "pipeline": self.pipeline,
                "input_dim": self.input_dim,
                "hidden_dims": self.hidden_dims,
                "model": self.model.state_dict(),
                "detector_params": self.detector_params,
            },
            path,
        )



class UEBADetector:

    def __init__(
        self,
        pipeline,
        model,
        detector,
        device="cpu",
        batch_size=256,
    ):

        self.pipeline = pipeline

        self.model = model.to(device)

        self.evaluator = Evaluator(
            self.model,
            device,
        )

        self.detector: ExponentialRiskDetector = detector

        self.dataset_builder = DatasetBuilder(
            pipeline=self.pipeline,
            dataset_cls=UEBADataset,
            batch_size=batch_size,
        )


    def detect(
        self,
        sample,
    ):
        """
        для объектов
        """

        if isinstance(sample, pd.Series):
            sample = sample.to_frame().T

        loader = self.dataset_builder.transform(
            sample,
            batch_size=1,
        )

        error = self.evaluator.reconstruction_errors(loader)[0]

        result = self.detector.update(error)

        explanation = self.evaluator.explain_loader(loader)

        return {
            "error": error,
            "risk": result["risk"],
            "alarm": result["alarm"],
            "explanation": explanation,
            "feature_names": self.pipeline.get_feature_names(),
        }


    def detect_batch(
        self,
        df,
        batch_size=256,
    ):
        """
        для батчей-датафремов
        """

        loader = self.dataset_builder.transform(
            df,
            batch_size=batch_size,
        )

        errors = self.evaluator.reconstruction_errors(loader)

        result = self.detector.predict(errors)

        explanation = self.evaluator.explain_loader(loader)

        return {
            "errors": errors,
            "risk": result["risk"],
            "alarm": result["alarm"],
            "explanation": explanation,
            "feature_names": self.pipeline.get_feature_names(),
        }


    @classmethod
    def load(
        cls,
        path,
        device="cpu",
    ):

        state = joblib.load(path)

        pipeline = state["pipeline"]

        model = AutoEncoder(
            input_dim=state["input_dim"],
            hidden_dims=state["hidden_dims"],
        )

        model.load_state_dict(
            state["model"]
        )

        detector = ExponentialRiskDetector(
            **state["detector_params"]
        )

        return cls(
            pipeline=pipeline,
            model=model,
            detector=detector,
            device=device,
        )