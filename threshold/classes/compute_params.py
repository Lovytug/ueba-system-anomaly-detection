import numpy as np


class RiskParameterEstimator:

    def __init__(
        self,
        evaluator,
        threshold_percentile=97.5,
        risk_percentile=98,
        alpha=0.8,
        round_step=0.5,
    ):

        self.evaluator = evaluator

        self.threshold_percentile = threshold_percentile
        self.risk_percentile = risk_percentile

        self.alpha = alpha
        self.round_step = round_step


    def fit(self, loader):

        errors = self.evaluator.reconstruction_errors(loader)

        T = np.percentile(
            errors,
            self.threshold_percentile
        )

        x = np.log1p(np.maximum(errors - T, 0))

        L = x.max()

        S = np.zeros_like(x)

        for i in range(1, len(x)):
            S[i] = self.alpha * S[i - 1] + np.minimum(x[i], L)

        H = np.percentile(
            S,
            self.risk_percentile
        )

        H = (
            np.round(H / self.round_step)
            * self.round_step
        )

        return {
            "T": T,
            "L": L,
            "H": H,
            "alpha": self.alpha,
        }