import numpy as np


class ExponentialRiskDetector:

    def __init__(
        self,
        T,
        L,
        H,
        alpha,
    ):

        self.T = T
        self.L = L
        self.H = H
        self.alpha = alpha

        self.reset()


    def reset(self):

        self.S = 0.0


    def update(self, error):

        r = max(0.0, error - self.T)

        x = min(np.log1p(r), self.L)

        self.S = self.alpha * self.S + x

        return {
            "risk": self.S,
            "alarm": self.S > self.H
        }


    def predict(self, errors):

        self.reset()

        risk = []
        alarm = []

        for error in errors:

            result = self.update(error)

            risk.append(result["risk"])
            alarm.append(result["alarm"])

        return {
            "risk": np.asarray(risk),
            "alarm": np.asarray(alarm)
        }