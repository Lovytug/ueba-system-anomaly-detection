import numpy as np
import torch
from model.classes.func_reconstructor import reconstruction_error


class Evaluator:

    def __init__(
        self,
        model,
        device="cpu"
    ):

        self.model = model.to(device)
        self.device = device


    @torch.no_grad()
    def reconstruct(self, x):
        """
        возв-ет реконструкцию входного батча
        """

        self.model.eval()

        x = x.to(self.device)

        return self.model(x)


    @torch.no_grad()
    def feature_errors(self, x):
        """
        возв-ет ошибку реконструкции для каждого признака
        """

        x = x.to(self.device)

        x_hat = self.reconstruct(x)

        return (x - x_hat).pow(2)


    @torch.no_grad()
    def reconstruction_error(self, x):
        """
        возв-ет среднюю ошибку реконструкции для каждого объекта
        """

        return self.feature_errors(x).mean(dim=1)


    @torch.no_grad()
    def explain(self, x):
        """
        возв-ет полную информацию для объяснения аномалии
        """

        x = x.to(self.device)

        x_hat = self.reconstruct(x)

        feature_error = (x - x_hat).pow(2)

        total_error = feature_error.mean(dim=1)

        return {
            "original": x.cpu().numpy(),
            "reconstruction": x_hat.cpu().numpy(),
            "feature_error": feature_error.cpu().numpy(),
            "total_error": total_error.cpu().numpy()
        }



    @torch.no_grad()
    def reconstruction_errors(self, loader):
        """
        воз-ет ошибки реконструкции для всего DataLoader
        """

        errors = []

        for batch in loader:

            err = self.reconstruction_error(batch)

            errors.extend(err.cpu().numpy())

        return np.asarray(errors)


    @torch.no_grad()
    def reconstruct_loader(self, loader):
        """
        воз-ет исходные объекты и их реконструкции для всего DataLoader
        """

        original = []
        reconstructed = []

        for batch in loader:

            x = batch.to(self.device)

            x_hat = self.reconstruct(x)

            original.append(x.cpu())
            reconstructed.append(x_hat.cpu())

        return (
            torch.cat(original).numpy(),
            torch.cat(reconstructed).numpy()
        )


    @torch.no_grad()
    def explain_loader(self, loader):

        original = []
        reconstruction = []
        feature_error = []
        total_error = []

        for batch in loader:

            exp = self.explain(batch)

            original.append(exp["original"])
            reconstruction.append(exp["reconstruction"])
            feature_error.append(exp["feature_error"])
            total_error.append(exp["total_error"])

        return {
            "original": np.concatenate(original),
            "reconstruction": np.concatenate(reconstruction),
            "feature_error": np.concatenate(feature_error),
            "total_error": np.concatenate(total_error),
        }