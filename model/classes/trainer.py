import torch
from model.classes.func_reconstructor import reconstruction_error


class History:

    def __init__(self):

        self.train_loss = []
        self.val_loss = []

    def add(
        self,
        train_loss,
        val_loss=None
    ):

        self.train_loss.append(train_loss)

        if val_loss is not None:
            self.val_loss.append(val_loss)
    


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        device="cpu"
    ):

        self.model = model.to(device)

        self.optimizer = optimizer

        self.device = device

        self.history = History()


    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs=50,
        verbose=True
    ):

        for epoch in range(epochs):

            train_loss = self._train_epoch(train_loader)

            if val_loader is not None:

                val_loss = self._validate_epoch(val_loader)

                self.history.add(
                    train_loss,
                    val_loss
                )

                if verbose:
                    print(
                        f"Epoch {epoch+1:03d} | "
                        f"train={train_loss:.6f} | "
                        f"val={val_loss:.6f}"
                    )

            else:

                self.history.add(train_loss)

                if verbose:
                    print(
                        f"Epoch {epoch+1:03d} | "
                        f"train={train_loss:.6f}"
                    )

        return self.history


    def _train_epoch(self, loader):

        self.model.train()

        total_loss = 0

        for batch in loader:

            x = batch.to(self.device)

            self.optimizer.zero_grad()

            x_hat = self.model(x)

            loss = reconstruction_error(x, x_hat).mean()

            loss.backward()

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)


    @torch.no_grad()
    def _validate_epoch(self, loader):

        self.model.eval()

        total_loss = 0

        for batch in loader:

            x = batch.to(self.device)

            x_hat = self.model(x)

            loss = reconstruction_error(x, x_hat).mean()

            total_loss += loss.item()

        return total_loss / len(loader)