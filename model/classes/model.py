import torch
import torch.nn as nn


class NetworkBuilder:

    @staticmethod
    def build_encoder(
        input_dim,
        hidden_dims,
        activation,
        dropout: float = 0.0,
        batch_norm: bool = False
    ):

        layers = []

        in_features = input_dim

        for out_features in hidden_dims:

            layers.append(nn.Linear(in_features, out_features))

            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))

            layers.append(activation())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_features = out_features

        return nn.Sequential(*layers)


    @staticmethod
    def build_decoder(
        input_dim,
        hidden_dims,
        activation,
        dropout: float = 0.0,
        batch_norm: bool = False
    ):

        layers = []

        decoder_dims = hidden_dims[::-1]

        in_features = decoder_dims[0]

        for out_features in decoder_dims[1:]:

            layers.append(nn.Linear(in_features, out_features))

            if batch_norm:
                layers.append(nn.BatchNorm1d(out_features))

            layers.append(activation())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            in_features = out_features

        layers.append(nn.Linear(in_features, input_dim))

        return nn.Sequential(*layers)



class AutoEncoder(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        activation=nn.ReLU,
        dropout: float = 0.0,
        batch_norm: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.encoder = NetworkBuilder.build_encoder(
            input_dim,
            hidden_dims,
            activation,
            dropout=dropout,
            batch_norm=batch_norm
        )

        self.decoder =  NetworkBuilder.build_decoder(
            input_dim,
            hidden_dims,
            activation,
            dropout=dropout,
            batch_norm=batch_norm
        )


    def encode(self, x):

        return self.encoder(x)

    def decode(self, z):

        return self.decoder(z)


    def forward(self, x):

        latent = self.encode(x)

        reconstruction = self.decode(latent)

        return reconstruction