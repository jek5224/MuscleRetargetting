import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


class SharedEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_sizes=(128, 256, 256), latent_dim=256):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(0.2))
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        layers.append(nn.LeakyReLU(0.2))
        self.fc = nn.Sequential(*layers)
        self.fc.apply(weights_init)

    def forward(self, x):
        return self.fc(x)


class MuscleDecoder(nn.Module):
    def __init__(self, latent_dim=256, hidden_sizes=(256, 256), output_dim=None):
        super().__init__()
        assert output_dim is not None, "output_dim (V*3) must be specified"
        layers = []
        prev = latent_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(0.2))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.fc = nn.Sequential(*layers)
        self.fc.apply(weights_init)

    def forward(self, x):
        return self.fc(x)


class DistillNet(nn.Module):
    def __init__(self, muscle_vertex_counts, input_dim=4):
        """
        Args:
            muscle_vertex_counts: dict mapping muscle_name -> num_vertices
            input_dim: number of input DOFs (default 4: 3 hip + 1 knee)
        """
        super().__init__()
        self.encoder = SharedEncoder(input_dim=input_dim)
        self.decoders = nn.ModuleDict({
            name: MuscleDecoder(output_dim=v_count * 3)
            for name, v_count in muscle_vertex_counts.items()
        })
        self.muscle_vertex_counts = muscle_vertex_counts

    def forward(self, x):
        latent = self.encoder(x)
        return {name: decoder(latent) for name, decoder in self.decoders.items()}
