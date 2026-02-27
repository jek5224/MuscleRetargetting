import math
import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (NeRF-style) for scalar DOF inputs.

    For each input dimension, appends sin(2^k * pi * x) and cos(2^k * pi * x)
    for k = 0..num_freqs-1.  Output dim = input_dim * (1 + 2 * num_freqs).
    """
    def __init__(self, input_dim, num_freqs=6):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        freqs = torch.tensor([2 ** k * math.pi for k in range(num_freqs)])
        self.register_buffer("freqs", freqs)  # (num_freqs,)

    @property
    def output_dim(self):
        return self.input_dim * (1 + 2 * self.num_freqs)

    def forward(self, x):
        # x: (..., input_dim)
        # Outer product with frequencies: (..., input_dim, num_freqs)
        xf = x.unsqueeze(-1) * self.freqs  # broadcast
        # Concatenate raw + sin + cos along last dim
        return torch.cat([x, xf.sin().flatten(-2), xf.cos().flatten(-2)], dim=-1)


class SharedEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_sizes=(256, 512, 512), latent_dim=512,
                 num_freqs=6):
        super().__init__()
        self.pe = PositionalEncoding(input_dim, num_freqs)
        layers = []
        prev = self.pe.output_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.LeakyReLU(0.2))
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        layers.append(nn.LeakyReLU(0.2))
        self.fc = nn.Sequential(*layers)
        self.fc.apply(weights_init)

    def forward(self, x):
        return self.fc(self.pe(x))


class MuscleDecoder(nn.Module):
    def __init__(self, latent_dim=512, hidden_sizes=(512, 512), output_dim=None):
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
        latent_dim = 512
        self.decoders = nn.ModuleDict({
            name: MuscleDecoder(latent_dim=latent_dim, output_dim=v_count * 3)
            for name, v_count in muscle_vertex_counts.items()
        })
        self.muscle_vertex_counts = muscle_vertex_counts

    def forward(self, x):
        latent = self.encoder(x)
        return {name: decoder(latent) for name, decoder in self.decoders.items()}
