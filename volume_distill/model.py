import math
import torch
import torch.nn as nn


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
        xf = x.unsqueeze(-1) * self.freqs
        return torch.cat([x, xf.sin().flatten(-2), xf.cos().flatten(-2)], dim=-1)


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.zero_()


class ResBlock(nn.Module):
    """Residual block with LeakyReLU."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.LeakyReLU(0.2)
        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)

    def forward(self, x):
        return x + self.act(self.fc2(self.act(self.fc1(x))))


class SharedEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=512, latent_dim=512,
                 num_res_blocks=3, num_freqs=6):
        super().__init__()
        self.pe = PositionalEncoding(input_dim, num_freqs)
        self.input_proj = nn.Sequential(
            nn.Linear(self.pe.output_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2),
        )
        self.input_proj.apply(weights_init)
        self.output_proj.apply(weights_init)

    def forward(self, x):
        h = self.input_proj(self.pe(x))
        h = self.res_blocks(h)
        return self.output_proj(h)


class MuscleDecoder(nn.Module):
    """Per-muscle decoder with residual blocks."""
    def __init__(self, latent_dim=512, hidden_dim=512, num_res_blocks=2,
                 output_dim=None):
        super().__init__()
        assert output_dim is not None
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim) for _ in range(num_res_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.input_proj.apply(weights_init)
        self.output_proj.apply(weights_init)

    def forward(self, x):
        h = self.input_proj(x)
        h = self.res_blocks(h)
        return self.output_proj(h)


class DistillNet(nn.Module):
    def __init__(self, muscle_vertex_counts, input_dim=4):
        """
        Args:
            muscle_vertex_counts: dict mapping muscle_name -> num_vertices
            input_dim: number of input DOFs (default 4: 3 hip + 1 knee)
        """
        super().__init__()
        self.encoder = SharedEncoder(input_dim=input_dim)
        # Skip connection: decoders receive encoder output + PE features
        pe_dim = self.encoder.pe.output_dim
        latent_dim = 512
        decoder_input = latent_dim + pe_dim
        self.decoders = nn.ModuleDict({
            name: MuscleDecoder(latent_dim=decoder_input, output_dim=v_count * 3)
            for name, v_count in muscle_vertex_counts.items()
        })
        self.muscle_vertex_counts = muscle_vertex_counts

    def forward(self, x):
        pe_x = self.encoder.pe(x)
        latent = self.encoder(x)
        combined = torch.cat([latent, pe_x], dim=-1)
        return {name: decoder(combined) for name, decoder in self.decoders.items()}
