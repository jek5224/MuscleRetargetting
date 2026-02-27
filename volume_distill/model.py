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


class SIRENLayer(nn.Module):
    """Linear layer with sin activation and SIREN-style initialization."""
    def __init__(self, in_dim, out_dim, is_first=False, omega_0=30.0):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1.0 / in_dim, 1.0 / in_dim)
            else:
                bound = math.sqrt(6.0 / in_dim) / omega_0
                self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.zero_()

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class SIRENResBlock(nn.Module):
    """Residual block with SIREN activations."""
    def __init__(self, dim, omega_0=30.0):
        super().__init__()
        self.layer1 = SIRENLayer(dim, dim, omega_0=omega_0)
        self.layer2 = SIRENLayer(dim, dim, omega_0=omega_0)

    def forward(self, x):
        return x + self.layer2(self.layer1(x))


class SharedEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=512, latent_dim=512,
                 num_res_blocks=3, num_freqs=6, omega_0=30.0):
        super().__init__()
        self.pe = PositionalEncoding(input_dim, num_freqs)
        self.input_proj = SIRENLayer(self.pe.output_dim, hidden_dim,
                                     is_first=True, omega_0=omega_0)
        self.res_blocks = nn.Sequential(
            *[SIRENResBlock(hidden_dim, omega_0=omega_0)
              for _ in range(num_res_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        bound = math.sqrt(6.0 / hidden_dim) / omega_0
        with torch.no_grad():
            self.output_proj.weight.uniform_(-bound, bound)
            self.output_proj.bias.zero_()

    def forward(self, x):
        h = self.input_proj(self.pe(x))
        h = self.res_blocks(h)
        return self.output_proj(h)


class MuscleDecoder(nn.Module):
    """Per-muscle decoder with SIREN residual blocks."""
    def __init__(self, latent_dim=512, hidden_dim=512, num_res_blocks=2,
                 output_dim=None, omega_0=30.0):
        super().__init__()
        assert output_dim is not None
        self.input_proj = SIRENLayer(latent_dim, hidden_dim,
                                     is_first=True, omega_0=omega_0)
        self.res_blocks = nn.Sequential(
            *[SIRENResBlock(hidden_dim, omega_0=omega_0)
              for _ in range(num_res_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            self.output_proj.weight.uniform_(-1e-4, 1e-4)
            self.output_proj.bias.zero_()

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
        latent_dim = self.encoder.output_proj.out_features
        decoder_input = latent_dim + pe_dim
        self.decoders = nn.ModuleDict({
            name: MuscleDecoder(latent_dim=decoder_input, output_dim=v_count * 3)
            for name, v_count in muscle_vertex_counts.items()
        })
        self.muscle_vertex_counts = muscle_vertex_counts

    def forward(self, x):
        pe_x = self.encoder.pe(x)
        latent = self.encoder(x)  # PE → SIREN layers → output_proj
        combined = torch.cat([latent, pe_x], dim=-1)
        return {name: decoder(combined) for name, decoder in self.decoders.items()}
