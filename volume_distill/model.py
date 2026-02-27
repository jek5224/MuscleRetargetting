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
        # x: (..., input_dim)
        xf = x.unsqueeze(-1) * self.freqs  # broadcast
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
    def __init__(self, input_dim=4, hidden_dim=512, latent_dim=256,
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
        # Output projection uses standard init (no sin activation after it)
        bound = math.sqrt(6.0 / hidden_dim) / omega_0
        with torch.no_grad():
            self.output_proj.weight.uniform_(-bound, bound)
            self.output_proj.bias.zero_()

    def forward(self, x):
        h = self.input_proj(self.pe(x))
        h = self.res_blocks(h)
        return self.output_proj(h)


class VertexDecoder(nn.Module):
    """Shared per-vertex decoder: (latent, rest_pos) → displacement (3).

    A single decoder shared across all muscles and vertices.
    Takes the concatenation of the DOF latent and vertex rest position,
    and predicts the 3D displacement for that vertex.
    """
    def __init__(self, latent_dim=256, pos_dim=3, embed_dim=16, hidden_dim=128,
                 num_res_blocks=2, omega_0=30.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.input_proj = SIRENLayer(latent_dim + pos_dim + embed_dim, hidden_dim,
                                     is_first=True, omega_0=omega_0)
        self.res_blocks = nn.Sequential(
            *[SIRENResBlock(hidden_dim, omega_0=omega_0)
              for _ in range(num_res_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, 3)
        # Small init on output so initial predictions are near zero
        with torch.no_grad():
            self.output_proj.weight.uniform_(-1e-4, 1e-4)
            self.output_proj.bias.zero_()

    def forward(self, latent, rest_pos, muscle_embed):
        """
        Args:
            latent: (B, latent_dim) or (B, V, latent_dim)
            rest_pos: (B, V, 3)
            muscle_embed: (embed_dim,) — expanded to (B, V, embed_dim)
        Returns:
            displacement: (B, V, 3)
        """
        B, V = rest_pos.shape[:2]
        if latent.dim() == 2:
            latent = latent.unsqueeze(1).expand(-1, V, -1)
        me = muscle_embed.unsqueeze(0).unsqueeze(0).expand(B, V, -1)
        h = self.input_proj(torch.cat([latent, rest_pos, me], dim=-1))
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
        latent_dim = self.encoder.output_proj.out_features
        embed_dim = 16
        self.decoder = VertexDecoder(latent_dim=latent_dim, embed_dim=embed_dim)
        self.muscle_names = list(muscle_vertex_counts.keys())
        self.muscle_vertex_counts = muscle_vertex_counts
        self.muscle_embeddings = nn.Embedding(len(self.muscle_names), embed_dim)
        self._muscle_name_to_idx = {name: i for i, name in enumerate(self.muscle_names)}

    def forward(self, x, rest_positions):
        """
        Args:
            x: (B, input_dim) DOF values
            rest_positions: dict of {muscle_name: (V, 3)} tensors
        Returns:
            dict of {muscle_name: (B, V*3)} predicted displacements (flat)
        """
        latent = self.encoder(x)  # (B, latent_dim)
        results = {}
        for name, v_count in self.muscle_vertex_counts.items():
            rp = rest_positions[name].unsqueeze(0).expand(x.shape[0], -1, -1)
            idx = self._muscle_name_to_idx[name]
            me = self.muscle_embeddings.weight[idx]  # (embed_dim,)
            disp = self.decoder(latent, rp, me)  # (B, V, 3)
            results[name] = disp.reshape(x.shape[0], -1)  # (B, V*3)
        return results
