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
    def __init__(self, muscle_vertex_counts, input_dim=4,
                 hidden_dim=512, num_encoder_res=3, num_decoder_res=2):
        """
        Args:
            muscle_vertex_counts: dict mapping muscle_name -> num_vertices
            input_dim: number of input DOFs (default 4: 3 hip + 1 knee)
            hidden_dim: hidden/latent dimension for encoder and decoders
            num_encoder_res: number of residual blocks in shared encoder
            num_decoder_res: number of residual blocks in each decoder
        """
        super().__init__()
        self.encoder = SharedEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=hidden_dim,
            num_res_blocks=num_encoder_res,
        )
        # Skip connection: decoders receive encoder output + PE features
        pe_dim = self.encoder.pe.output_dim
        decoder_input = hidden_dim + pe_dim
        self.decoders = nn.ModuleDict({
            name: MuscleDecoder(
                latent_dim=decoder_input, hidden_dim=hidden_dim,
                num_res_blocks=num_decoder_res, output_dim=v_count * 3,
            )
            for name, v_count in muscle_vertex_counts.items()
        })
        self.muscle_vertex_counts = muscle_vertex_counts

    def forward(self, x):
        pe_x = self.encoder.pe(x)
        latent = self.encoder(x)
        combined = torch.cat([latent, pe_x], dim=-1)
        return {name: decoder(combined) for name, decoder in self.decoders.items()}


class DistillNetV2(nn.Module):
    """V2: single decoder conditioned on muscle embedding, PCA output, linear baseline + residual."""
    def __init__(self, num_muscles, muscle_name_to_idx, input_dim=20,
                 hidden_dim=768, num_encoder_res=5, num_decoder_res=3,
                 embed_dim=64, pca_k=64, num_freqs=6):
        super().__init__()
        self.num_muscles = num_muscles
        self.muscle_name_to_idx = muscle_name_to_idx
        self.input_dim = input_dim
        self.pca_k = pca_k
        self.embed_dim = embed_dim

        # Shared encoder
        self.encoder = SharedEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=hidden_dim,
            num_res_blocks=num_encoder_res, num_freqs=num_freqs,
        )
        pe_dim = self.encoder.pe.output_dim

        # Muscle identity embedding
        self.muscle_embed = nn.Embedding(num_muscles, embed_dim)

        # Single decoder: input = latent + PE + embed
        decoder_input = hidden_dim + pe_dim + embed_dim
        self.decoder = MuscleDecoder(
            latent_dim=decoder_input, hidden_dim=hidden_dim,
            num_res_blocks=num_decoder_res, output_dim=pca_k,
        )

        # Linear baseline: input DOFs + embed → PCA coeffs
        self.linear_baseline = nn.Linear(input_dim + embed_dim, pca_k)
        self.linear_baseline.apply(weights_init)

    def forward(self, x, muscle_indices=None):
        """Forward pass.

        Args:
            x: (B, input_dim) DOF window
            muscle_indices: optional (M,) tensor of muscle indices to predict.
                If None, predicts all muscles.

        Returns:
            dict {muscle_idx: (B, pca_k)} PCA coefficients per muscle
        """
        if muscle_indices is None:
            muscle_indices = torch.arange(self.num_muscles, device=x.device)

        B = x.shape[0]

        # Encode once: shared across all muscles
        pe_x = self.encoder.pe(x)       # (B, pe_dim)
        latent = self.encoder(x)         # (B, hidden_dim)

        # Loop over muscles to avoid B*M memory explosion
        result = {}
        for m_idx in muscle_indices:
            embed = self.muscle_embed(m_idx)             # (embed_dim,)
            embed_exp = embed.unsqueeze(0).expand(B, -1)  # (B, embed_dim)

            # Decoder: residual path
            decoder_in = torch.cat([latent, pe_x, embed_exp], dim=-1)  # (B, hidden+pe+embed)
            residual = self.decoder(decoder_in)  # (B, pca_k)

            # Linear baseline
            linear_in = torch.cat([x, embed_exp], dim=-1)  # (B, input+embed)
            baseline = self.linear_baseline(linear_in)  # (B, pca_k)

            result[m_idx.item()] = baseline + residual

        return result
