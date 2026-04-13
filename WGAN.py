import os
import math
import random
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ============================================================
# Paths / IO helpers
# ============================================================
def safe_stem(path: str) -> str:
    """Turn a filepath into a safe filename stem for saving artifacts."""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    return stem.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")


def ensure_dir(path: str) -> None:
    """Create directory if not exists."""
    os.makedirs(path, exist_ok=True)


def append_line(filepath: str, line: str) -> None:
    """Append one line to a text file."""
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


@dataclass
class RunPaths:
    """Standard folder layout for a GitHub-friendly experiment run."""
    root: str = "runs"
    models_dirname: str = "models"
    plots_dirname: str = "plots"
    outputs_dirname: str = "outputs"
    logs_dirname: str = "logs"

    def __post_init__(self):
        ensure_dir(self.root)
        ensure_dir(self.models_dir)
        ensure_dir(self.plots_dir)
        ensure_dir(self.outputs_dir)
        ensure_dir(self.logs_dir)

    @property
    def models_dir(self) -> str:
        return os.path.join(self.root, self.models_dirname)

    @property
    def plots_dir(self) -> str:
        return os.path.join(self.root, self.plots_dirname)

    @property
    def outputs_dir(self) -> str:
        return os.path.join(self.root, self.outputs_dirname)

    @property
    def logs_dir(self) -> str:
        return os.path.join(self.root, self.logs_dirname)


# ============================================================
# Device
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Model components
# ============================================================
class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding.
    Expected input shape: [T, B, E]
    """
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, E]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :]


def compute_delta_t(data_m: torch.Tensor) -> torch.Tensor:
    """
    Compute time-lag matrix δ based on missing mask.
    Expects data_m with shape [B, T, D], where:
        1 = observed, 0 = missing
    Returns:
        time_lag with same shape [B, T, D]
    """
    time_lag = torch.ones_like(data_m).to(data_m.device)
    for t in range(1, data_m.shape[1]):
        time_lag[:, t] = (time_lag[:, t - 1] + 1) * (1 - data_m[:, t]) + data_m[:, t]
    return time_lag


class TransformerEncoderBlockD(nn.Module):
    """Vanilla Transformer encoder block for the discriminator (no time-decay)."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [T, B, E]
        attn_output, _ = self.attn(x, x, x, key_padding_mask=mask)
        out1 = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.norm2(out1 + self.dropout2(ffn_output))
        return out2


class TimeDecayMultiheadAttention(nn.Module):
    """
    Custom multi-head attention followed by *multiplication* with a time-decay factor.
    Your original logic:
      attn_output = out_proj(attn_output) * time_decay
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        time_decay: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # query/key/value: [T, B, E]
        # time_decay:      [T, B, E]
        seq_len, batch_size, _ = query.size()

        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        # -> [B, H, T, Hd]
        Q = Q.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        K = K.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        V = V.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # If you ever introduce an attention mask, ensure it is broadcastable to [B,H,T,T]
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # [B,H,T,Hd]

        # back to [T,B,E]
        attn_output = (
            attn_output.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, self.embed_dim)
        )
        attn_output = self.out_proj(attn_output)

        # --- time decay (your original design choice) ---
        attn_output = attn_output * time_decay
        return attn_output


class EncoderLayer(nn.Module):
    """One time-decay Transformer encoder layer (MHA + FFN + residual norms)."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn = TimeDecayMultiheadAttention(embed_dim, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_decay: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_output = self.attn(x, x, x, time_decay, mask)
        out1 = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(out1)
        out2 = self.norm2(out1 + self.dropout2(ffn_output))
        return out2


class TransformerEncoderBlock(nn.Module):
    """Stack of time-decay encoder layers."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, time_decay: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, time_decay, mask)
        return x


class Generator(nn.Module):
    """
    Two-encoder generator:
    - Encoder1 (per sensor): learns sensor-specific reconstruction and fills missing entries
    - Encoder2 (global): models inter-sensor dependencies after concatenation
    """
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        num_heads: int,
        ffdim_gen: int,      # <-- new: generator FFN dim
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensors = feature_dim

        # Sensor-wise embedding and inverse projection
        self.fc1_x = nn.Linear(1, embed_dim)
        self.fc1_td = nn.Linear(1, embed_dim)           # time-decay embedding for sensor stream
        self.fc_inverse_x = nn.Linear(embed_dim, 1)

        self.pos_encoder = PositionalEncoding(embed_dim)

        # Encoder1: per-sensor (shared weights across sensors)
        self.encoder1 = TransformerEncoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ffdim_gen,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Combine all sensors into one embedding stream for encoder2
        self.fc_x_prime_combined = nn.Linear(feature_dim, embed_dim)

        # Time-decay embedding for global stream
        self.time_decay_embed = nn.Linear(feature_dim, embed_dim)

        # Encoder2: global
        self.encoder2 = TransformerEncoderBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ffdim_gen,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Output back to D sensors
        self.fc_output = nn.Linear(embed_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, m: torch.Tensor, time_decay: torch.Tensor) -> torch.Tensor:
        """
        x, m, time_decay: [B, T, D]
          m = 1 observed, 0 missing
        returns: [B, T, D] in [0,1] due to sigmoid
        """
        batch_size, seq_len, d = x.shape
        assert d == self.num_sensors, "feature_dim mismatch"

        # -----------------------------
        # Encoder1: per-sensor fill
        # -----------------------------
        x_prime_list = []
        for s in range(self.num_sensors):
            x_s = x[:, :, s].unsqueeze(-1)          # [B,T,1]
            m_s = m[:, :, s].unsqueeze(-1)          # [B,T,1]
            td_s = time_decay[:, :, s].unsqueeze(-1)  # [B,T,1]

            # Value embedding
            x_embed = self.fc1_x(x_s)               # [B,T,E]

            # Time-decay embedding (your exp(-relu(.)) pattern)
            td_embed = self.fc1_td(td_s)            # [B,T,E]
            td_embed = torch.relu(td_embed)
            td_embed = torch.exp(-td_embed)

            # Transformer expects [T,B,E]
            x_embed = self.pos_encoder(x_embed.permute(1, 0, 2))
            td_embed = td_embed.permute(1, 0, 2)

            # Sensor-specific representation
            rep = self.encoder1(x_embed, td_embed, mask=None)     # [T,B,E]

            # Project back to original sensor space
            x_rec = self.fc_inverse_x(rep).permute(1, 0, 2)       # [B,T,1]
            x_rec = self.sigmoid(x_rec)

            # Fill only missing
            x_prime = m_s * x_s + (1 - m_s) * x_rec
            x_prime_list.append(x_prime)

        x_prime_combined = torch.cat(x_prime_list, dim=-1)        # [B,T,D]

        # -----------------------------
        # Encoder2: global refinement
        # -----------------------------
        x2 = self.fc_x_prime_combined(x_prime_combined)           # [B,T,E]
        x2 = self.pos_encoder(x2.permute(1, 0, 2))                # [T,B,E]

        td2 = self.time_decay_embed(time_decay)                   # [B,T,E]
        td2 = torch.relu(td2)
        td2 = torch.exp(-td2)
        td2 = td2.permute(1, 0, 2)                                # [T,B,E]

        out = self.encoder2(x2, td2, mask=None)                   # [T,B,E]
        out = out.permute(1, 0, 2)                                # [B,T,E]

        out = self.fc_output(out)                                 # [B,T,D]
        out = self.sigmoid(out)
        return out


class Discriminator(nn.Module):
    """
    Transformer-based discriminator.
    Input: (x, m) -> embed both -> concat -> Transformer -> per-feature score in [0,1].
    """
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        num_heads: int,
        ffdim_disc: int,     # <-- new: discriminator FFN dim
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1_x = nn.Linear(feature_dim, embed_dim)
        self.fc1_m = nn.Linear(feature_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        # concat(x_embed, m_embed) => 2E
        self.block = TransformerEncoderBlockD(
            embed_dim=embed_dim * 2,
            num_heads=num_heads,
            ff_dim=ffdim_disc,
            dropout=dropout,
        )
        self.fc2 = nn.Linear(embed_dim * 2, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        # x,m: [B,T,D]
        x_embed = self.fc1_x(x)                                  # [B,T,E]
        x_embed = self.pos_encoder(x_embed.permute(1, 0, 2))     # [T,B,E]

        m_embed = self.fc1_m(m)                                  # [B,T,E]
        m_embed = self.pos_encoder(m_embed.permute(1, 0, 2))     # [T,B,E]

        x_with_mask = torch.cat([x_embed, m_embed], dim=2)       # [T,B,2E]
        d_out = self.block(x_with_mask)                          # [T,B,2E]
        d_out = d_out.permute(1, 0, 2)                           # [B,T,2E]
        out = self.fc2(d_out)                                    # [B,T,D]
        return self.sigmoid(out)


# ============================================================
# Data utilities
# ============================================================
def uniform_sampler(low: float, high: float, rows: int, cols: int) -> np.ndarray:
    """Sample uniform noise U(low, high) for generator corruption."""
    return np.random.uniform(low, high, size=[rows, cols])


def create_sequences(data: np.ndarray, sequence_length: int) -> torch.Tensor:
    """
    Convert [N, D] -> [N-seq+1, seq, D] sliding windows.
    """
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i : i + sequence_length])
    return torch.tensor(np.array(sequences, dtype=np.float32))


def split_data(data: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Simple chronological split (train first, test last)."""
    total_size = len(data)
    ts = int(total_size * test_size)
    return data[: total_size - ts], data[total_size - ts :]


# ============================================================
# Metrics (missing-only)
# ============================================================
def rmse_loss(ori_data: np.ndarray, imputed_data: np.ndarray, data_m: np.ndarray) -> float:
    """Missing-only RMSE (computed only where mask==0)."""
    ori = ori_data.reshape(-1, ori_data.shape[2])
    imp = imputed_data.reshape(-1, imputed_data.shape[2])
    m = data_m.reshape(-1, data_m.shape[2])

    num = np.sum(((1 - m) * ori - (1 - m) * imp) ** 2)
    den = np.sum(1 - m)
    if den == 0:
        return float("nan")
    return float(np.sqrt(num / float(den)))


def mae_loss(ori_data: np.ndarray, imputed_data: np.ndarray, data_m: np.ndarray) -> float:
    """Missing-only MAE (computed only where mask==0)."""
    ori = ori_data.reshape(-1, ori_data.shape[2])
    imp = imputed_data.reshape(-1, imputed_data.shape[2])
    m = data_m.reshape(-1, data_m.shape[2])

    num = np.sum(np.abs((1 - m) * ori - (1 - m) * imp))
    den = np.sum(1 - m)
    if den == 0:
        return float("nan")
    return float(num / float(den))


# ============================================================
# Losses (WGAN-style)
# ============================================================
def discriminative_loss(D: nn.Module, fake_x: torch.Tensor, real_x: torch.Tensor, real_m: torch.Tensor, fake_m: torch.Tensor) -> torch.Tensor:
    """
    WGAN critic loss: E[D(fake)] - E[D(real)]
    NOTE: In your implementation, D returns a tensor [B,T,D], so we mean() it.
    """
    d_fake = D(fake_x, fake_m)
    d_real = D(real_x, real_m)
    return torch.mean(d_fake) - torch.mean(d_real)


def generator_loss(D: nn.Module, fake_x: torch.Tensor) -> torch.Tensor:
    """WGAN generator loss: -E[D(fake)]."""
    fake_m = torch.ones_like(fake_x, device=fake_x.device)
    d_fake = D(fake_x, fake_m)
    return -torch.mean(d_fake)


def masked_reconstruction_loss(original: torch.Tensor, generated: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Reconstruction loss on observed entries only.
    mask = 1 on observed.
    """
    # Prevent divide-by-zero (should not happen, but safe)
    denom = torch.mean(mask) + 1e-8
    return torch.mean((mask * original - mask * generated) ** 2) / denom


# ============================================================
# Training / Testing
# ============================================================
def _safe_choice(n: int, k: int) -> np.ndarray:
    """Sample k indices from [0..n-1]; with replacement if n < k (avoids crash)."""
    replace = n < k
    return np.random.choice(n, k, replace=replace)


def train_wgan(
    train_data: np.ndarray,
    params: dict,
    model_path: str,
) -> Tuple[List[float], List[float], List[int]]:
    """
    Train WGAN on sequences with NaNs indicating missing values.
    Saves best generator checkpoint (by avg G loss) to model_path.
    """
    seq_length = params["seq_length"]
    embed_dim = params["embed_dim"]
    num_heads = params["num_heads"]
    ffdim_gen = params["ffdim_gen"]
    ffdim_disc = params["ffdim_disc"]
    lambda_ = params["lambda"]
    num_layers = params["num_layers"]
    max_iterations = params["iterations"]
    batch_size = params["batch_size"]
    n_critic = params.get("n_critic", 5)
    weight_clipping = params.get("weight_clipping", 0.1)
    eval_interval = params.get("eval_interval", 100)

    # Mask: 1 observed, 0 missing
    train_m_np = 1 - np.isnan(train_data)
    train_filled = np.nan_to_num(train_data, nan=0.0)

    x_train = torch.tensor(train_filled, dtype=torch.float32, device=DEVICE)
    m_train = torch.tensor(train_m_np, dtype=torch.float32, device=DEVICE)

    feature_dim = x_train.shape[2]
    total_samples = x_train.shape[0]

    G = Generator(
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffdim_gen=ffdim_gen,
        num_layers=num_layers,
        dropout=0.1,
    ).to(DEVICE)

    D = Discriminator(
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffdim_disc=ffdim_disc,
        dropout=0.1,
    ).to(DEVICE)

    G_opt = optim.Adam(G.parameters(), lr=params["learning_rate"])
    D_opt = optim.Adam(D.parameters(), lr=params["learning_rate"])

    # Schedulers track loss trends
    G_sched = lr_scheduler.ReduceLROnPlateau(G_opt, mode="min", factor=0.5, patience=1, verbose=True, min_lr=1e-5)
    D_sched = lr_scheduler.ReduceLROnPlateau(D_opt, mode="min", factor=0.5, patience=1, verbose=True, min_lr=1e-5)

    best_G = float("inf")
    patience = params.get("patience", 3)
    patience_ctr = 0

    G_losses, D_losses, iters = [], [], []
    total_G, total_D = 0.0, 0.0

    iteration = 0
    while iteration < max_iterations:
        # -----------------------------
        # Critic updates
        # -----------------------------
        for _ in range(n_critic):
            idx = _safe_choice(total_samples, batch_size)
            x_real = x_train[idx]     # [B,T,D]
            m = m_train[idx]          # [B,T,D]

            # Noise U(0,1) only on missing entries
            z = uniform_sampler(0, 1, x_real.shape[0], x_real.shape[1] * x_real.shape[2])
            z = torch.tensor(z, dtype=torch.float32, device=DEVICE).reshape_as(x_real)
            x_corrupt = m * x_real + (1 - m) * z

            delta_t = compute_delta_t(m)

            # Generator output
            g_out = G(x_corrupt, m, delta_t)

            # For critic input, keep observed real and fill missing with g_out
            g_for_disc = m * x_real + (1 - m) * g_out

            fake_m = torch.ones_like(m, device=DEVICE)

            d_loss = discriminative_loss(D, g_for_disc, x_real, m, fake_m)

            D_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            D_opt.step()

            # Weight clipping (classic WGAN)
            for p in D.parameters():
                p.data.clamp_(-weight_clipping, weight_clipping)

            total_D += float(d_loss.item())

        # -----------------------------
        # Generator update
        # -----------------------------
        idx = _safe_choice(total_samples, batch_size)
        x_real = x_train[idx]
        m = m_train[idx]

        z = uniform_sampler(0, 1, x_real.shape[0], x_real.shape[1] * x_real.shape[2])
        z = torch.tensor(z, dtype=torch.float32, device=DEVICE).reshape_as(x_real)
        x_corrupt = m * x_real + (1 - m) * z

        delta_t = compute_delta_t(m)
        g_out = G(x_corrupt, m, delta_t)

        g_for_disc = m * x_real + (1 - m) * g_out

        Lr = masked_reconstruction_loss(x_real, g_out, m)
        Lg = generator_loss(D, g_for_disc)
        g_loss = lambda_ * Lr + Lg

        G_opt.zero_grad(set_to_none=True)
        g_loss.backward()
        G_opt.step()

        total_G += float(g_loss.item())
        iteration += 1

        # -----------------------------
        # Periodic logging / checkpoint
        # -----------------------------
        if iteration % eval_interval == 0:
            avg_G = total_G / eval_interval
            avg_D = total_D / (eval_interval * n_critic)

            G_losses.append(avg_G)
            D_losses.append(avg_D)
            iters.append(iteration)

            total_G, total_D = 0.0, 0.0

            print(
                f"Iter {iteration}/{max_iterations} | "
                f"G_loss {avg_G:.8f} | D_loss {avg_D:.8f} | "
                f"lrG {G_opt.param_groups[0]['lr']:.6f} | lrD {D_opt.param_groups[0]['lr']:.6f}"
            )

            G_sched.step(avg_G)
            D_sched.step(avg_D)

            # Save best generator by avg_G
            if avg_G < best_G:
                best_G = avg_G
                patience_ctr = 0
                torch.save(G.state_dict(), model_path)
            else:
                patience_ctr += 1
                print(f"patience_counter: {patience_ctr}/{patience}")
                if patience_ctr >= patience:
                    print(f"Early stopping at iteration {iteration}")
                    break

    print("Training completed.")
    return G_losses, D_losses, iters


@torch.no_grad()
def test_generator(
    test_data: np.ndarray,
    original_test_data: np.ndarray,
    params: dict,
    model_path: str,
) -> Tuple[np.ndarray, float, float]:
    """Load best generator and compute missing-only RMSE/MAE on test split."""
    embed_dim = params["embed_dim"]
    num_heads = params["num_heads"]
    ffdim_gen = params["ffdim_gen"]
    num_layers = params["num_layers"]

    test_m_np = 1 - np.isnan(test_data)
    test_filled = np.nan_to_num(test_data, nan=0.0)

    x_test = torch.tensor(test_filled, dtype=torch.float32, device=DEVICE)
    m_test = torch.tensor(test_m_np, dtype=torch.float32, device=DEVICE)

    feature_dim = x_test.shape[2]

    G = Generator(
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffdim_gen=ffdim_gen,
        num_layers=num_layers,
        dropout=0.1,
    ).to(DEVICE)

    G.load_state_dict(torch.load(model_path, map_location=DEVICE))
    G.eval()

    z = uniform_sampler(0, 1, x_test.shape[0], x_test.shape[1] * x_test.shape[2])
    z = torch.tensor(z, dtype=torch.float32, device=DEVICE).reshape_as(x_test)

    x_corrupt = m_test * x_test + (1 - m_test) * z
    delta_t = compute_delta_t(m_test)

    pred = G(x_corrupt, m_test, delta_t).cpu().numpy()

    imputed = m_test.cpu().numpy() * x_test.cpu().numpy() + (1 - m_test.cpu().numpy()) * pred

    rmse = rmse_loss(original_test_data, imputed, m_test.cpu().numpy())
    mae = mae_loss(original_test_data, imputed, m_test.cpu().numpy())
    return imputed, rmse, mae


# ============================================================
# Experiment runner
# ============================================================
def run_one_missing_pattern(
    paths: RunPaths,
    params: dict,
    data_normalized: np.ndarray,
    missing_file: str,
    num_runs: int,
    results_file: str,
) -> None:
    """Train/test for a single missingness pattern file."""
    tag = safe_stem(missing_file)
    append_line(results_file, "")
    append_line(results_file, f"Processing missing file: {missing_file}")
    print(f"\nProcessing missing index file: {missing_file}")

    missing_mask = np.load(missing_file)
    data_with_missing = data_normalized.copy()
    data_with_missing[missing_mask == 1] = np.nan

    # Create sliding-window sequences
    seq_len = params["seq_length"]
    data_sequences = create_sequences(data_with_missing, seq_len).cpu().numpy()
    original_sequences = create_sequences(data_normalized, seq_len).cpu().numpy()

    train_data, test_data = split_data(data_sequences)
    _, original_test_sequences = split_data(original_sequences)

    generator_path = os.path.join(paths.models_dir, f"best_generator_{tag}_noiseU01.pth")

    rmse_list: List[float] = []
    mae_list: List[float] = []

    for run_idx in range(num_runs):
        append_line(results_file, f"Run {run_idx+1}/{num_runs} for file {missing_file}")
        print(f"Run {run_idx+1}/{num_runs} for file {missing_file}")

        # -------- TRAIN --------
        G_losses, D_losses, iters = train_wgan(
            train_data=train_data,
            params=params,
            model_path=generator_path,
        )

        np.save(os.path.join(paths.outputs_dir, f"{tag}_run{run_idx+1}_iters.npy"), np.array(iters))
        np.save(os.path.join(paths.outputs_dir, f"{tag}_run{run_idx+1}_G_losses.npy"), np.array(G_losses))
        np.save(os.path.join(paths.outputs_dir, f"{tag}_run{run_idx+1}_D_losses.npy"), np.array(D_losses))

        # Plot generator loss curve
        plt.figure()
        plt.plot(iters, G_losses, label="Training G Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title(f"Training G Loss - {os.path.basename(missing_file)} - Run {run_idx+1}")
        plt.legend()
        plot_path = os.path.join(paths.plots_dir, f"Loss_{tag}_Run_{run_idx+1}_noiseU01.png")
        plt.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close()

        # -------- TEST --------
        imputed, test_rmse, test_mae = test_generator(
            test_data=test_data,
            original_test_data=original_test_sequences,
            params=params,
            model_path=generator_path,
        )

        out_path = os.path.join(paths.outputs_dir, f"imputed_{tag}_run{run_idx+1}.npy")
        np.save(out_path, imputed)

        append_line(results_file, f"Test RMSE: {test_rmse:.6f}")
        append_line(results_file, f"Test MAE:  {test_mae:.6f}")
        print(f"Test RMSE: {test_rmse:.4f} | Test MAE: {test_mae:.4f}")

        rmse_list.append(float(test_rmse))
        mae_list.append(float(test_mae))

    avg_rmse = float(sum(rmse_list) / len(rmse_list))
    avg_mae = float(sum(mae_list) / len(mae_list))

    append_line(results_file, f"Average RMSE: {avg_rmse:.6f}")
    append_line(results_file, f"Average MAE:  {avg_mae:.6f}")
    print(f"Average RMSE for {missing_file}: {avg_rmse:.4f} | Average MAE: {avg_mae:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WGAN Two-Encoder Transformer Imputer (Noise U(0,1))")

    parser.add_argument("--sensor_csv", type=str, default="sensor.csv", help="Path to sensor CSV (no header assumed).")
    parser.add_argument("--runs_root", type=str, default="runs_WGAN_two_encoder_noiseU01", help="Root output folder.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--num_runs", type=int, default=1, help="Runs per missingness pattern.")

    # Core hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seq_length", type=int, default=100)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=1)

    # NEW: separate FF dims
    parser.add_argument("--ffdim_gen", type=int, default=512, help="Generator FFN dimension.")
    parser.add_argument("--ffdim_disc", type=int, default=512, help="Discriminator FFN dimension.")

    # WGAN specifics
    parser.add_argument("--lambda_rec", type=float, default=50, help="Reconstruction weight lambda.")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--weight_clipping", type=float, default=0.02)
    parser.add_argument("--eval_interval", type=int, default=100)

    # Missingness files (you can also pass your own list in a txt later if you want)
    parser.add_argument("--missing_files_mode", type=str, default="default",
                        choices=["default"], help="Currently only 'default' list is embedded.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    print("Running on", DEVICE)

    # --- Create run folders ---
    paths = RunPaths(root=args.runs_root)

    # --- Load and normalize data ---
    dataset = pd.read_csv(args.sensor_csv).values
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(dataset)

    # --- WGAN parameter dict ---
    params = {
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "learning_rate": args.learning_rate,
        "seq_length": args.seq_length,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,

        "ffdim_gen": args.ffdim_gen,
        "ffdim_disc": args.ffdim_disc,


        "lambda": args.lambda_rec,
        "patience": args.patience,
        "n_critic": args.n_critic,
        "weight_clipping": args.weight_clipping,
        "eval_interval": args.eval_interval,
    }

    # --- Missingness files list (same as yours, just kept together) ---
    missing_files = [
        # MCAR
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_10_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_20_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_30_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_40_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_50_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_60_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_70_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_80_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_90_percent_mcar.npy",

        # Temporal Only
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_10_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_20_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_30_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_40_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_50_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_60_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_70_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_80_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_90_percent_temporal_only.npy",

        # Spatial Only
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_10_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_20_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_30_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_40_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_50_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_60_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_70_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_80_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_90_percent_spatial_only.npy",

        # Temporal and Spatial
        "SKAB Data Missingness Patterns/Temporal and Spatial/skab_missing_indices_10_percent_temporal_and_spatial.npy",
        "SKAB Data Missingness Patterns/Temporal and Spatial/skab_missing_indices_20_percent_temporal_and_spatial.npy",
        "SKAB Data Missingness Patterns/Temporal and Spatial/skab_missing_indices_30_percent_temporal_and_spatial.npy",
        "SKAB Data Missingness Patterns/Temporal and Spatial/skab_missing_indices_40_percent_temporal_and_spatial.npy",
        "SKAB Data Missingness Patterns/Temporal and Spatial/skab_missing_indices_50_percent_temporal_and_spatial.npy",
        "SKAB Data Missingness Patterns/Temporal and Spatial/skab_missing_indices_60_percent_temporal_and_spatial.npy",
        "SKAB Data Missingness Patterns/Temporal and Spatial/skab_missing_indices_70_percent_temporal_and_spatial.npy",
        "SKAB Data Missingness Patterns/Temporal and Spatial/skab_missing_indices_80_percent_temporal_and_spatial.npy",
        "SKAB Data Missingness Patterns/Temporal and Spatial/skab_missing_indices_90_percent_temporal_and_spatial.npy",
    ]

    # --- Logging ---
    results_file = os.path.join(paths.logs_dir, "WGAN_results.txt")
    append_line(results_file, "==================== NEW RUN ====================")
    append_line(results_file, f"Device: {DEVICE}")
    append_line(results_file, f"Seed: {args.seed}")
    append_line(results_file, f"Params: {params}")

    # --- Run across missingness patterns ---
    for mf in missing_files:
        run_one_missing_pattern(
            paths=paths,
            params=params,
            data_normalized=data_normalized,
            missing_file=mf,
            num_runs=args.num_runs,
            results_file=results_file,
        )


if __name__ == "__main__":
    main()