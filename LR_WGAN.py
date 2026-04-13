import os
import math
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# =============================================================================
# Device
# =============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on", device)

# =============================================================================
# Paths / Output structure
# =============================================================================
RUNS_DIR = "runs"
MODELS_DIR = os.path.join(RUNS_DIR, "models")
PLOTS_DIR = os.path.join(RUNS_DIR, "plots")
OUTPUTS_DIR = os.path.join(RUNS_DIR, "outputs")
LOGS_DIR = os.path.join(RUNS_DIR, "logs")
for d in [RUNS_DIR, MODELS_DIR, PLOTS_DIR, OUTPUTS_DIR, LOGS_DIR]:
    os.makedirs(d, exist_ok=True)

RESULTS_FILE = os.path.join(LOGS_DIR, "lr_wgan_results.txt")

# =============================================================================
# Utilities
# =============================================================================
def safe_stem(path: str) -> str:
    """File-safe stem for folder names."""
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    stem = stem.replace(" ", "_").replace("/", "_").replace("\\", "_").replace(":", "_")
    return stem

def create_sequences(data: np.ndarray, sequence_length: int) -> torch.Tensor:
    """
    Creates sliding window sequences:
      input:  (T, F)
      output: (N, L, F) where N = T - L + 1
    """
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    return torch.tensor(np.array(sequences, dtype=np.float32))

def split_data(data: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """Split along first dimension."""
    total_size = len(data)
    ts = int(total_size * test_size)
    return data[:total_size - ts], data[total_size - ts:]

def compute_delta_t(data_m: torch.Tensor) -> torch.Tensor:
    """
    Time lag matrix δ based on mask (1=observed, 0=missing)
    data_m: (B, T, F)
    """
    time_lag = torch.ones_like(data_m).to(data_m.device)
    for i in range(1, data_m.shape[1]):
        time_lag[:, i] = (time_lag[:, i - 1] + 1) * (1 - data_m[:, i]) + data_m[:, i]
    return time_lag

def rmse_loss_missing_only(ori_data: np.ndarray, imputed_data: np.ndarray, obs_mask: np.ndarray) -> float:
    """
    Missing-only RMSE.
    """
    ori_flat = ori_data.reshape(-1, ori_data.shape[2])
    imp_flat = imputed_data.reshape(-1, imputed_data.shape[2])
    m_flat = obs_mask.reshape(-1, obs_mask.shape[2])

    diff2 = ((1 - m_flat) * (ori_flat - imp_flat)) ** 2
    denom = np.sum(1 - m_flat)
    if denom == 0:
        return float("nan")
    return float(np.sqrt(np.sum(diff2) / denom))

def mae_loss_missing_only(ori_data: np.ndarray, imputed_data: np.ndarray, obs_mask: np.ndarray) -> float:
    """
    Missing-only MAE.
    obs_mask: 1 for observed, 0 for missing
    """
    ori_flat = ori_data.reshape(-1, ori_data.shape[2])
    imp_flat = imputed_data.reshape(-1, imputed_data.shape[2])
    m_flat = obs_mask.reshape(-1, obs_mask.shape[2])

    diff = np.abs((1 - m_flat) * (ori_flat - imp_flat))
    denom = np.sum(1 - m_flat)
    if denom == 0:
        return float("nan")
    return float(np.sum(diff) / denom)

# =============================================================================
# Positional Encoding
# =============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, embed_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (T, B, D)
        """
        return x + self.pe[:x.size(0), :]

# =============================================================================
# Transformer blocks
# =============================================================================
class TransformerEncoderBlockD(nn.Module):
    """Simple Transformer block used in discriminator (standard MHA)."""
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
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        """
        x: (T, B, D)
        """
        attn_out, _ = self.attn(x, x, x, key_padding_mask=mask)
        out1 = self.norm1(x + self.drop1(attn_out))
        ffn_out = self.ffn(out1)
        out2 = self.norm2(out1 + self.drop2(ffn_out))
        return out2

class TimeDecayMultiheadAttention(nn.Module):
    """
    Your time-decay MHA: after attention, you multiply output by time_decay
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

    def forward(self, query, key, value, time_decay, mask=None):
        """
        query/key/value: (T, B, D)
        time_decay:      (T, B, D) 
        """
        seq_len, batch_size, _ = query.size()

        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)

        Q = Q.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        K = K.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        V = V.view(seq_len, batch_size, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(seq_len, batch_size, self.embed_dim)
        attn_output = self.out_proj(attn_output)

        attn_output = attn_output * time_decay
        return attn_output

class EncoderLayer(nn.Module):
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
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, time_decay: torch.Tensor, mask=None) -> torch.Tensor:
        attn_out = self.attn(x, x, x, time_decay, mask)
        out1 = self.norm1(x + self.drop1(attn_out))
        ffn_out = self.ffn(out1)
        out2 = self.norm2(out1 + self.drop2(ffn_out))
        return out2

class TransformerEncoderBlock(nn.Module):
    """Stack of EncoderLayer."""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor, time_decay: torch.Tensor, mask=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, time_decay, mask)
        return x

# =============================================================================
# Generator (Two-encoder)
# =============================================================================
class Generator(nn.Module):
    """
    1) Sensor-specific encoder processes each sensor individually (1-dim signal)
    2) Global encoder processes concatenated imputed sensors
    """
    def __init__(
        self,
        seq_length: int,
        feature_dim: int,
        embed_dim: int,
        num_heads: int,
        ffdim_gen: int,
        num_layers: int,
        num_sensors: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensors = num_sensors

        # Per-sensor embedding
        self.fc1_x = nn.Linear(1, embed_dim)
        self.fc1_m = nn.Linear(1, embed_dim)          
        self.fc_inverse_x = nn.Linear(embed_dim, 1)

        self.pos_encoder = PositionalEncoding(embed_dim)

        # Sensor-specific encoder
        self.encoder = TransformerEncoderBlock(embed_dim, num_heads, ffdim_gen, num_layers=1, dropout=dropout)

        # Global encoder
        self.fc_x_prime_combined = nn.Linear(num_sensors, embed_dim)
        self.encoder2 = TransformerEncoderBlock(embed_dim, num_heads, ffdim_gen, num_layers=1, dropout=dropout)

        self.time_decay_embed = nn.Linear(feature_dim, embed_dim)  # embeds full time-lag vector
        self.fc_output = nn.Linear(embed_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, m: torch.Tensor, time_decay: torch.Tensor) -> torch.Tensor:
        """
        x:          (B, T, F)
        m:          (B, T, F)  1=observed, 0=missing
        time_decay: (B, T, F)  lag values
        """
        batch_size, seq_len, num_sensors = x.shape
        X_prime_list = []

        # --- first pass: sensor-wise
        for s in range(self.num_sensors):
            x_s = x[:, :, s].unsqueeze(-1)          # (B, T, 1)
            m_s = m[:, :, s].unsqueeze(-1)          # (B, T, 1)
            td_s = time_decay[:, :, s].unsqueeze(-1)# (B, T, 1)

            x_emb = self.fc1_x(x_s)                 # (B, T, D)
            td_emb = self.fc1_m(td_s)               # (B, T, D)
            td_emb = torch.relu(td_emb)
            td_emb = torch.exp(-td_emb)             # (B, T, D)

            # Transformer expects (T, B, D)
            td_emb = td_emb.permute(1, 0, 2)
            x_emb = self.pos_encoder(x_emb.permute(1, 0, 2))

            rep = self.encoder(x_emb, td_emb, mask=None)  # (T, B, D)

            # Back to sensor value
            x_hat_s = self.fc_inverse_x(rep).permute(1, 0, 2)  # (B, T, 1)
            x_hat_s = self.sigmoid(x_hat_s)

            # Fill missing
            X_prime = m_s * x_s + (1 - m_s) * x_hat_s
            X_prime_list.append(X_prime)

        X_prime_combined = torch.cat(X_prime_list, dim=-1)       # (B, T, F)

        # --- second pass: global
        Xg = self.fc_x_prime_combined(X_prime_combined)          # (B, T, D)
        Xg = self.pos_encoder(Xg.permute(1, 0, 2))               # (T, B, D)

        td_full = self.time_decay_embed(time_decay)              # (B, T, D)
        td_full = torch.relu(td_full)
        td_full = torch.exp(-td_full)
        td_full = td_full.permute(1, 0, 2)                       # (T, B, D)

        out = self.encoder2(Xg, td_full, mask=None)              # (T, B, D)
        out = out.permute(1, 0, 2)                               # (B, T, D)

        out = self.fc_output(out)                                # (B, T, F)
        out = self.sigmoid(out)
        return out

# =============================================================================
# Discriminator
# =============================================================================
class Discriminator(nn.Module):
    """
    Discriminator sees concatenation of:
      - embedded data
      - embedded mask
    then outputs per-feature authenticity scores.
    """
    def __init__(
        self,
        feature_dim: int,
        embed_dim: int,
        num_heads: int,
        ffdim_disc: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fc1_x = nn.Linear(feature_dim, embed_dim)
        self.fc1_m = nn.Linear(feature_dim, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)

        self.block = TransformerEncoderBlockD(embed_dim * 2, num_heads, ffdim_disc, dropout=dropout)
        self.fc2 = nn.Linear(embed_dim * 2, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        m: (B, T, F)
        """
        x_emb = self.fc1_x(x)
        x_emb = self.pos_encoder(x_emb.permute(1, 0, 2))  # (T, B, D)

        m_emb = self.fc1_m(m)
        m_emb = self.pos_encoder(m_emb.permute(1, 0, 2))  # (T, B, D)

        xm = torch.cat([x_emb, m_emb], dim=2)              # (T, B, 2D)
        out = self.block(xm)                               # (T, B, 2D)
        out = out.permute(1, 0, 2)                         # (B, T, 2D)

        out = self.fc2(out)                                # (B, T, F)
        return self.sigmoid(out)

# =============================================================================
# WGAN losses
# =============================================================================
def discriminative_loss(D: nn.Module, fake_data: torch.Tensor, real_data: torch.Tensor,
                        real_mask: torch.Tensor, fake_mask: torch.Tensor) -> torch.Tensor:
    D_fake = D(fake_data, fake_mask)
    D_real = D(real_data, real_mask)
    return torch.mean(D_fake) - torch.mean(D_real)

def generator_adv_loss(D: nn.Module, fake_data: torch.Tensor) -> torch.Tensor:
    fake_mask = torch.ones_like(fake_data).to(fake_data.device)
    D_fake = D(fake_data, fake_mask)
    return -torch.mean(D_fake)

def masked_reconstruction_loss(original: torch.Tensor, generated: torch.Tensor, mask_obs: torch.Tensor) -> torch.Tensor:
    """
    IMPORTANT:
    Your current implementation uses OBSERVED-only reconstruction (mask_obs = 1 for observed).
    That means it reconstructs observed values, not missing values.
    We keep it EXACTLY as you wrote.
    """
    return torch.mean(((mask_obs * original - mask_obs * generated) ** 2)) / torch.mean(mask_obs)

# =============================================================================
# Training / Testing
# =============================================================================
def train_lr_wgan(
    train_data: np.ndarray,
    params: Dict,
    save_path: str,
    current_missing_file: str,
) -> Tuple[List[float], List[float], List[int]]:
    """
    Trains LR-WGAN on train_data sequences with LRMR initialization.
    """
    # Unpack parameters
    seq_length = params["seq_length"]
    embed_dim = params["embed_dim"]
    num_heads = params["num_heads"]
    ffdim_gen = params["ffdim_gen"]
    ffdim_disc = params["ffdim_disc"]
    num_layers = params["num_layers"]
    lambda_ = params["lambda"]
    max_iterations = params["iterations"]
    batch_size = params["batch_size"]
    n_critic = params.get("n_critic", 5)
    weight_clipping = params.get("weight_clipping", 0.02)
    eval_interval = params.get("eval_interval", 100)
    patience = params.get("patience", 5)

    # Build observed mask (1 observed, 0 missing) from NaNs in sequences
    train_obs = 1 - np.isnan(train_data)
    train_filled = np.nan_to_num(train_data, nan=0.0)

    X_train = torch.tensor(train_filled, dtype=torch.float32).to(device)  # (N, T, F)
    M_train = torch.tensor(train_obs, dtype=torch.float32).to(device)     # (N, T, F)

    feature_dim = X_train.shape[2]
    num_sensors = feature_dim

    
    lrmr_path = os.path.join(
        "LRMR outputs",
        os.path.basename(current_missing_file).replace(".npy", "_lrmr_imputed.npy")
    )
    if not os.path.exists(lrmr_path):
        raise FileNotFoundError(f"LRMR file not found: {lrmr_path}")

    lrmr_all = np.load(lrmr_path)                   # should match sequence count
    lrmr_train, _ = split_data(lrmr_all)            # same split rule as sequences
    LR_train = torch.tensor(lrmr_train, dtype=torch.float32).to(device)

    # Models
    G = Generator(
        seq_length=seq_length,
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffdim_gen=ffdim_gen,
        num_layers=num_layers,
        num_sensors=num_sensors,
        dropout=0.1,
    ).to(device)

    D = Discriminator(
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffdim_disc=ffdim_disc,
        dropout=0.1,
    ).to(device)

    G_opt = optim.Adam(G.parameters(), lr=params["learning_rate"])
    D_opt = optim.Adam(D.parameters(), lr=params["learning_rate"])

    # Optional LR schedulers
    G_sched = lr_scheduler.ReduceLROnPlateau(G_opt, mode="min", factor=0.5, patience=1, verbose=True, min_lr=1e-5)
    D_sched = lr_scheduler.ReduceLROnPlateau(D_opt, mode="min", factor=0.5, patience=1, verbose=True, min_lr=1e-5)

    total_samples = X_train.shape[0]

    G_losses, D_losses, iters = [], [], []
    best_metric = float("inf")
    patience_counter = 0

    total_G, total_D = 0.0, 0.0
    iteration = 0

    while iteration < max_iterations:
        # -------------------
        # Critic updates
        # -------------------
        for _ in range(n_critic):
            idx = np.random.choice(total_samples, batch_size, replace=False)
            X_real = X_train[idx]
            M = M_train[idx]
            td = compute_delta_t(M)

            # LRMR-filled input to generator
            X_gen_in = M * X_real + (1 - M) * LR_train[idx]

            G_out = G(X_gen_in, M, td)               # (B, T, F)
            X_fake = M * X_real + (1 - M) * G_out    # keep observed, fill missing

            fake_mask = torch.ones_like(M)

            d_loss = discriminative_loss(D, X_fake, X_real, M, fake_mask)

            D_opt.zero_grad()
            d_loss.backward()
            D_opt.step()

            # WGAN weight clipping
            for p in D.parameters():
                p.data.clamp_(-weight_clipping, weight_clipping)

            total_D += float(d_loss.item())

        # -------------------
        # Generator update
        # -------------------
        idx = np.random.choice(total_samples, batch_size, replace=False)
        X_real = X_train[idx]
        M = M_train[idx]
        td = compute_delta_t(M)

        X_gen_in = M * X_real + (1 - M) * LR_train[idx]
        G_out = G(X_gen_in, M, td)
        X_fake = M * X_real + (1 - M) * G_out

        Lr = masked_reconstruction_loss(X_real, G_out, M)   # observed-only reconstruction (as you coded)
        Lg = generator_adv_loss(D, X_fake)
        g_loss = lambda_ * Lr + Lg

        G_opt.zero_grad()
        g_loss.backward()
        G_opt.step()

        total_G += float(g_loss.item())
        iteration += 1

        # -------------------
        # Logging / early stop check
        # -------------------
        if iteration % eval_interval == 0:
            avg_G = total_G / eval_interval
            avg_D = total_D / (eval_interval * n_critic)

            G_losses.append(avg_G)
            D_losses.append(avg_D)
            iters.append(iteration)

            total_G, total_D = 0.0, 0.0

            print(
                f"Iter {iteration}/{max_iterations} | "
                f"G: {avg_G:.6f} | D: {avg_D:.6f} | "
                f"LR(G): {G_opt.param_groups[0]['lr']:.6f} | LR(D): {D_opt.param_groups[0]['lr']:.6f}"
            )

            # Step schedulers (optional)
            G_sched.step(avg_G)
            D_sched.step(avg_D)

            # Early stopping on avg_G (you can switch to validation RMSE later if you want)
            if avg_G < best_metric:
                best_metric = avg_G
                patience_counter = 0
                torch.save(G.state_dict(), save_path)
            else:
                patience_counter += 1
                print(f"patience_counter: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print(f"Early stopping at iteration {iteration}")
                    break

    return G_losses, D_losses, iters

@torch.no_grad()
def test_generator(
    test_data: np.ndarray,
    original_test_data: np.ndarray,
    params: Dict,
    current_missing_file: str,
    model_path: str,
) -> Tuple[np.ndarray, float, float]:
    """
    Returns:
      imputed_data_full: (N, T, F) with observed preserved and missing filled
      missing-only RMSE
      missing-only MAE
    """
    seq_length = params["seq_length"]
    embed_dim = params["embed_dim"]
    num_heads = params["num_heads"]
    ffdim_gen = params["ffdim_gen"]
    num_layers = params["num_layers"]

    test_obs = 1 - np.isnan(test_data)
    test_filled = np.nan_to_num(test_data, nan=0.0)

    X_test = torch.tensor(test_filled, dtype=torch.float32).to(device)
    M_test = torch.tensor(test_obs, dtype=torch.float32).to(device)

    feature_dim = X_test.shape[2]
    num_sensors = feature_dim

    # Load LRMR for test split (same split function)
    lrmr_path = os.path.join(
        "LRMR outputs",
        os.path.basename(current_missing_file).replace(".npy", "_lrmr_imputed.npy")
    )
    if not os.path.exists(lrmr_path):
        raise FileNotFoundError(f"LRMR file not found: {lrmr_path}")

    lrmr_all = np.load(lrmr_path)
    _, lrmr_test = split_data(lrmr_all)
    LR_test = torch.tensor(lrmr_test, dtype=torch.float32).to(device)

    # Load Generator
    G = Generator(
        seq_length=seq_length,
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ffdim_gen=ffdim_gen,
        num_layers=num_layers,
        num_sensors=num_sensors,
        dropout=0.1,
    ).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    X_gen_in = M_test * X_test + (1 - M_test) * LR_test
    td = compute_delta_t(M_test)

    G_out = G(X_gen_in, M_test, td).cpu().numpy()
    X_test_np = X_test.cpu().numpy()
    M_test_np = M_test.cpu().numpy()

    # Preserve observed, fill missing
    imputed_full = M_test_np * X_test_np + (1 - M_test_np) * G_out

    rmse = rmse_loss_missing_only(original_test_data, imputed_full, M_test_np)
    mae = mae_loss_missing_only(original_test_data, imputed_full, M_test_np)
    return imputed_full, rmse, mae

def plot_losses(iters: List[int], G_losses: List[float], D_losses: List[float], out_path: str, title: str):
    plt.figure(figsize=(8, 5))
    plt.plot(iters, G_losses, label="Generator Loss (G)", linewidth=2)
    plt.plot(iters, D_losses, label="Discriminator Loss (D)", linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

# =============================================================================
# Main
# =============================================================================
def main():
    # ---------------------
    # Parameters (GitHub-friendly and explicit)
    # ---------------------
    params = {
        "batch_size": 32,
        "iterations": 50000,           
        "learning_rate": 1e-4,
        "seq_length": 100,
        "embed_dim": 128,
        "num_heads": 8,
        "ffdim_gen": 256,              # feed-forward dim in Generator encoders
        "ffdim_disc": 512,             # feed-forward dim in Discriminator block
        "num_layers": 1,
        "lambda": 50,
        "patience": 5,
        "n_critic": 5,
        "weight_clipping": 0.02,
        "eval_interval": 200,
    }
    num_runs = 3 

    # ---------------------
    # Load + normalize data
    # ---------------------
    dataset = pd.read_csv("sensor.csv").values
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(dataset).astype(np.float32)  # (T, F)

    # ---------------------
    # Missing mask files
    # ---------------------
    missing_files = [
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_10_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_20_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_30_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_40_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_50_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_60_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_70_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_80_percent_mcar.npy",
        "SKAB Data Missingness Patterns/MCAR/skab_missing_indices_90_percent_mcar.npy",

        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_10_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_20_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_30_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_40_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_50_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_60_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_70_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_80_percent_temporal_only.npy",
        "SKAB Data Missingness Patterns/Temporal Only/skab_missing_indices_90_percent_temporal_only.npy",

        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_10_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_20_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_30_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_40_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_50_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_60_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_70_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_80_percent_spatial_only.npy",
        "SKAB Data Missingness Patterns/Spatial Only/skab_missing_indices_90_percent_spatial_only.npy",

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

    with open(RESULTS_FILE, "a") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write("LR-WGAN Two-Encoder Results\n")
        log.write(f"num_runs={num_runs}, params={params}\n")
        log.write("=" * 100 + "\n")

        for missing_file in missing_files:
            miss_stem = safe_stem(missing_file)

            # Per-mask folders
            mask_dir = os.path.join(RUNS_DIR, miss_stem)
            mask_models_dir = os.path.join(mask_dir, "models")
            mask_plots_dir = os.path.join(mask_dir, "plots")
            mask_outputs_dir = os.path.join(mask_dir, "outputs")
            mask_logs_dir = os.path.join(mask_dir, "logs")
            for d in [mask_dir, mask_models_dir, mask_plots_dir, mask_outputs_dir, mask_logs_dir]:
                os.makedirs(d, exist_ok=True)

            print(f"\nProcessing mask: {missing_file}")
            log.write(f"\nMask: {missing_file}\n")

            # Load mask and apply missingness to normalized data
            missing_mask = np.load(missing_file)
            data_with_missingness = data_normalized.copy()
            data_with_missingness[missing_mask == 1] = np.nan

            # Build sequences
            data_sequences = create_sequences(data_with_missingness, params["seq_length"]).cpu().numpy()
            original_sequences = create_sequences(data_normalized, params["seq_length"]).cpu().numpy()

            # Split train/test
            train_data, test_data = split_data(data_sequences)
            _, original_test = split_data(original_sequences)

            # Run multiple times
            rmse_list, mae_list = [], []

            for run_id in range(1, num_runs + 1):
                run_dir = os.path.join(mask_dir, f"run{run_id}")
                run_models_dir = os.path.join(run_dir, "models")
                run_plots_dir = os.path.join(run_dir, "plots")
                run_outputs_dir = os.path.join(run_dir, "outputs")
                run_logs_dir = os.path.join(run_dir, "logs")
                for d in [run_dir, run_models_dir, run_plots_dir, run_outputs_dir, run_logs_dir]:
                    os.makedirs(d, exist_ok=True)

                print(f"  Run {run_id}/{num_runs}")
                log.write(f"  Run {run_id}/{num_runs}\n")

                # Save model per run (avoid overwriting)
                generator_path = os.path.join(run_models_dir, f"best_generator_{miss_stem}_run{run_id}.pth")

                # Train
                G_losses, D_losses, iters = train_lr_wgan(
                    train_data=train_data,
                    params=params,
                    save_path=generator_path,
                    current_missing_file=missing_file,
                )

                # Plot losses
                plot_path = os.path.join(run_plots_dir, f"losses_{miss_stem}_run{run_id}.png")
                plot_losses(
                    iters=iters,
                    G_losses=G_losses,
                    D_losses=D_losses,
                    out_path=plot_path,
                    title=f"Training Losses | {os.path.basename(missing_file)} | Run {run_id}",
                )

                # Test
                imputed, rmse, mae = test_generator(
                    test_data=test_data,
                    original_test_data=original_test,
                    params=params,
                    current_missing_file=missing_file,
                    model_path=generator_path,
                )

                # Save imputed outputs
                imp_path = os.path.join(run_outputs_dir, f"imputed_{miss_stem}_run{run_id}.npy")
                np.save(imp_path, imputed)

                rmse_list.append(rmse)
                mae_list.append(mae)

                print(f"    RMSE: {rmse:.6f} | MAE: {mae:.6f}")
                log.write(f"    saved_model={generator_path}\n")
                log.write(f"    saved_plot={plot_path}\n")
                log.write(f"    saved_imputed={imp_path}\n")
                log.write(f"    RMSE={rmse:.6f}, MAE={mae:.6f}\n")

            # Aggregate across runs
            avg_rmse = float(np.mean(rmse_list))
            std_rmse = float(np.std(rmse_list))
            avg_mae = float(np.mean(mae_list))
            std_mae = float(np.std(mae_list))

            print(f"  FINAL | RMSE: {avg_rmse:.6f} ± {std_rmse:.6f} | MAE: {avg_mae:.6f} ± {std_mae:.6f}")
            log.write(f"  FINAL | RMSE: {avg_rmse:.6f} ± {std_rmse:.6f} | MAE: {avg_mae:.6f} ± {std_mae:.6f}\n")
            log.write("-" * 100 + "\n")


if __name__ == "__main__":
    main()