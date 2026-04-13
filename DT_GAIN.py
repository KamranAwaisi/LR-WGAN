
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ============================================================
# CONFIG
# ============================================================
SENSOR_CSV = "sensor.csv"

SEQ_LEN   = 100
TEST_SIZE = 0.2

NUM_RUNS    = 3
EPOCH_ITERS = 12000          # DT-GAIN iterations (not epochs)
BATCH_SIZE  = 32
LR          = 0.001
HINT_RATE   = 0.5
ALPHA       = 100.0

# Where to save everything
RUNS_ROOT   = "DT_GAIN_runs"
RESULTS_TXT = "DT_GAIN_results.txt"
SUMMARY_CSV = "DT_GAIN_summary.csv"

# Choose missingness files here
MISSING_FILES = [
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
    "SKAB Data Missingness Patterns/Temporal and Spatial/skab_missing_indices_90_percent_temporal_and_spatial.npy"
]

# ============================================================
# Helpers (I/O)
# ============================================================
def mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def append_txt(path: str, line: str):
    with open(path, "a") as f:
        f.write(line.rstrip() + "\n")

def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

# ============================================================
# Data utils
# ============================================================
def uniform_sampler(low, high, shape):
    return np.random.uniform(low, high, size=shape).astype(np.float32)

def binary_sampler(p, shape):
    return (np.random.uniform(0., 1., size=shape) < p).astype(np.float32)

def create_sequences_np(arr_2d: np.ndarray, seq_len: int) -> np.ndarray:
    """Sliding windows: (N,F) -> (N-seq_len+1, seq_len, F)"""
    N, F = arr_2d.shape
    if N < seq_len:
        raise ValueError(f"Not enough rows ({N}) for seq_len={seq_len}")
    out = np.stack([arr_2d[i:i+seq_len] for i in range(N - seq_len + 1)], axis=0)
    return out.astype(np.float32)

def split_train_test(a: np.ndarray, test_size: float = 0.2):
    n = len(a)
    n_test = int(n * test_size)
    if n_test <= 0:
        raise ValueError("test_size too small.")
    return a[:-n_test], a[-n_test:]

def rmse_missing_only(gt: np.ndarray, pred: np.ndarray, m_obs: np.ndarray) -> float:
    gt2 = gt.reshape(-1, gt.shape[-1])
    pr2 = pred.reshape(-1, pred.shape[-1])
    m2  = m_obs.reshape(-1, m_obs.shape[-1])
    miss = (1.0 - m2)

    den = float(np.sum(miss))
    if den == 0:
        return float("nan")
    num = float(np.sum((miss * (gt2 - pr2)) ** 2))
    return float(np.sqrt(num / den))

def mae_missing_only(gt: np.ndarray, pred: np.ndarray, m_obs: np.ndarray) -> float:
    gt2 = gt.reshape(-1, gt.shape[-1])
    pr2 = pred.reshape(-1, pred.shape[-1])
    m2  = m_obs.reshape(-1, m_obs.shape[-1])
    miss = (1.0 - m2)

    den = float(np.sum(miss))
    if den == 0:
        return float("nan")
    num = float(np.sum(miss * np.abs(gt2 - pr2)))
    return float(num / den)

# ============================================================
# Time decay (delta_t) computation
# ============================================================
def compute_delta_t(m_obs: torch.Tensor) -> torch.Tensor:
    B, T, F = m_obs.shape
    delta_t = torch.ones_like(m_obs, device=m_obs.device)
    for t in range(1, T):
        delta_t[:, t] = (delta_t[:, t - 1] + 1) * (1 - m_obs[:, t]) + m_obs[:, t]
    return delta_t

# ============================================================
# Positional Encoding
# ============================================================
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(1))  # (max_len,1,E)

    def forward(self, x):
        return x + self.pe[:x.size(0)]

# ============================================================
# Time-Decay Multihead Attention
# ============================================================
class TimeDecayMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, time_decay, attn_mask=None):
        T, B, E = x.shape

        Q = self.q_linear(x)
        K = self.k_linear(x)
        V = self.v_linear(x)

        Q = Q.view(T, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        K = K.view(T, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
        V = V.view(T, B, self.num_heads, self.head_dim).permute(1, 2, 0, 3)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        out  = torch.matmul(attn, V)

        out = out.permute(2, 0, 1, 3).contiguous().view(T, B, E)
        out = self.out_proj(out)
        out = out * time_decay
        return out

# ============================================================
# Transformer Encoder Blocks
# ============================================================
class TransformerEncoderBlockG(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
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

    def forward(self, x, time_decay, attn_mask=None):
        a = self.attn(x, time_decay, attn_mask)
        x = self.norm1(x + self.drop1(a))
        f = self.ffn(x)
        x = self.norm2(x + self.drop2(f))
        return x

class TransformerEncoderBlockD(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        a, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.drop1(a))
        f = self.ffn(x)
        x = self.norm2(x + self.drop2(f))
        return x

# ============================================================
# Generator / Discriminator
# ============================================================
class Generator(nn.Module):
    def __init__(self, feature_dim, embed_dim, num_heads, ff_dim, n_layers, dropout=0.1):
        super().__init__()
        self.fc_x = nn.Linear(feature_dim, embed_dim)
        self.pos = PositionalEncoding(embed_dim)
        self.delta_embed = nn.Linear(feature_dim, embed_dim)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlockG(embed_dim, num_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        self.out = nn.Linear(embed_dim, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, delta_t):
        x = self.fc_x(x).permute(1, 0, 2)  # (T,B,E)
        x = self.pos(x)

        decay = self.delta_embed(delta_t)           # (B,T,E)
        decay = torch.relu(decay)
        decay = torch.exp(-decay).permute(1, 0, 2)  # (T,B,E)

        for blk in self.blocks:
            x = blk(x, decay)

        x = x.permute(1, 0, 2)      # (B,T,E)
        x = self.out(x)             # (B,T,F)
        return self.sigmoid(x)

class Discriminator(nn.Module):
    def __init__(self, feature_dim, embed_dim, num_heads, ff_dim, n_layers, dropout=0.1):
        super().__init__()
        self.fc_x = nn.Linear(feature_dim, embed_dim)
        self.fc_h = nn.Linear(feature_dim, embed_dim)
        self.pos = PositionalEncoding(embed_dim * 2)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlockD(embed_dim * 2, num_heads, ff_dim, dropout) for _ in range(n_layers)
        ])
        self.out = nn.Linear(embed_dim * 2, feature_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        x = self.fc_x(x).permute(1, 0, 2)
        h = self.fc_h(h).permute(1, 0, 2)
        z = torch.cat([x, h], dim=-1)   # (T,B,2E)
        z = self.pos(z)

        for blk in self.blocks:
            z = blk(z)

        z = z.permute(1, 0, 2)
        z = self.out(z)
        return self.sigmoid(z)

# ============================================================
# DT-GAIN Train + Test
# ============================================================
def train_dt_gain(train_X_miss: np.ndarray, cfg: dict, save_path: str):
    eps = 1e-8

    m_obs = (~np.isnan(train_X_miss)).astype(np.float32)
    x_filled = np.nan_to_num(train_X_miss, nan=0.0).astype(np.float32)

    X = torch.tensor(x_filled, dtype=torch.float32, device=device)
    M = torch.tensor(m_obs,   dtype=torch.float32, device=device)

    N, T, F = X.shape

    G = Generator(F, cfg["embed_dim_g"], cfg["num_heads_g"], cfg["ff_dim_g"], cfg["num_layers_g"]).to(device)
    D = Discriminator(F, cfg["embed_dim_d"], cfg["num_heads_d"], cfg["ff_dim_d"], cfg["num_layers_d"]).to(device)

    optG = optim.Adam(G.parameters(), lr=cfg["lr"])
    optD = optim.Adam(D.parameters(), lr=cfg["lr"])

    iters = cfg["iterations"]
    bs = cfg["batch_size"]

    G.train(); D.train()

    for _ in tqdm(range(iters), desc="Training DT-GAIN"):
        idx = np.random.permutation(N)[:bs]
        X_mb = X[idx]
        M_mb = M[idx]

        delta_t = compute_delta_t(M_mb)

        Z = torch.tensor(uniform_sampler(0, 0.01, X_mb.shape), device=device)

        H_temp = torch.tensor(binary_sampler(cfg["hint_rate"], X_mb.shape), device=device)
        H = M_mb * H_temp

        X_tilde = M_mb * X_mb + (1 - M_mb) * Z

        # ---- D ----
        G_sample = G(X_tilde, delta_t)
        Hat_X = M_mb * X_mb + (1 - M_mb) * G_sample
        D_prob = D(Hat_X, H)

        D_loss = -torch.mean(
            M_mb * torch.log(D_prob + eps) +
            (1 - M_mb) * torch.log(1 - D_prob + eps)
        )

        optD.zero_grad()
        D_loss.backward()
        optD.step()

        # ---- G ----
        G_sample = G(X_tilde, delta_t)
        Hat_X = M_mb * X_mb + (1 - M_mb) * G_sample
        D_prob = D(Hat_X, H)

        G_adv = -torch.mean((1 - M_mb) * torch.log(D_prob + eps))
        G_rec = torch.mean(((M_mb * X_mb) - (M_mb * G_sample)) ** 2) / (torch.mean(M_mb) + eps)
        G_loss = G_adv + cfg["alpha"] * G_rec

        optG.zero_grad()
        G_loss.backward()
        optG.step()

    torch.save(G.state_dict(), save_path)

def impute_with_generator(test_X_miss: np.ndarray, cfg: dict, model_path: str):
    m_obs = (~np.isnan(test_X_miss)).astype(np.float32)
    x_filled = np.nan_to_num(test_X_miss, nan=0.0).astype(np.float32)

    X = torch.tensor(x_filled, dtype=torch.float32, device=device)
    M = torch.tensor(m_obs,   dtype=torch.float32, device=device)

    N, T, F = X.shape

    G = Generator(F, cfg["embed_dim_g"], cfg["num_heads_g"], cfg["ff_dim_g"], cfg["num_layers_g"]).to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    Z = torch.tensor(uniform_sampler(0, 0.01, X.shape), device=device)
    X_tilde = M * X + (1 - M) * Z
    delta_t = compute_delta_t(M)

    with torch.no_grad():
        G_sample = G(X_tilde, delta_t).detach().cpu().numpy()

    X_np = X.detach().cpu().numpy()
    M_np = M.detach().cpu().numpy()
    imputed = M_np * X_np + (1 - M_np) * G_sample
    return imputed, M_np

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    mkdir(RUNS_ROOT)

    raw = pd.read_csv(SENSOR_CSV).values.astype(np.float32)
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(raw).astype(np.float32)

    cfg = dict(
        seq_len=SEQ_LEN,
        iterations=EPOCH_ITERS,
        batch_size=BATCH_SIZE,
        lr=LR,
        hint_rate=HINT_RATE,
        alpha=ALPHA,

        embed_dim_g=64,
        ff_dim_g=256,
        num_heads_g=8,
        num_layers_g=1,

        embed_dim_d=128,
        ff_dim_d=512,
        num_heads_d=8,
        num_layers_d=1,
    )

    mkdir(RUNS_ROOT)

    append_txt(RESULTS_TXT, "\n" + "=" * 140)
    append_txt(RESULTS_TXT, f"DT-GAIN runs | {datetime.now().isoformat()} | device={device}")
    append_txt(RESULTS_TXT, f"CFG: {json.dumps(cfg)}")
    append_txt(RESULTS_TXT, "=" * 140)

    summary_rows = []

    for missing_file in MISSING_FILES:
        tag = os.path.basename(missing_file).replace(".npy", "")
        run_dir = mkdir(os.path.join(RUNS_ROOT, f"seq{SEQ_LEN}", tag))
        save_json(os.path.join(run_dir, "config.json"), {"missing_file": missing_file, **cfg})

        miss = np.load(missing_file)
        assert miss.shape == data_norm.shape, f"Mask shape {miss.shape} != data {data_norm.shape}"
        miss = (miss > 0.5).astype(np.int8)

        data_miss = data_norm.copy()
        data_miss[miss == 1] = np.nan

        X_miss = create_sequences_np(data_miss, SEQ_LEN)
        X_gt   = create_sequences_np(data_norm, SEQ_LEN)

        train_X, test_X = split_train_test(X_miss, test_size=TEST_SIZE)
        _, test_gt      = split_train_test(X_gt,   test_size=TEST_SIZE)

        append_txt(RESULTS_TXT, "\n" + "-" * 90)
        append_txt(RESULTS_TXT, f"Missing file: {missing_file}")
        append_txt(RESULTS_TXT, f"Train={len(train_X)} | Test={len(test_X)} | seq_len={SEQ_LEN}")

        rmse_runs, mae_runs = [], []

        for r in range(1, NUM_RUNS + 1):
            ckpt_dir = mkdir(os.path.join(run_dir, "ckpt"))
            ckpt_path = os.path.join(ckpt_dir, f"generator_run{r}.pth")

            print(f"\n[{tag}] Run {r}/{NUM_RUNS}")
            append_txt(RESULTS_TXT, f"\nRun {r}/{NUM_RUNS}")

            train_dt_gain(train_X, cfg, ckpt_path)
            imputed, m_obs_test = impute_with_generator(test_X, cfg, ckpt_path)

            rmse = rmse_missing_only(test_gt, imputed, m_obs_test)
            mae  = mae_missing_only(test_gt, imputed, m_obs_test)

            rmse_runs.append(rmse)
            mae_runs.append(mae)

            msg = (
                f"Run {r} | RMSE(missing-only) = {rmse:.6f} | "
                f"MAE(missing-only) = {mae:.6f} | ckpt={ckpt_path}"
            )
            print(msg)
            append_txt(RESULTS_TXT, msg)

        # ----------------------------
        # After all runs: log averages
        # ----------------------------
        avg_rmse = float(np.mean(rmse_runs))
        avg_mae  = float(np.mean(mae_runs))

        avg_msg = (
            f"[{tag}] AVG RMSE(missing-only) = {avg_rmse:.6f}\n"
            f"[{tag}] AVG MAE (missing-only) = {avg_mae:.6f}"
        )

        print("\n" + avg_msg)
        append_txt(RESULTS_TXT, avg_msg)

        summary_rows.append({
            "missing_file": missing_file,
            "missing_tag": tag,
            "seq_len": SEQ_LEN,
            "num_runs": NUM_RUNS,

            "avg_rmse_missing_only": avg_rmse,

            "avg_mae_missing_only": avg_mae,
            "run_dir": run_dir,
        })

    pd.DataFrame(summary_rows).to_csv(SUMMARY_CSV, index=False)
    print("\nSaved:")
    print("  -", RESULTS_TXT)
    print("  -", SUMMARY_CSV)
    print("  - runs under:", RUNS_ROOT)
