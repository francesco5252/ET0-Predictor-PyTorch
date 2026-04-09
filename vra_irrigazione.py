"""
vra_irrigazione.py
------------------
Aggiornamento 3: VRA (Variable Rate Application) — Mappa di Prescrizione Irrigazione

Genera una griglia spaziale 50x50 (2500 celle) che rappresenta un campo di 500m x 500m
nella Pianura Padana. Ogni cella riceve variazioni realistiche di microclima.
La rete neurale ET0Predictor elabora tutte le 2500 celle in batch -> heatmap ET0.

CLI:
    python vra_irrigazione.py
    python vra_irrigazione.py --giorno 210 --soglia-stress 5.0

Output:
    vra_heatmap.png  (se output_path non e' None)
"""

import sys
import os
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import joblib
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model import ET0Predictor

MODEL_FILE  = ROOT / "model_et0.pth"
SCALER_FILE = ROOT / "scaler_X.pkl"
DATA_FILE   = ROOT / "dati_meteo_agricoli.xlsx"
OUTPUT_PNG  = ROOT / "vra_heatmap.png"

FEATURE_NAMES = [
    "T_max_C", "T_min_C", "Umidita_Relativa_%",
    "Rad_Solare_MJ_m2", "Rad_Extraterr_MJ_m2", "Velocita_Vento_m_s"
]

N_ROWS = 50
N_COLS = 50


def _carica_modello_scaler():
    if not MODEL_FILE.exists():
        raise RuntimeError(
            f"Modello non trovato: {MODEL_FILE}\n"
            "Esegui prima: python train.py"
        )
    if not SCALER_FILE.exists():
        raise RuntimeError(
            f"Scaler non trovato: {SCALER_FILE}\n"
            "Esegui prima: python train.py"
        )
    torch.serialization.add_safe_globals([ET0Predictor])
    ckpt  = torch.load(str(MODEL_FILE), map_location="cpu", weights_only=True)
    model = ET0Predictor(input_dim=ckpt.get("input_dim", 6))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = joblib.load(str(SCALER_FILE))
    return model, scaler


def _leggi_parametri_base(giorno: int) -> dict:
    if not DATA_FILE.exists():
        return {
            "T_max_C": 30.0,
            "T_min_C": 18.0,
            "Umidita_Relativa_%": 55.0,
            "Rad_Solare_MJ_m2": 22.0,
            "Rad_Extraterr_MJ_m2": 38.0,
            "Velocita_Vento_m_s": 2.0,
        }
    df = pd.read_excel(str(DATA_FILE), sheet_name="Dati_Meteo_Giornalieri")
    for _candidate in ("DOY", "Giorno_Anno", "giorno_anno"):
        if _candidate in df.columns:
            doy_col = _candidate
            break
    else:
        doy_col = df.columns[0]
    idx = (df[doy_col] - giorno).abs().idxmin()
    row = df.loc[idx]
    return {feat: float(row[feat]) for feat in FEATURE_NAMES}


def _genera_griglia(base: dict, n_rows: int = N_ROWS, n_cols: int = N_COLS) -> np.ndarray:
    rows_idx = np.repeat(np.arange(n_rows), n_cols)
    cols_idx = np.tile(np.arange(n_cols), n_rows)
    row_norm = rows_idx / (n_rows - 1)
    col_norm = cols_idx / (n_cols - 1)
    delta_t = -0.5 + row_norm * 1.0
    dist_nw = np.sqrt(row_norm**2 + col_norm**2) / np.sqrt(2)
    delta_vento = (1.0 - dist_nw) * 0.3
    delta_umid = 5.0 * np.exp(-((row_norm - 0.5)**2) / (2 * 0.1**2))
    grid = np.column_stack([
        np.full(n_rows * n_cols, base["T_max_C"])             + delta_t,
        np.full(n_rows * n_cols, base["T_min_C"])             + delta_t,
        np.clip(
            np.full(n_rows * n_cols, base["Umidita_Relativa_%"]) + delta_umid,
            10.0, 100.0
        ),
        np.full(n_rows * n_cols, base["Rad_Solare_MJ_m2"]),
        np.full(n_rows * n_cols, base["Rad_Extraterr_MJ_m2"]),
        np.clip(
            np.full(n_rows * n_cols, base["Velocita_Vento_m_s"]) + delta_vento,
            0.0, 15.0
        ),
    ]).astype(np.float32)
    return grid


def genera_vra(giorno: int = 180, soglia_stress: float = None,
               output_path=OUTPUT_PNG):
    """
    Ritorna (fig, stats).
    stats = {
        "et0_min": float, "et0_max": float, "et0_mean": float, "et0_std": float,
        "n_celle_stress": int, "pct_stress": float, "soglia": float,
    }
    Salva PNG solo se output_path is not None.
    """
    model, scaler = _carica_modello_scaler()
    base          = _leggi_parametri_base(giorno)
    grid          = _genera_griglia(base)

    X_sc = scaler.transform(grid)
    with torch.no_grad():
        et0_flat = model(torch.tensor(X_sc, dtype=torch.float32))
        et0_flat = np.maximum(et0_flat.numpy().flatten(), 0.0)

    et0_grid = et0_flat.reshape(N_ROWS, N_COLS)

    media = float(et0_flat.mean())
    std   = float(et0_flat.std())
    if soglia_stress is None:
        soglia_stress = media + std

    n_stress = int(np.sum(et0_flat > soglia_stress))
    pct      = n_stress / len(et0_flat) * 100.0

    stats = {
        "et0_min": float(et0_flat.min()),
        "et0_max": float(et0_flat.max()),
        "et0_mean": media,
        "et0_std": std,
        "n_celle_stress": n_stress,
        "pct_stress": pct,
        "soglia": soglia_stress,
    }

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        et0_grid, cmap="RdYlGn_r", origin="upper",
        aspect="equal", interpolation="bilinear",
    )
    ax.contour(et0_grid, levels=[soglia_stress],
               colors=["white"], linewidths=[1.5], linestyles=["--"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("ET0 (mm/giorno)", fontsize=10)
    ax.set_title(
        f"Mappa VRA — ET0 Neurale (mm/g) | Giorno {giorno}\n"
        f"Media: {media:.2f} | Soglia stress: {soglia_stress:.2f} | "
        f"Zone critiche: {pct:.1f}%",
        fontsize=11,
    )
    ax.set_xlabel("Longitudine (W -> E)", fontsize=9)
    ax.set_ylabel("Latitudine (N -> S)", fontsize=9)
    ax.set_xticks([0, 24, 49])
    ax.set_xticklabels(["Ovest", "Centro", "Est"], fontsize=8)
    ax.set_yticks([0, 24, 49])
    ax.set_yticklabels(["Nord", "Centro", "Sud"], fontsize=8)
    plt.tight_layout()

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")

    return fig, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VRA Irrigazione — Mappa ET0 spaziale")
    parser.add_argument("--giorno", type=int, default=180,
                        help="Giorno dell'anno (1-365, default: 180)")
    parser.add_argument("--soglia-stress", type=float, default=None,
                        help="Soglia ET0 zona stress mm/g (default: media+1std)")
    args = parser.parse_args()

    print("[vra] Generazione griglia 50x50 (2500 celle)...")
    t0 = time.perf_counter()
    fig, stats = genera_vra(
        giorno=args.giorno,
        soglia_stress=args.soglia_stress,
        output_path=OUTPUT_PNG,
    )
    plt.close(fig)
    elapsed = time.perf_counter() - t0

    print(f"[vra] ET0 media: {stats['et0_mean']:.2f} mm/g | "
          f"Min: {stats['et0_min']:.2f} | Max: {stats['et0_max']:.2f}")
    print(f"[vra] Zone stress (ET0 > {stats['soglia']:.2f}): "
          f"{stats['n_celle_stress']} celle ({stats['pct_stress']:.1f}%)")
    print(f"[vra] Tempo inference: {elapsed*1000:.0f} ms")
    print(f"[vra] Salvata -> {OUTPUT_PNG.name}")
