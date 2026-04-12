"""
edge_export.py
--------------
Aggiornamento 4: Edge AI — Quantizzazione del Modello (MLOps)

Riduce il modello ET0Predictor da Float32 a Int8 tramite Dynamic Quantization.
Salva model_et0_int8.pt e stampa un benchmark completo su 30 giorni di test.

Caso d'uso: deploy su hardware edge (Raspberry Pi, centraline meteo solari).

CLI:
    python edge_export.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model import ET0Predictor  # noqa: E402

MODEL_FILE    = ROOT / "model_et0.pth"
SCALER_FILE   = ROOT / "scaler_X.pkl"
DATA_FILE     = ROOT / "dati_meteo_agricoli.xlsx"
OUTPUT_INT8   = ROOT / "model_et0_int8.pt"

FEATURE_NAMES = [
    "T_max_C", "T_min_C", "Umidita_Relativa_%",
    "Rad_Solare_MJ_m2", "Rad_Extraterr_MJ_m2", "Velocita_Vento_m_s"
]
N_BENCH_DAYS   = 30
N_REPEAT_BENCH = 100


def _carica_fp32():
    """Carica modello Float32 originale."""
    if not MODEL_FILE.exists():
        raise RuntimeError(f"Modello non trovato: {MODEL_FILE}\nEsegui: python train.py")
    try:
        torch.serialization.add_safe_globals([ET0Predictor])
    except AttributeError:
        pass  # PyTorch < 2.4
    try:
        ckpt = torch.load(str(MODEL_FILE), map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(str(MODEL_FILE), map_location="cpu")
    model = ET0Predictor(input_dim=ckpt.get("input_dim", 6))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def _quantizza(model_fp32: ET0Predictor) -> torch.nn.Module:
    """Applica Dynamic Quantization Int8 sui layer Linear."""
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return model_int8


def _carica_dati_benchmark():
    """
    Legge gli ultimi N_BENCH_DAYS record dal dataset.
    Ritorna (X_sc, y) come array numpy float32.
    """
    if not DATA_FILE.exists():
        raise RuntimeError(f"Dataset non trovato: {DATA_FILE}\nEsegui: python generate_dataset.py")
    if not SCALER_FILE.exists():
        raise RuntimeError(f"Scaler non trovato: {SCALER_FILE}\nEsegui: python train.py")

    df     = pd.read_excel(str(DATA_FILE), sheet_name="Dati_Meteo_Giornalieri")
    df_ben = df.tail(N_BENCH_DAYS).reset_index(drop=True)
    if len(df_ben) < N_BENCH_DAYS:
        print(f"[edge] Attenzione: dataset ha solo {len(df_ben)} righe (richieste {N_BENCH_DAYS})")
    scaler = joblib.load(str(SCALER_FILE))

    X_raw = df_ben[FEATURE_NAMES].values.astype(np.float32)
    X_sc  = scaler.transform(X_raw).astype(np.float32)

    et0_col = "ET0_Hargreaves_mm"
    y       = df_ben[et0_col].values.astype(np.float32)

    return X_sc, y


def _misura_velocita(model, X_tensor: torch.Tensor) -> float:
    """Ritorna tempo medio di inferenza in ms su N_REPEAT_BENCH ripetizioni."""
    with torch.no_grad():
        for _ in range(5):
            _ = model(X_tensor)
        t0 = time.perf_counter()
        for _ in range(N_REPEAT_BENCH):
            _ = model(X_tensor)
        elapsed = time.perf_counter() - t0
    return elapsed / N_REPEAT_BENCH * 1000.0


def _calcola_mae(model, X_tensor: torch.Tensor, y: np.ndarray) -> float:
    """Ritorna MAE rispetto ai valori ET0 reali."""
    with torch.no_grad():
        pred = model(X_tensor).numpy().flatten()
    pred = np.maximum(pred, 0.0)
    return float(np.mean(np.abs(pred - y)))


if __name__ == "__main__":
    print("[edge] Caricamento model_et0.pth (Float32)...")
    model_fp32 = _carica_fp32()

    print("[edge] Quantizzazione Dynamic Int8...")
    model_int8 = _quantizza(model_fp32)

    torch.save(model_int8, str(OUTPUT_INT8))
    print(f"[edge] Salvato -> {OUTPUT_INT8.name}")

    print(f"\n[edge] Caricamento {N_BENCH_DAYS} giorni di test...")
    X_sc, y = _carica_dati_benchmark()
    X_tensor = torch.tensor(X_sc, dtype=torch.float32)

    size_fp32  = os.path.getsize(str(MODEL_FILE))  / (1024 * 1024)
    size_int8  = os.path.getsize(str(OUTPUT_INT8)) / (1024 * 1024)
    size_delta = (size_fp32 - size_int8) / size_fp32 * 100

    vel_fp32   = _misura_velocita(model_fp32, X_tensor)
    vel_int8   = _misura_velocita(model_int8, X_tensor)
    vel_delta  = (vel_fp32 - vel_int8) / vel_fp32 * 100

    mae_fp32   = _calcola_mae(model_fp32, X_tensor, y)
    mae_int8   = _calcola_mae(model_int8, X_tensor, y)
    mae_delta  = mae_int8 - mae_fp32
    degradazione = abs(mae_delta / max(mae_fp32, 1e-6)) * 100

    print(f"\n{'='*54}")
    print(f"  BENCHMARK ({N_BENCH_DAYS} giorni test, {N_REPEAT_BENCH} ripetizioni)")
    print(f"{'='*54}")
    print(f"  {'Metrica':<22} {'Float32':>8}  {'Int8':>8}  {'Delta':>8}")
    print(f"  {'-'*50}")
    print(f"  {'Dimensione (MB)':<22} {size_fp32:>8.3f}  {size_int8:>8.3f}  {-size_delta:>+7.0f}%")
    print(f"  {'Inferenza (ms)':<22} {vel_fp32:>8.2f}  {vel_int8:>8.2f}  {-vel_delta:>+7.0f}%")
    print(f"  {'MAE ET0 (mm/g)':<22} {mae_fp32:>8.3f}  {mae_int8:>8.3f}  {mae_delta:>+8.3f}")
    print(f"{'='*54}")

    if degradazione < 5.0:
        print(f"[edge] Qualita' preservata: degradazione {degradazione:.1f}% < 5% -> deploy-ready")
    else:
        print(f"[edge] Attenzione: degradazione {degradazione:.1f}% > 5% -> verificare training")
