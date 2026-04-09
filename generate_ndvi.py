"""
generate_ndvi.py
----------------
Pipeline GIS Sintetica: Generazione Dati NDVI Satellitari Simulati

Simula la curva stagionale di NDVI (Normalized Difference Vegetation Index)
per mais e frumento in Pianura Padana, con aggiunta di rumore statistico
che replica le perturbazioni da copertura nuvolosa tipiche di Copernicus/Sentinel-2.

Concetto: i dati satellitari reali sono spesso "sporcati" da nuvole.
Il dataset simulato include NaN (nuvola opaca) e valori ridotti (nuvola semi-trasparente),
insegnando al sistema a gestire dati mancanti tipici del telerilevamento.

Output:
  ndvi_satellitare.csv  -- dataset giornaliero NDVI con rumore nuvoloso
  ndvi_plot.png         -- visualizzazione della curva stagionale

Aggiornamenti 2 ET0-Predictor: Pipeline GIS Sintetica
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
OUTPUT_CSV = ROOT / "ndvi_satellitare.csv"
OUTPUT_PNG = ROOT / "ndvi_plot.png"

SEED = 42
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Curve NDVI stagionali per coltura
# ---------------------------------------------------------------------------

def curva_ndvi_mais(doy: np.ndarray) -> np.ndarray:
    """
    Curva NDVI stagionale per mais (aprile-settembre, Pianura Padana).

    Fasi:
      DOY < 110  : suolo nudo / preparazione
      110-180    : emergenza e crescita vegetativa rapida
      180-230    : massimo verde (area fogliare massima)
      230-280    : ingiallimento post-fioritura e maturazione
      > 280      : senescenza / raccolta
    """
    ndvi = np.full(len(doy), 0.08, dtype=float)  # baseline suolo nudo
    for i, d in enumerate(doy):
        if 110 <= d <= 280:
            x = (d - 110) / (280 - 110)
            # Crescita logistica + declino post-fioritura
            crescita = 0.85 / (1 + np.exp(-12 * (x - 0.30)))
            declino  = 1 - 0.35 * max(0, x - 0.60) / 0.40
            ndvi[i]  = max(0.08, crescita * declino)
    return np.clip(ndvi + np.random.normal(0, 0.015, len(doy)), 0.05, 0.95)


def curva_ndvi_frumento(doy: np.ndarray) -> np.ndarray:
    """
    Curva NDVI stagionale per frumento invernale (ottobre-luglio).

    Fasi:
      DOY 270-365: semina e emergenza autunnale
      DOY 1-120  : ripresa primaverile e accestimento
      DOY 120-180: levata, spigatura, massima biomassa
      DOY 180-210: maturazione e ingiallimento
    """
    ndvi = np.full(len(doy), 0.10, dtype=float)
    for i, d in enumerate(doy):
        if 270 <= d <= 365:
            x = (d - 270) / 95
            ndvi[i] = 0.25 * x + 0.10
        elif 1 <= d <= 120:
            x = d / 120
            ndvi[i] = 0.25 + 0.55 * x
        elif 120 < d <= 175:
            x = (d - 120) / 55
            ndvi[i] = 0.80 - 0.05 * x
        elif 175 < d <= 210:
            x = (d - 175) / 35
            ndvi[i] = 0.75 - 0.65 * x
    return np.clip(ndvi + np.random.normal(0, 0.015, len(doy)), 0.05, 0.92)


def aggiungi_rumore_nuvoloso(ndvi: np.ndarray, prob_nuvola: float = 0.15) -> np.ndarray:
    """
    Simula la copertura nuvolosa tipica di Sentinel-2.

    - 70% delle perturbazioni: NaN (nuvola opaca, dato mancante)
    - 30% delle perturbazioni: valore ridotto (nuvola semi-trasparente)

    Questo tipo di rumore replica la realta del telerilevamento:
    un satellite in orbita non passa ogni giorno, e quando passa
    potrebbe essere coperto da nuvole.
    """
    ndvi_noisy = ndvi.copy()
    mask_cloud  = np.random.random(len(ndvi)) < prob_nuvola
    nan_mask    = mask_cloud & (np.random.random(len(ndvi)) < 0.70)
    dim_mask    = mask_cloud & ~nan_mask

    ndvi_noisy[nan_mask] = np.nan
    ndvi_noisy[dim_mask] = ndvi_noisy[dim_mask] * np.random.uniform(0.30, 0.70, dim_mask.sum())
    return ndvi_noisy


# ---------------------------------------------------------------------------
# Generazione dataset multi-anno
# ---------------------------------------------------------------------------

def genera_ndvi_dataset(n_anni: int = 3) -> pd.DataFrame:
    records = []
    anno_start = 2022

    for anno in range(anno_start, anno_start + n_anni):
        doy_arr = np.arange(1, 366)

        ndvi_mais     = curva_ndvi_mais(doy_arr)
        ndvi_frumento = curva_ndvi_frumento(doy_arr)

        ndvi_mais_noisy     = aggiungi_rumore_nuvoloso(ndvi_mais,     prob_nuvola=0.15)
        ndvi_frumento_noisy = aggiungi_rumore_nuvoloso(ndvi_frumento, prob_nuvola=0.15)

        for j, doy in enumerate(doy_arr):
            try:
                data = pd.Timestamp(year=anno, month=1, day=1) + pd.Timedelta(days=int(doy) - 1)
            except Exception:
                continue

            val_mais = ndvi_mais_noisy[j]
            val_frum = ndvi_frumento_noisy[j]

            records.append({
                "Anno":              anno,
                "DOY":               int(doy),
                "Data":              data.strftime("%Y-%m-%d"),
                "NDVI_Mais":         round(float(np.clip(val_mais, 0, 1)), 4)
                                     if not np.isnan(val_mais) else None,
                "NDVI_Frumento":     round(float(np.clip(val_frum, 0, 1)), 4)
                                     if not np.isnan(val_frum) else None,
                "Copertura_Nuv_pct": round(float(np.random.beta(0.8, 4) * 80), 1),
                "Satellite":         "Sentinel-2 (simulato)",
                "Risoluzione_m":     10,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Visualizzazione
# ---------------------------------------------------------------------------

def visualizza_ndvi(df: pd.DataFrame):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("Pipeline GIS Sintetica — NDVI Sentinel-2 (Simulato)\nPianura Padana | mais e frumento",
                 fontsize=13, fontweight="bold")

    info = [
        ("NDVI_Mais",     "NDVI Mais (Zea mays L.)",           "#4CAF50"),
        ("NDVI_Frumento", "NDVI Frumento (Triticum aestivum)", "#FF9800"),
    ]

    for ax, (col, titolo, color) in zip(axes, info):
        for anno in sorted(df["Anno"].unique()):
            sub   = df[df["Anno"] == anno].copy()
            validi = sub[col].notna()
            ax.scatter(sub.loc[validi, "DOY"], sub.loc[validi, col],
                       alpha=0.55, s=9, color=color,
                       label=str(anno) if anno == df["Anno"].min() else "")
            # Linea di tendenza (media mobile)
            sub_v = sub.loc[validi].copy()
            if len(sub_v) > 5:
                sub_v_sorted = sub_v.sort_values("DOY")
                ndvi_smooth  = pd.Series(sub_v_sorted[col].values).rolling(7, center=True).mean()
                ax.plot(sub_v_sorted["DOY"].values, ndvi_smooth.values,
                        color=color, alpha=0.5, linewidth=1.2)

        nan_mask = df[col].isna()
        ax.scatter(df.loc[nan_mask, "DOY"], [0.02] * nan_mask.sum(),
                   marker="x", color="red", s=10, alpha=0.4,
                   label="Dato mancante (nuvola)")

        ax.set_ylabel("NDVI", fontsize=10)
        ax.set_xlabel("DOY (Giorno dell'anno)", fontsize=10)
        ax.set_title(f"{titolo} - con rumore nuvoloso (x = NaN da copertura)", fontsize=10)
        ax.set_ylim(-0.02, 1.0)
        ax.axhline(0.3, color="#B71C1C", linestyle="--", linewidth=0.8,
                   label="Soglia vegetazione attiva (0.3)")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Grafico NDVI salvato -> {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generazione dataset NDVI sintetico (Sentinel-2, 3 anni)...")
    df = genera_ndvi_dataset(n_anni=3)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Dataset salvato -> {OUTPUT_CSV} ({len(df)} osservazioni)")

    visualizza_ndvi(df)

    print("\nStatistiche dataset NDVI:")
    print(f"  Anni coperti   : {df['Anno'].min()} - {df['Anno'].max()}")
    print(f"  Tot. osservaz. : {len(df)}")
    for col in ["NDVI_Mais", "NDVI_Frumento"]:
        nan_pct = df[col].isna().mean() * 100
        print(f"  {col:<20}: NaN {nan_pct:.1f}% (copertura nuvolosa simulata)")
