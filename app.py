"""
app.py
------
Fase 4: Dashboard Web Interattiva - ET0 Predictor

Confronto in tempo reale tra:
  - Formula agronomica di Hargreaves-Samani (ET0_HS)
  - Rete Neurale PyTorch addestrata su 3 anni di dati (ET0_NN)

Avvio: streamlit run app.py
"""

import os
import subprocess
import numpy as np
import pandas as pd
import torch
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from pathlib import Path

from model import ET0Predictor

# ---------------------------------------------------------------------------
# Costanti e path
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).parent
DATA_FILE   = ROOT / "dati_meteo_agricoli.xlsx"
MODEL_FILE  = ROOT / "model_et0.pth"
SCALER_FILE = ROOT / "scaler_X.pkl"
PLOT_FILE   = ROOT / "risultati_modello.png"

FEATURE_NAMES = [
    "T_max_C", "T_min_C", "Umidita_Relativa_%",
    "Rad_Solare_MJ_m2", "Rad_Extraterr_MJ_m2", "Velocita_Vento_m_s"
]

# ---------------------------------------------------------------------------
# Funzioni agronomiche
# ---------------------------------------------------------------------------

def calcola_Ra(giorno_anno: int, latitude_deg: float = 45.0) -> float:
    """Radiazione extraterrestre Ra (MJ/m²/giorno) - FAO-56."""
    J        = float(giorno_anno)
    lat_rad  = np.radians(latitude_deg)
    dr       = 1 + 0.033 * np.cos(2 * np.pi / 365 * J)
    delta    = 0.409 * np.sin(2 * np.pi / 365 * J - 1.39)
    omega_s  = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    GSC      = 0.0820
    Ra = (24 * 60 / np.pi) * GSC * dr * (
        omega_s * np.sin(lat_rad) * np.sin(delta)
        + np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
    )
    return float(Ra)


def hargreaves_samani(T_max: float, T_min: float, Ra: float) -> float:
    """Formula Hargreaves-Samani (1985) per ET0 giornaliera."""
    T_med   = (T_max + T_min) / 2.0
    delta_T = max(T_max - T_min, 0.0)
    ET0     = 0.0023 * Ra * (T_med + 17.8) * np.sqrt(delta_T)
    return float(max(ET0, 0.0))


# ---------------------------------------------------------------------------
# Caricamento modello (cached)
# ---------------------------------------------------------------------------

@st.cache_resource
def carica_modello():
    """Carica ET0Predictor e scaler se esistono, altrimenti None."""
    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        return None, None, None
    try:
        ckpt   = torch.load(str(MODEL_FILE), map_location="cpu", weights_only=True)
        model  = ET0Predictor(input_dim=ckpt.get("input_dim", 6))
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        scaler = joblib.load(str(SCALER_FILE))
        meta   = {k: ckpt[k] for k in ("r2", "mae", "rmse") if k in ckpt}
        return model, scaler, meta
    except Exception as e:
        st.warning(f"Errore caricamento modello: {e}")
        return None, None, None


def predici_nn(model, scaler, t_max, t_min, umid, rad_sol, ra, vento) -> float:
    """Predizione ET0 con la rete neurale."""
    x = np.array([[t_max, t_min, umid, rad_sol, ra, vento]], dtype=np.float32)
    x_sc = scaler.transform(x)
    with torch.no_grad():
        pred = model(torch.tensor(x_sc)).item()
    return float(max(pred, 0.0))


# ---------------------------------------------------------------------------
# Layout Streamlit
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ET0 Predictor — PyTorch",
    page_icon="🌿",
    layout="wide",
)

st.title("🌿 ET0 Predictor — Evapotraspirazione di Riferimento")
st.subheader("Rete Neurale PyTorch vs Formula Hargreaves-Samani | Pianura Padana")

# Carica modello
model_nn, scaler_nn, meta_nn = carica_modello()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Azioni")

    if model_nn is None:
        st.warning("Modello non trovato.\nEsegui `python train.py` per addestrarlo.")
        if st.button("▶️ Genera Dataset + Addestra", type="primary"):
            with st.spinner("Generazione dataset..."):
                try:
                    if not DATA_FILE.exists():
                        subprocess.run(
                            ["python", str(ROOT / "generate_dataset.py")],
                            check=True, cwd=str(ROOT)
                        )
                    subprocess.run(
                        ["python", str(ROOT / "train.py")],
                        check=True, cwd=str(ROOT)
                    )
                    st.success("Training completato! Riavvia l'app.")
                    st.cache_resource.clear()
                except Exception as e:
                    st.error(f"Errore: {e}")
    else:
        st.success("✅ Modello caricato")
        if meta_nn:
            st.metric("R² Test", f"{meta_nn.get('r2', 0):.4f}")
            st.metric("MAE Test", f"{meta_nn.get('mae', 0):.3f} mm/gg")
            st.metric("RMSE Test", f"{meta_nn.get('rmse', 0):.3f} mm/gg")

    st.divider()
    st.header("📂 Dataset")
    if st.button("Genera dataset meteo"):
        with st.spinner("Generazione in corso..."):
            try:
                subprocess.run(
                    ["python", str(ROOT / "generate_dataset.py")],
                    check=True, cwd=str(ROOT)
                )
                st.success("Dataset generato.")
            except Exception as e:
                st.error(f"Errore: {e}")

    if PLOT_FILE.exists():
        st.divider()
        st.markdown("**Risultati training**")
        st.image(str(PLOT_FILE), caption="ET0 Reale vs Predetta (test set)")

# ---------------------------------------------------------------------------
# Pannello principale — Inputs
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 🎛️ Parametri Meteorologici")
st.caption("Inserisci i dati del giorno per calcolare ET0 con entrambi i metodi.")

col_left, col_right = st.columns([3, 2])

with col_left:
    c1, c2, c3 = st.columns(3)
    with c1:
        giorno = st.slider("📅 Giorno anno", 1, 365, 197,
                           help="1 = 1° Gen | 197 = 16° Lug | 365 = 31° Dic")
        t_max  = st.slider("🌡️ T max (°C)", -10.0, 45.0, 32.0, step=0.5)
        t_min  = st.slider("🌡️ T min (°C)", -15.0, 35.0, 20.0, step=0.5)
    with c2:
        umid    = st.slider("💧 Umidità rel. (%)", 10.0, 100.0, 45.0, step=1.0)
        rad_sol = st.slider("☀️ Rad. Solare (MJ/m²)", 0.0, 35.0, 22.0, step=0.5)
        vento   = st.slider("💨 Vento (m/s)", 0.0, 12.0, 2.5, step=0.1)
    with c3:
        ra = calcola_Ra(giorno)
        st.markdown("**📡 Rad. Extraterr. (Ra)**")
        st.info(f"**{ra:.2f}** MJ/m²/giorno\n\n_(calcolata automaticamente\nper lat. 45°N, giorno {giorno})_")

        # Validazione temperatura
        if t_min >= t_max:
            st.error("⚠️ T_min deve essere < T_max")

# ---------------------------------------------------------------------------
# Calcolo ET0
# ---------------------------------------------------------------------------

et0_hs = hargreaves_samani(t_max, t_min, ra)
et0_nn = predici_nn(model_nn, scaler_nn, t_max, t_min, umid, rad_sol, ra, vento) \
         if model_nn is not None else None

with col_right:
    st.markdown("### 📊 Risultati")
    res1, res2 = st.columns(2)

    with res1:
        st.markdown("#### 📐 Hargreaves-Samani")
        st.metric(
            label="ET0 (formula)",
            value=f"{et0_hs:.3f} mm/gg",
            help="Formula agronomica FAO-56 — solo T_max, T_min e Ra"
        )
        st.caption("_Solo temperatura e radiazione_")

    with res2:
        st.markdown("#### 🤖 Neural Network")
        if et0_nn is not None:
            delta = et0_nn - et0_hs
            st.metric(
                label="ET0 (PyTorch NN)",
                value=f"{et0_nn:.3f} mm/gg",
                delta=f"{delta:+.3f} vs HS",
                delta_color="off",
                help="Rete neurale addestrata su 6 variabili meteorologiche"
            )
            st.caption("_Considera anche umidità e vento_")
        else:
            st.metric(label="ET0 (PyTorch NN)", value="—")
            st.caption("_Addestra il modello dalla sidebar_")

# ---------------------------------------------------------------------------
# Grafico comparativo rapido
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("### 📈 Curva ET0 Stagionale (giorno per giorno, condizioni correnti)")

giorni  = np.arange(1, 366)
ra_arr  = np.array([calcola_Ra(g) for g in giorni])
et0_hs_arr = np.array([hargreaves_samani(t_max, t_min, ra_g) for ra_g in ra_arr])

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(giorni, et0_hs_arr, color="#2196F3", linewidth=1.8,
        label="ET0 Hargreaves-Samani (formula)", zorder=3)

if et0_nn is not None:
    # Calcola curva NN per tutti i giorni mantenendo le altre variabili costanti
    feats = np.column_stack([
        np.full(365, t_max),
        np.full(365, t_min),
        np.full(365, umid),
        np.full(365, rad_sol),
        ra_arr,
        np.full(365, vento),
    ]).astype(np.float32)
    feats_sc = scaler_nn.transform(feats)
    with torch.no_grad():
        et0_nn_arr = np.maximum(
            model_nn(torch.tensor(feats_sc)).numpy().flatten(), 0.0
        )
    ax.plot(giorni, et0_nn_arr, color="#FF5722", linewidth=1.8, linestyle="--",
            label="ET0 Neural Network (PyTorch)", zorder=4)

ax.axvline(giorno, color="#4CAF50", linewidth=2, linestyle=":", label=f"Giorno selezionato ({giorno})")
ax.set_xlabel("Giorno dell'anno", fontsize=10)
ax.set_ylabel("ET0 (mm/giorno)", fontsize=10)
ax.set_title("Andamento stagionale ET0 — con i parametri attuali dei slider", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 365)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ---------------------------------------------------------------------------
# Sezione dataset
# ---------------------------------------------------------------------------

st.markdown("---")
if DATA_FILE.exists():
    with st.expander("📋 Mostra dati storici (prime righe)"):
        df = pd.read_excel(DATA_FILE, sheet_name="Dati_Meteo_Giornalieri")
        st.dataframe(df.head(10), use_container_width=True)
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("ET0 media (mm/gg)", f"{df['ET0_Hargreaves_mm'].mean():.3f}")
        col_b.metric("ET0 max (mm/gg)",   f"{df['ET0_Hargreaves_mm'].max():.3f}")
        col_c.metric("Giorni nel dataset", f"{len(df):,}")
else:
    st.info("Dataset non trovato. Usa il pulsante in sidebar per generarlo.")

st.markdown("---")
st.caption(
    "**ET0 Predictor PyTorch** — Pianura Padana | "
    "Formula: Hargreaves-Samani 1985 (FAO-56) | "
    "Modello: feedforward 6→128→64→32→1 con BatchNorm e Dropout"
)
