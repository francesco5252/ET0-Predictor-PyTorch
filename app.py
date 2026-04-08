import streamlit as st
import pandas as pd
import numpy as np
import subprocess
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "dati_meteo_agricoli.xlsx"
MODEL_PLOT = ROOT / "risultati_modello.png"


def calcola_Ra(giorno_anno: int, latitude_deg: float = 45.0) -> float:
    """Radiazione extraterrestre giornaliera Ra (MJ/m2/giorno) - FAO-56."""
    J = np.asarray(giorno_anno)
    lat_rad = np.radians(latitude_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * J)
    delta = 0.409 * np.sin(2 * np.pi / 365 * J - 1.39)
    omega_s = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    GSC = 0.0820
    Ra = (24 * 60 / np.pi) * GSC * dr * (
        omega_s * np.sin(lat_rad) * np.sin(delta)
        + np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
    )
    return float(Ra)


def hargreaves_samani(T_max: float, T_min: float, Ra: float) -> float:
    T_med = (T_max + T_min) / 2.0
    delta_T = max(T_max - T_min, 0.0)
    ET0 = 0.0023 * Ra * (T_med + 17.8) * np.sqrt(delta_T)
    return float(max(ET0, 0.0))


st.set_page_config(page_title="ET0 Predictor — Demo", layout="wide")
st.title("ET0 Predictor — Demo (Pianura Padana)")

with st.sidebar:
    st.header("Azioni rapide")
    if st.button("Genera dataset (rapido)"):
        st.info("Avvio generate_dataset.py (salva dati in dati_meteo_agricoli.xlsx)")
        try:
            subprocess.run(["python", str(ROOT / "generate_dataset.py")], check=True)
            st.success("Dataset generato con successo.")
        except Exception as e:
            st.error(f"Errore generazione dataset: {e}")

    if MODEL_PLOT.exists():
        if st.button("Mostra grafico modello (risultati_modello.png)"):
            st.image(str(MODEL_PLOT), caption="Risultati modello ET0 (reale vs predetto)")
    else:
        st.markdown("_(Nessun grafico modello trovato. Esegui `train.py` per generarlo.)_")

st.markdown("---")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Dataset sample / Statistiche")
    if DATA_FILE.exists():
        df = pd.read_excel(DATA_FILE, sheet_name="Dati_Meteo_Giornalieri")
        st.dataframe(df.head(8))
        st.metric("ET0 media (mm/giorno)", f"{df['ET0_Hargreaves_mm'].mean():.3f}")
        st.metric("ET0 totale (mm)", f"{df['ET0_Hargreaves_mm'].sum():.1f}")

        # Plot semplice ET0 sul primo anno disponibile
        try:
            df_year = df.iloc[:365]
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(df_year['Data'], df_year['ET0_Hargreaves_mm'], color="#2196F3")
            ax.set_title("ET0 (Hargreaves) - primo anno")
            ax.set_ylabel("ET0 (mm/giorno)")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        except Exception:
            st.warning("Impossibile plottare la serie ET0.")
    else:
        st.warning("File dati_meteo_agricoli.xlsx non trovato. Clicca 'Genera dataset' nella sidebar.")

with col2:
    st.subheader("Calcolo ET0 - Hargreaves (Interattivo)")
    giorno = st.slider("Giorno dell'anno", 1, 365, 197)
    tmax = st.slider("Temperatura massima (°C)", -5.0, 45.0, 32.0)
    tmin = st.slider("Temperatura minima (°C)", -10.0, 30.0, 20.0)
    ra = calcola_Ra(giorno)
    st.write(f"Radiazione extraterrestre (Ra): {ra:.2f} MJ/m²/giorno")
    et0_val = hargreaves_samani(tmax, tmin, ra)
    st.metric("ET0 (Hargreaves)", f"{et0_val:.3f} mm/giorno")

    st.markdown("---")
    st.write("Se vuoi, puoi simulare uno scenario di stress idrico usando il toggle sotto.")
    stress = st.slider("Fattore stress (0 = siccità estrema, 1 = nessuno)", 0.0, 1.0, 1.0)
    st.metric("ET0 corretto per stress", f"{et0_val * stress:.3f} mm/giorno")

st.markdown("---")
st.info("Questa demo mostra la formula agronomica di riferimento. Per integrare la Rete Neurale:")
st.write("- Esegui `python train.py` per addestrare il modello e generare 'risultati_modello.png'.")
st.write("- Implementazioni future: pulsante 'Train rapido' che addestra e carica il modello in memoria.")
