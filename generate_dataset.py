"""
generate_dataset.py
-------------------
Fase 1: Generazione del Dataset Meteorologico - Pianura Padana

Simula 365 giorni di dati climatici realistici per la Pianura Padana
(latitudine ~45 gradi N, zona Milano/Bologna) e calcola l'Evapotraspirazione
di riferimento (ET0) con la formula di Hargreaves-Samani.

L'ET0 e' il parametro fondamentale in agronomia per stimare il fabbisogno
idrico delle colture (ET_coltura = ET0 x Kc, con Kc coefficiente colturale).

Autore: Portfolio - Sustainable Agriculture & ML
"""

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Costanti geografiche - Pianura Padana
# ---------------------------------------------------------------------------
LATITUDE_DEG  = 45.0                         # latitudine media (gradi)
LATITUDE_RAD  = np.radians(LATITUDE_DEG)     # latitudine in radianti
GSC           = 0.0820                       # costante solare (MJ/m2/min)


# ---------------------------------------------------------------------------
# 1. Calcolo della Radiazione Extraterrestre (Ra)
#    Formula FAO-56 (Allen et al., 1998)
#    Ra e' la radiazione solare che raggiunge il limite superiore
#    dell'atmosfera; e' funzione del giorno dell'anno e della latitudine.
# ---------------------------------------------------------------------------

def calcola_Ra(giorno_anno: np.ndarray) -> np.ndarray:
    """
    Calcola la radiazione extraterrestre giornaliera Ra (MJ/m2/giorno).

    Parametri:
        giorno_anno: array di interi da 1 a 365

    Returns:
        Ra: radiazione extraterrestre (MJ/m2/giorno)
    """
    J = giorno_anno

    # Distanza relativa inversa Terra-Sole
    dr = 1 + 0.033 * np.cos(2 * np.pi / 365 * J)

    # Declinazione solare (radianti)
    delta = 0.409 * np.sin(2 * np.pi / 365 * J - 1.39)

    # Angolo orario al tramonto (radianti)
    omega_s = np.arccos(-np.tan(LATITUDE_RAD) * np.tan(delta))

    # Radiazione extraterrestre (MJ/m2/giorno)
    Ra = (24 * 60 / np.pi) * GSC * dr * (
        omega_s * np.sin(LATITUDE_RAD) * np.sin(delta)
        + np.cos(LATITUDE_RAD) * np.cos(delta) * np.sin(omega_s)
    )
    return Ra


# ---------------------------------------------------------------------------
# 2. Formula di Hargreaves-Samani per ET0
#    ET0 = 0.0023 * Ra * (T_med + 17.8) * sqrt(delta_T)
#    E' una formula empirica che richiede solo dati di temperatura e Ra,
#    molto usata quando mancano misure di umidita' e vento (es. dati storici).
# ---------------------------------------------------------------------------

def hargreaves_samani(T_max: np.ndarray,
                      T_min: np.ndarray,
                      Ra:    np.ndarray) -> np.ndarray:
    """
    Calcola ET0 con la formula di Hargreaves-Samani (mm/giorno).

    Parametri:
        T_max: temperatura massima giornaliera (gradi C)
        T_min: temperatura minima giornaliera (gradi C)
        Ra:    radiazione extraterrestre (MJ/m2/giorno)

    Returns:
        ET0: evapotraspirazione di riferimento (mm/giorno)
    """
    T_med    = (T_max + T_min) / 2.0
    delta_T  = np.maximum(T_max - T_min, 0)   # escursione termica (>= 0)
    ET0      = 0.0023 * Ra * (T_med + 17.8) * np.sqrt(delta_T)
    return np.maximum(ET0, 0.0)                # ET0 non puo' essere negativa


# ---------------------------------------------------------------------------
# 3. Generazione dati meteorologici realistici - Pianura Padana
#    Modello stagionale con variazione sinusoidale + rumore gaussiano
# ---------------------------------------------------------------------------

# Generiamo 3 anni di dati (2022-2024) per avere variabilita' stagionale
# completa in entrambi i set train e test con lo split random.
N_ANNI = 3
N_GIORNI = 365 * N_ANNI   # 1095 giorni

# Giorno dell'anno ciclico (1-365 ripetuto per ogni anno)
days_of_year = np.tile(np.arange(1, 366), N_ANNI)
# Giorno assoluto progressivo (per ordinamento temporale e date)
days_abs = np.arange(N_GIORNI)

# — Temperature (gradi C) —
# Ciclo stagionale: inverno freddo (gen), estate calda (lug)
# Picco estivo al giorno ~197 (16 luglio): fase = pi/2 - 2*pi*197/365
T_base_mean = 12.0   # temperatura media annua Pianura Padana
T_ampl      = 11.0   # semiampiezza stagionale
phase_shift = np.pi / 2 - 2 * np.pi * 197 / 365   # -1.818 rad -> picco a J=197

T_mean_base = T_base_mean + T_ampl * np.sin(2 * np.pi / 365 * days_of_year + phase_shift)

# Escursione termica giornaliera (maggiore in estate e primavera)
delta_T_base = 8.0 + 4.0 * np.sin(2 * np.pi / 365 * days_of_year + phase_shift)

T_max = T_mean_base + delta_T_base / 2 + np.random.normal(0, 1.5, N_GIORNI)
T_min = T_mean_base - delta_T_base / 2 + np.random.normal(0, 1.2, N_GIORNI)

# Assicura T_max > T_min con almeno 2 gradi di escursione
T_max = np.where(T_max - T_min < 2.0, T_min + 2.0 + np.random.uniform(0, 1, N_GIORNI), T_max)

# — Umidita' Relativa (%) —
# Pianura Padana: umidita' elevata in inverno (nebbie), minore in estate
UR_base = 78.0 - 12.0 * np.sin(2 * np.pi / 365 * days_of_year + phase_shift)
UR = np.clip(UR_base + np.random.normal(0, 6, N_GIORNI), 30, 99)

# — Radiazione Solare in superficie (MJ/m2/giorno) —
# Derivata da Ra con un coefficiente di trasmissivita' stagionale
Ra = calcola_Ra(days_of_year)
# Trasmissivita' atmosferica: minore in inverno (nuvole/nebbia), maggiore in estate
tau = 0.50 + 0.12 * np.sin(2 * np.pi / 365 * days_of_year + phase_shift)
tau = np.clip(tau + np.random.normal(0, 0.05, N_GIORNI), 0.20, 0.75)
Rs = Ra * tau   # radiazione solare in superficie

# — Velocita' del Vento (m/s a 2 m) —
# Leggermente piu' ventoso in primavera, calmo in estate
vento_base = 2.0 + 0.5 * np.sin(2 * np.pi / 365 * days_of_year - np.pi / 4)
vento = np.clip(vento_base + np.random.exponential(0.5, N_GIORNI), 0.5, 7.0)

# — ET0 con Hargreaves-Samani —
ET0 = hargreaves_samani(T_max, T_min, Ra)


# ---------------------------------------------------------------------------
# 4. Costruzione DataFrame e Export Excel
# ---------------------------------------------------------------------------

date_range = pd.date_range(start="2022-01-01", periods=N_GIORNI, freq="D")

df = pd.DataFrame({
    "Data":                  date_range,
    "Giorno_Anno":           days_of_year,
    "T_max_C":               np.round(T_max, 1),
    "T_min_C":               np.round(T_min, 1),
    "T_media_C":             np.round((T_max + T_min) / 2, 1),
    "Umidita_Relativa_%":    np.round(UR, 1),
    "Rad_Solare_MJ_m2":      np.round(Rs, 2),
    "Rad_Extraterr_MJ_m2":   np.round(Ra, 2),   # feature per il modello ML
    "Velocita_Vento_m_s":    np.round(vento, 2),
    "ET0_Hargreaves_mm":     np.round(ET0, 2),
})

OUTPUT_FILE = "dati_meteo_agricoli.xlsx"

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Dati_Meteo_Giornalieri")

    # Riepilogo mensile
    df["Mese"] = df["Data"].dt.month_name()
    mensile = df.groupby("Mese", sort=False).agg(
        T_max_media=("T_max_C", "mean"),
        T_min_media=("T_min_C", "mean"),
        UR_media=("Umidita_Relativa_%", "mean"),
        Rs_media=("Rad_Solare_MJ_m2", "mean"),
        Vento_medio=("Velocita_Vento_m_s", "mean"),
        ET0_totale_mm=("ET0_Hargreaves_mm", "sum"),
        ET0_media_giorn=("ET0_Hargreaves_mm", "mean"),
    ).round(2)

    # Ordine cronologico mesi
    mesi_ord = ["January", "February", "March", "April", "May", "June",
                "July", "August", "September", "October", "November", "December"]
    mensile = mensile.reindex(mesi_ord)
    mensile.to_excel(writer, sheet_name="Riepilogo_Mensile")

print(f"Dataset generato: {N_GIORNI} giorni ({N_ANNI} anni) -> '{OUTPUT_FILE}'")
print(f"\nET0 annuale totale: {ET0.sum():.1f} mm")
print(f"ET0 media giornaliera: {ET0.mean():.2f} mm/giorno")
print(f"\nStatistiche descrittive:")
print(df[["T_max_C", "T_min_C", "Umidita_Relativa_%",
          "Rad_Solare_MJ_m2", "Velocita_Vento_m_s", "ET0_Hargreaves_mm"]].describe().round(2))
