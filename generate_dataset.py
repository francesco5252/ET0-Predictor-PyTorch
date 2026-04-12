"""
generate_dataset.py
-------------------
Fase 1: Generazione del Dataset Meteorologico - Pianura Padana

Simula 365 giorni di dati climatici realistici per la Pianura Padana
(latitudine ~45 gradi N, zona Milano/Bologna) e calcola l'Evapotraspirazione
di riferimento (ET0) con la formula di Hargreaves-Samani.

L'ET0 e' il parametro fondamentale in agronomia per stimare il fabbisogno
idrico delle colture (ET_coltura = ET0 x Kc, con Kc coefficiente colturale).

Aggiornamenti 2 ET0-Predictor - Stress-Test Anomalie Climatiche:
  Modalita' scenario con flag --scenario:
    normale   (default): clima Pianura Padana storico
    siccita  : T_max +4 gradi C, Umidita -15%, Rad +2 MJ/m2  -> ET0 molto piu' alta
    alluvione: T_max -3 gradi C, Umidita +15%, Rad -3 MJ/m2  -> ET0 piu' bassa

Uso:
  python generate_dataset.py                    # scenario normale
  python generate_dataset.py --scenario siccita
  python generate_dataset.py --scenario alluvione

Autore: Portfolio - Sustainable Agriculture & ML
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
SEED = 42

# ---------------------------------------------------------------------------
# Costanti e Formule Base
# ---------------------------------------------------------------------------
LATITUDE_DEG = 45.0
LATITUDE_RAD = np.radians(LATITUDE_DEG)
GSC          = 0.0820   # costante solare (MJ/m2/min)

def calcola_Ra(giorno_anno: np.ndarray) -> np.ndarray:
    """Calcola la radiazione extraterrestre giornaliera Ra (MJ/m2/giorno)."""
    J       = giorno_anno
    dr      = 1 + 0.033 * np.cos(2 * np.pi / 365 * J)
    delta   = 0.409 * np.sin(2 * np.pi / 365 * J - 1.39)
    omega_s = np.arccos(-np.tan(LATITUDE_RAD) * np.tan(delta))

    Ra = (24 * 60 / np.pi) * GSC * dr * (
        omega_s * np.sin(LATITUDE_RAD) * np.sin(delta)
        + np.cos(LATITUDE_RAD) * np.cos(delta) * np.sin(omega_s)
    )
    return Ra

def hargreaves_samani(T_max: np.ndarray, T_min: np.ndarray, Ra: np.ndarray) -> np.ndarray:
    """Calcola ET0 con la formula di Hargreaves-Samani (mm/giorno)."""
    T_med   = (T_max + T_min) / 2.0
    delta_T = np.maximum(T_max - T_min, 0)
    ET0     = 0.0023 * Ra * (T_med + 17.8) * np.sqrt(delta_T)
    return np.maximum(ET0, 0.0)

# ---------------------------------------------------------------------------
# Classe OOP
# ---------------------------------------------------------------------------

class MeteoDatasetGenerator:
    """Generatore di dataset meteorologici sintetici per la predizione di ET0."""
    
    SCENARI = {
        "normale":   {"delta_T": 0.0,  "delta_UR": 0.0,  "delta_Rs": 0.0},
        "siccita":   {"delta_T": 4.0,  "delta_UR": -15.0, "delta_Rs": 2.0},
        "alluvione": {"delta_T": -3.0, "delta_UR": 15.0,  "delta_Rs": -3.0},
    }

    def __init__(self, scenario: str = "normale", n_anni: int = 3, seed: int = 42):
        if scenario not in self.SCENARI:
            raise ValueError(f"Scenario non valido: '{scenario}'. Scegli tra {list(self.SCENARI.keys())}")
        self.scenario = scenario
        self.n_anni = n_anni
        self.seed = seed
        self.modificatori = self.SCENARI[scenario]

    def genera(self) -> pd.DataFrame:
        """Genera e restituisce il dataset."""
        np.random.seed(self.seed)
        N_GIORNI = 365 * self.n_anni
        days_of_year = np.tile(np.arange(1, 366), self.n_anni)
        phase_shift = np.pi / 2 - 2 * np.pi * 197 / 365

        # Temperature
        T_base_mean = 12.0 + self.modificatori["delta_T"]
        T_ampl = 11.0
        T_mean_base = T_base_mean + T_ampl * np.sin(2 * np.pi / 365 * days_of_year + phase_shift)
        delta_T_base = 8.0 + 4.0 * np.sin(2 * np.pi / 365 * days_of_year + phase_shift)

        T_max = T_mean_base + delta_T_base / 2 + np.random.normal(0, 1.5, N_GIORNI)
        T_min = T_mean_base - delta_T_base / 2 + np.random.normal(0, 1.2, N_GIORNI)
        T_max = np.where(T_max - T_min < 2.0, T_min + 2.0 + np.random.uniform(0, 1, N_GIORNI), T_max)

        if self.scenario == "siccita":
            T_max += np.random.uniform(0.5, 1.5, N_GIORNI) 

        # Umidita Relativa
        UR_base = 78.0 - 12.0 * np.sin(2 * np.pi / 365 * days_of_year + phase_shift)
        UR_base += self.modificatori["delta_UR"]
        UR = np.clip(UR_base + np.random.normal(0, 6, N_GIORNI), 15, 99)

        # Radiazione Solare
        Ra = calcola_Ra(days_of_year)
        tau = 0.50 + 0.12 * np.sin(2 * np.pi / 365 * days_of_year + phase_shift)
        tau = np.clip(tau + np.random.normal(0, 0.05, N_GIORNI), 0.20, 0.75)
        Rs = Ra * tau + self.modificatori["delta_Rs"]
        Rs = np.maximum(Rs, 0.5)

        # Vento
        vento_base = 2.0 + 0.5 * np.sin(2 * np.pi / 365 * days_of_year - np.pi / 4)
        vento_extra = 0.5 if self.scenario == "siccita" else 0.0
        vento = np.clip(vento_base + vento_extra + np.random.exponential(0.5, N_GIORNI), 0.5, 7.0)

        # ET0 Hargreaves-Samani
        ET0 = hargreaves_samani(T_max, T_min, Ra)

        date_range = pd.date_range(start="2022-01-01", periods=N_GIORNI, freq="D")

        df = pd.DataFrame({
            "Data":                 date_range,
            "Giorno_Anno":          days_of_year,
            "T_max_C":              np.round(T_max, 1),
            "T_min_C":              np.round(T_min, 1),
            "T_media_C":            np.round((T_max + T_min) / 2, 1),
            "Umidita_Relativa_%":   np.round(UR, 1),
            "Rad_Solare_MJ_m2":     np.round(Rs, 2),
            "Rad_Extraterr_MJ_m2":  np.round(Ra, 2),
            "Velocita_Vento_m_s":   np.round(vento, 2),
            "ET0_Hargreaves_mm":    np.round(ET0, 2),
            "Scenario":             self.scenario,
        })
        return df

    def salva_excel(self, df: pd.DataFrame, output_file: str) -> str:
        """Esporta il dataset in Excel."""
        df_exp = df.copy()
        df_exp["Mese"] = df_exp["Data"].dt.month_name()

        mensile = df_exp.groupby("Mese", sort=False).agg(
            T_max_media=("T_max_C", "mean"),
            T_min_media=("T_min_C", "mean"),
            UR_media=("Umidita_Relativa_%", "mean"),
            Rs_media=("Rad_Solare_MJ_m2", "mean"),
            Vento_medio=("Velocita_Vento_m_s", "mean"),
            ET0_totale_mm=("ET0_Hargreaves_mm", "sum"),
            ET0_media_giorn=("ET0_Hargreaves_mm", "mean"),
        ).round(2)

        mesi_ord = ["January", "February", "March", "April", "May", "June",
                    "July", "August", "September", "October", "November", "December"]
        mensile = mensile.reindex(mesi_ord)

        with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
            df_exp.to_excel(writer, index=False, sheet_name="Dati_Meteo_Giornalieri")
            mensile.to_excel(writer, sheet_name="Riepilogo_Mensile")
        return output_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera dataset meteorologico ET0 per Pianura Padana.")
    parser.add_argument("--scenario", choices=list(MeteoDatasetGenerator.SCENARI.keys()), default="normale", help="Scenario climatico")
    parser.add_argument("--anni", type=int, default=3, help="Numero di anni da simulare (default: 3)")
    args = parser.parse_args()

    print(f"Generazione dataset — scenario: {args.scenario.upper()} | anni: {args.anni}")
    
    generator = MeteoDatasetGenerator(scenario=args.scenario, n_anni=args.anni)
    df = generator.genera()
    
    output_file = ROOT / "dati_meteo_agricoli.xlsx"
    generator.salva_excel(df, str(output_file))

    ET0 = df["ET0_Hargreaves_mm"]
    print(f"Dataset generato: {len(df)} giorni -> '{output_file}'")
    print(f"\nET0 annuale totale : {ET0.sum():.1f} mm\nET0 media giorn.   : {ET0.mean():.2f} mm/giorno")
