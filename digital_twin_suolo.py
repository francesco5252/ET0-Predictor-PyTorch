"""
digital_twin_suolo.py
---------------------
Digital Twin del Suolo: Simulazione Rete di Sensori IoT Multi-Livello

Simula una rete di sensori virtuali di umidita del suolo a 3 profondita
(10 cm, 30 cm, 50 cm) e calcola il bilancio idrico sotterraneo in tempo reale.

Replica il concetto di IoT AgriTech: un campo reale avrebbe sensori FDR/TDR
interrati a diverse profondita per monitorare la riserva idrica del profilo.

Concetti agronomici:
  - Capacita di campo (CC): massima acqua ritenuta dopo drenaggio libero
  - Punto di appassimento (PA): minima acqua disponibile per le piante
  - Acqua disponibile (AD) = CC - PA
  - Bilancio idrico: delta_umidita = Pioggia - ET0_coltura - Drenaggio

Output:
  bilancio_idrico_iot.csv  -- log giornaliero dei 3 sensori
  digital_twin_plot.png    -- visualizzazione multi-profondita

Aggiornamenti 2 ET0-Predictor: Digital Twin del Suolo
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT       = Path(__file__).parent
OUTPUT_CSV = ROOT / "bilancio_idrico_iot.csv"
OUTPUT_PNG = ROOT / "digital_twin_plot.png"

SEED = 42
np.random.seed(SEED)

# Parametri fisici suolo franco-limoso Pianura Padana
FIELD_CAPACITY = {10: 42.0, 30: 38.0, 50: 34.0}  # % volumetrica
WILTING_POINT  = {10: 18.0, 30: 16.0, 50: 14.0}
SATURAZIONE    = {10: 60.0, 30: 55.0, 50: 50.0}

# Temperature medie mensili Pianura Padana (t_max_media, t_min_media)
TEMP_MENSILI = {
    1:  (6.5,  0.5),  2:  (9.0,  2.0),  3:  (14.0,  5.5),
    4:  (18.5,  9.0), 5:  (23.5, 13.5), 6:  (28.0, 17.5),
    7:  (31.5, 20.0), 8:  (30.5, 19.0), 9:  (25.5, 14.5),
    10: (18.5,  9.0), 11: (11.5,  4.5), 12: ( 7.0,  1.0),
}


def calcola_Ra(doy: int, lat_deg: float = 45.0) -> float:
    """Radiazione extraterrestre (MJ/m2/gg) - FAO-56."""
    J       = float(doy)
    lat_rad = np.radians(lat_deg)
    dr      = 1 + 0.033 * np.cos(2 * np.pi / 365 * J)
    delta   = 0.409 * np.sin(2 * np.pi / 365 * J - 1.39)
    omega_s = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    GSC     = 0.0820
    return (24 * 60 / np.pi) * GSC * dr * (
        omega_s * np.sin(lat_rad) * np.sin(delta)
        + np.cos(lat_rad) * np.cos(delta) * np.sin(omega_s)
    )


def hargreaves_et0(t_max: float, t_min: float, ra: float) -> float:
    """ET0 semplificata Hargreaves-Samani."""
    t_med   = (t_max + t_min) / 2.0
    delta_t = max(0.0, t_max - t_min)
    return max(0.0, 0.0023 * ra * (t_med + 17.8) * np.sqrt(delta_t))


def classifica_stress(umidita: float, profondita: int) -> str:
    """Classifica lo stress idrico in base all acqua disponibile residua."""
    fc = FIELD_CAPACITY[profondita]
    wp = WILTING_POINT[profondita]
    pct_ad = (umidita - wp) / max(fc - wp, 1.0)
    if pct_ad < 0.20:
        return "CRITICO"
    elif pct_ad < 0.45:
        return "MODERATO"
    elif pct_ad < 0.75:
        return "LIEVE"
    else:
        return "OTTIMALE"


class SensoreIoT:
    """Sensore FDR/TDR virtuale a profondita fissa."""

    def __init__(self, profondita_cm: int, umidita_iniziale: float = None):
        self.profondita = profondita_cm
        self.fc  = FIELD_CAPACITY[profondita_cm]
        self.wp  = WILTING_POINT[profondita_cm]
        self.sat = SATURAZIONE[profondita_cm]
        self.umidita = umidita_iniziale if umidita_iniziale is not None else self.fc * 0.85
        self.storico = []

    def aggiorna(self, precip_eff_mm: float, et0_mm: float) -> float:
        """
        Bilancio idrico giornaliero semplificato.
        La precipitazione efficace diminuisce con la profondita (filtrata).
        ET0 ridotta dal coefficiente colturale (Kc mais media stagionale ~0.75).
        """
        Kc = 0.75
        # Sensore rumoroso: ±0.2%
        variazione = (precip_eff_mm - et0_mm * Kc) / (self.profondita * 0.35)
        rumore     = np.random.normal(0, 0.2)
        self.umidita = float(np.clip(self.umidita + variazione + rumore,
                                     self.wp, self.sat))
        self.storico.append(round(self.umidita, 2))
        return self.umidita


class DigitalTwinCampo:
    """Gemello Digitale del campo con rete IoT a 3 profondita."""

    def __init__(self):
        self.sensori = {
            10: SensoreIoT(10),
            30: SensoreIoT(30),
            50: SensoreIoT(50),
        }
        self.log = []

    def simula_giorno(self, doy: int, t_max: float, t_min: float, precip_mm: float) -> dict:
        ra   = calcola_Ra(doy)
        et0  = hargreaves_et0(t_max, t_min, ra)

        letture = {}
        stress  = {}
        for depth, sensore in self.sensori.items():
            # Precipitazione efficace: diminuisce esponenzialmente con la profondita
            precip_eff = precip_mm * np.exp(-0.018 * depth)
            letture[depth] = sensore.aggiorna(precip_eff, et0)
            stress[depth]  = classifica_stress(letture[depth], depth)

        record = {
            "DOY":             doy,
            "T_max_C":         round(t_max, 1),
            "T_min_C":         round(t_min, 1),
            "Precip_mm":       round(precip_mm, 1),
            "ET0_mm":          round(et0, 2),
            "Umid_10cm_pct":   letture[10],
            "Umid_30cm_pct":   letture[30],
            "Umid_50cm_pct":   letture[50],
            "Stress_10cm":     stress[10],
            "Stress_30cm":     stress[30],
            "Stress_50cm":     stress[50],
            "Umid_media_pct":  round(np.mean(list(letture.values())), 2),
        }
        self.log.append(record)
        return record

    def simula_stagione(self, n_giorni: int = 90, doy_inizio: int = 120) -> pd.DataFrame:
        """Simula n_giorni dalla data di semina (maggio = DOY 120)."""
        for i in range(n_giorni):
            doy = (doy_inizio + i - 1) % 365 + 1
            # Mese corrente per le temperature medie
            data = pd.Timestamp(year=2024, month=1, day=1) + pd.Timedelta(days=doy - 1)
            m    = data.month
            t_max_m, t_min_m = TEMP_MENSILI.get(m, (20, 10))
            t_max = t_max_m + np.random.normal(0, 2.0)
            t_min = t_min_m + np.random.normal(0, 1.5)
            t_min = min(t_min, t_max - 2)  # t_min sempre < t_max

            # Precipitazione: 25% probabilita di pioggia giornaliera
            precip = float(np.random.exponential(6.0)) if np.random.random() < 0.25 else 0.0

            self.simula_giorno(doy, t_max, t_min, precip)

        return pd.DataFrame(self.log)


# ---------------------------------------------------------------------------
# Visualizzazione
# ---------------------------------------------------------------------------

def visualizza_twin(df: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("Digital Twin del Campo — Bilancio Idrico IoT Multi-Profondita\nPianura Padana | Mais (Zea mays L.)",
                 fontsize=13, fontweight="bold")

    giorni = range(len(df))
    colors = {10: "#2196F3", 30: "#4CAF50", 50: "#FF9800"}

    # --- Pannello 1: profilo umidita ---
    ax1 = axes[0]
    for depth, col in colors.items():
        ax1.plot(giorni, df[f"Umid_{depth}cm_pct"], color=col,
                 label=f"Sensore {depth} cm", linewidth=1.8, alpha=0.9)
    ax1.axhline(FIELD_CAPACITY[30], color="gray", linestyle="--", linewidth=1,
                label=f"Cap. di campo 30cm ({FIELD_CAPACITY[30]}%)")
    ax1.axhline(WILTING_POINT[30], color="#B71C1C", linestyle=":", linewidth=1,
                label=f"Punto appass. 30cm ({WILTING_POINT[30]}%)")
    ax1.set_ylabel("Umidita suolo (%)")
    ax1.set_title("Profilo di Umidita (Sensori IoT FDR/TDR a 10, 30, 50 cm)")
    ax1.legend(fontsize=8, ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(10, 65)

    # --- Pannello 2: ET0 e precipitazioni ---
    ax2 = axes[1]
    ax2.bar(giorni, df["Precip_mm"], color="#90CAF9", alpha=0.75,
            label="Precipitazione (mm)", width=0.85)
    ax2b = ax2.twinx()
    ax2b.plot(giorni, df["ET0_mm"], color="#E91E63", linewidth=1.8,
              label="ET0 (mm/gg)")
    ax2.set_ylabel("Precipitazione (mm)", color="#1565C0")
    ax2b.set_ylabel("ET0 Hargreaves (mm/gg)", color="#E91E63")
    ax2.set_title("Forzanti Idriche: Precipitazione vs Evapotraspirazione")
    lines1, lab1 = ax2.get_legend_handles_labels()
    lines2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Pannello 3: livello stress ---
    stress_map  = {"OTTIMALE": 0, "LIEVE": 1, "MODERATO": 2, "CRITICO": 3}
    stress_vals = df["Stress_30cm"].map(stress_map).fillna(0).values
    stress_cols = ["#4CAF50" if v == 0 else "#FF9800" if v == 1
                   else "#FF5722" if v == 2 else "#B71C1C" for v in stress_vals]
    ax3 = axes[2]
    ax3.bar(giorni, stress_vals + 0.6, color=stress_cols, alpha=0.85, width=0.9)
    ax3.set_yticks([0.3, 1.3, 2.3, 3.3])
    ax3.set_yticklabels(["Ottimale", "Lieve", "Moderato", "Critico"])
    ax3.set_xlabel("Giorni dalla semina")
    ax3.set_ylabel("Stress idrico (30 cm)")
    ax3.set_title("Classificazione Stress Idrico Giornaliero (sensore 30 cm)")
    ax3.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
    print(f"Grafico Digital Twin salvato -> {OUTPUT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Digital Twin del Campo IoT")
    parser.add_argument("--giorni",    type=int, default=90,
                        help="Numero di giorni da simulare (default: 90)")
    parser.add_argument("--doy_inizio", type=int, default=120,
                        help="DOY di inizio simulazione (default: 120 = 1 maggio)")
    args = parser.parse_args()

    print(f"Digital Twin del Campo — {args.giorni} giorni | DOY inizio: {args.doy_inizio}")
    print("Sensori attivi: 10 cm, 30 cm, 50 cm")

    twin = DigitalTwinCampo()
    df   = twin.simula_stagione(n_giorni=args.giorni, doy_inizio=args.doy_inizio)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Log IoT salvato -> {OUTPUT_CSV} ({len(df)} giorni)")

    visualizza_twin(df)

    print("\n=== RIEPILOGO DIGITAL TWIN ===")
    print(f"  Giorni simulati  : {len(df)}")
    print(f"  Umid. media 30cm : {df['Umid_30cm_pct'].mean():.1f}%")
    print(f"  ET0 media gg     : {df['ET0_mm'].mean():.2f} mm/gg")
    print(f"  Prec. totale     : {df['Precip_mm'].sum():.1f} mm")
    giorni_crit = (df["Stress_30cm"] == "CRITICO").sum()
    print(f"  Giorni critici   : {giorni_crit} ({giorni_crit/len(df)*100:.0f}% stagione)")
