import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import joblib

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from model import ET0Predictor  # noqa: E402
from generate_dataset import calcola_Ra  # noqa: E402

MODEL_FILE  = ROOT / "model_et0.pth"
SCALER_FILE = ROOT / "scaler_X.pkl"
OUTPUT_PNG  = ROOT / "vra_heatmap.png"

class VRAIrrigationSimulator:
    """
    Simulatore per Variable Rate Application (VRA) dell'irrigazione.
    Genera un campo agricolo sintetico 2D con variabilita' microclimatica
    locale e utilizza la rete neurale per inferire la mappa ET0 ad alta risoluzione.
    """
    
    def __init__(self, grid_dim: int = 50, cell_size: int = 10, giorno: int = 210, soglia_stress: float = 5.0):
        self.grid_dim = grid_dim
        self.cell_size = cell_size
        self.giorno = giorno
        self.soglia_stress = soglia_stress
        self.n_celle = self.grid_dim * self.grid_dim

        print(f"\n[1/4] Inizializzazione VRA Simulatore — Campo {self.grid_dim}x{self.grid_dim}")
        self._carica_modello_scaler()

    def _carica_modello_scaler(self):
        """Carica la rete neurale PyTorch e lo scaler pre-addestrato."""
        if not SCALER_FILE.exists() or not MODEL_FILE.exists():
            raise FileNotFoundError("Mancano model_et0.pth o scaler_X.pkl. Esegui prima train.py!")

        self.scaler = joblib.load(SCALER_FILE)
        self.model = ET0Predictor(input_dim=6)
        
        checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    def _leggi_parametri_base(self):
        """Temperatura ed umidita' di base per il giorno dell'anno."""
        t_max_base = 32.0 + 3 * np.sin(2 * np.pi * self.giorno / 365)
        t_min_base = 20.0 + 2 * np.sin(2 * np.pi * self.giorno / 365)
        ur_base    = 50.0 - 10 * np.cos(2 * np.pi * self.giorno / 365)
        rad_base   = 22.0
        vento_base = 2.0
        return t_max_base, t_min_base, ur_base, rad_base, vento_base
        
    def _genera_griglia(self):
        """Genera i tensori 2D di rumorosita' microclimatica tramite random walk."""
        x = np.linspace(0, 10, self.grid_dim)
        y = np.linspace(0, 10, self.grid_dim)
        Xg, Yg = np.meshgrid(x, y)

        elevazione = np.sin(Xg*0.5) * np.cos(Yg*0.5)
        lai = np.sin(Xg*0.8 + 2) * np.sin(Yg*0.8)

        t_max_noise = elevazione * 1.5 - lai * 0.5 
        t_min_noise = elevazione * 0.8
        ur_noise    = lai * 5.0 - elevazione * 3.0
        
        return t_max_noise, t_min_noise, ur_noise

    def genera_vra(self):
        """
        Calcola la mappa di prescrizione idrica, restituendo Figure e Statistiche.
        """
        t_max_base, t_min_base, ur_base, rad_base, vento_base = self._leggi_parametri_base()
        t_max_noise, t_min_noise, ur_noise = self._genera_griglia()
        
        T_max_2d = t_max_base + t_max_noise
        T_min_2d = t_min_base + t_min_noise
        UR_2d    = np.clip(ur_base + ur_noise, 5.0, 100.0)
        
        Ra_scalar = float(calcola_Ra(np.array([self.giorno]))[0])

        T_max_flat = T_max_2d.flatten()
        T_min_flat = T_min_2d.flatten()
        UR_flat    = UR_2d.flatten()
        
        X_batch = np.column_stack([
            T_max_flat,
            T_min_flat,
            UR_flat,
            np.full(self.n_celle, rad_base),
            np.full(self.n_celle, Ra_scalar),
            np.full(self.n_celle, vento_base)
        ]).astype(np.float32)
        
        print("\n[3/4] Inferenza Rete Neurale...")
        t0 = time.time()
        X_scaled = self.scaler.transform(X_batch)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            pred_flat = self.model(X_tensor).numpy().flatten()
            
        ET0_flat = np.maximum(pred_flat, 0.0)
        infer_time = time.time() - t0
        print(f"      Processate {self.n_celle} celle in {infer_time*1000:.1f} ms")

        ET0_2d = ET0_flat.reshape(self.grid_dim, self.grid_dim)
        
        dose_irrigua = ET0_2d * 1.2
        dose_flat = dose_irrigua.flatten()
        
        celle_stress = np.sum(dose_flat > self.soglia_stress)
        percentuale_stress = (celle_stress / self.n_celle) * 100

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(dose_irrigua, cmap="YlGnBu", origin="lower",
                       extent=[0, self.grid_dim * self.cell_size, 0, self.grid_dim * self.cell_size])
        
        ax.set_title(f"Mappa VRA - Giorno {self.giorno} (Risoluzione {self.cell_size}m)", fontsize=14, pad=15)
        ax.set_xlabel("Coord X (m)")
        ax.set_ylabel("Coord Y (m)")
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Fabbisogno Idrico Stimato (mm/gg)", rotation=270, labelpad=15)

        stats = {
            "et0_min": float(ET0_2d.min()),
            "et0_max": float(ET0_2d.max()),
            "et0_mean": float(ET0_2d.mean()),
            "perc_stress": float(percentuale_stress),
            "dose_media": float(dose_irrigua.mean())
        }
        
        textstr = (
            f"ET0 Media: {stats['et0_mean']:.2f} mm\n"
            f"Min: {stats['et0_min']:.2f} mm | Max: {stats['et0_max']:.2f} mm\n"
            f"Zone critiche (>{self.soglia_stress}mm): {stats['perc_stress']:.1f}%"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.03, 0.96, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        return fig, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variable Rate Application (VRA) Heatmap con PyTorch")
    parser.add_argument("--giorno", type=int, default=210, help="Giorno dell'anno (default: 210 = fine luglio)")
    parser.add_argument("--dim", type=int, default=50, help="Dimensione griglia NxN (default: 50)")
    parser.add_argument("--soglia-stress", type=float, default=5.0, dest="soglia_stress", 
                        help="Soglia ET0 per considerare la cella in stress idrico")
    args = parser.parse_args()

    simulator = VRAIrrigationSimulator(grid_dim=args.dim, giorno=args.giorno, soglia_stress=args.soglia_stress)
    fig, stats = simulator.genera_vra()

    fig.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
    print(f"\n[4/4] Mappa calcolata e salvata -> '{OUTPUT_PNG}'")
    print("-" * 50)
    print(f"Fabbisogno medio del campo : {stats['dose_media']:.2f} mm")
    print(f"Celle in zona critica      : {stats['perc_stress']:.1f}% (> {args.soglia_stress} mm)")
