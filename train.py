"""
train.py
--------
Fase 3: Addestramento e Visualizzazione - ET0 Predictor

Pipeline:
  1. Caricamento dataset Excel
  2. Preprocessing (StandardScaler, split cronologico train/test)
  3. Training con MSELoss e Adam optimizer
  4. Valutazione: R2, MAE, RMSE
  5. Grafico confronto ET0 reale vs predetta -> risultati_modello.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

from model import ET0Predictor
from generate_dataset import calcola_Ra, hargreaves_samani

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------

SEED          = 42
BATCH_SIZE    = 32
EPOCHS        = 200
LEARNING_RATE = 1e-3
TEST_SIZE     = 0.20  # split random stratificato per stagione
DATASET_FILE  = "dati_meteo_agricoli.xlsx"
OUTPUT_PNG    = "risultati_modello.png"
MODEL_FILE    = "model_et0.pth"
SCALER_FILE   = "scaler_X.pkl"

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ---------------------------------------------------------------------------
# 1. Caricamento e Preprocessing
# ---------------------------------------------------------------------------

print("\n[1/4] Caricamento dataset...")
df = pd.read_excel(DATASET_FILE, sheet_name="Dati_Meteo_Giornalieri")

FEATURE_COLS = [
    "T_max_C",
    "T_min_C",
    "Umidita_Relativa_%",
    "Rad_Solare_MJ_m2",
    "Rad_Extraterr_MJ_m2",   # Ra: vettore stagionale astronomico, chiave per ET0_HS
    "Velocita_Vento_m_s"
]
TARGET_COL = "ET0_Hargreaves_mm"

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)

# Split random stratificato per stagione: garantisce che train e test
# contengano campioni da tutte le stagioni dell'anno, evitando distribution
# mismatch tipico dello split cronologico su dataset con forte stagionalita'.
idx = np.arange(len(X))
idx_train, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)
idx_test_sorted = np.sort(idx_test)   # ordine cronologico per il grafico

X_train, X_test = X[idx_train], X[idx_test_sorted]
y_train, y_test = y[idx_train], y[idx_test_sorted]
dates_test = df["Data"].values[idx_test_sorted]

# StandardScaler: fit solo su train (evita data leakage dal test)
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test  = scaler_X.transform(X_test)

# Nessuno scaling su y: la rete predice direttamente in mm/giorno.
# Questo evita il problema di Softplus/linear output con target negativi
# nel feature space normalizzato.
y_train_scaled = y_train
y_test_scaled  = y_test

n_test = len(X_test)
print(f"Train: {len(X_train)} giorni | Test: {n_test} giorni")
print(f"ET0 range train: [{y_train.min():.2f}, {y_train.max():.2f}] mm/giorno")

# Tensori PyTorch
def to_tensor(arr):
    return torch.tensor(arr, dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(to_tensor(X_train), to_tensor(y_train_scaled).unsqueeze(1)),
    batch_size=BATCH_SIZE, shuffle=True
)

# ---------------------------------------------------------------------------
# 2. Modello, Loss, Optimizer
# ---------------------------------------------------------------------------

print("\n[2/4] Inizializzazione modello...")
model     = ET0Predictor(input_dim=6).to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------

print(f"\n[3/4] Addestramento ({EPOCHS} epoche)...")
history = {"train_loss": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()

        # Gradient clipping per stabilita'
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        epoch_loss += loss.item() * len(y_batch)

    scheduler.step()
    avg_loss = epoch_loss / len(X_train)
    history["train_loss"].append(avg_loss)

    if epoch % 20 == 0 or epoch == 1:
        print(f"  Epoca {epoch:3d}/{EPOCHS} | MSE Loss: {avg_loss:.5f}")

# ---------------------------------------------------------------------------
# 4. Valutazione sul Test Set
# ---------------------------------------------------------------------------

print("\n[4/4] Valutazione sul test set...")
model.eval()
with torch.no_grad():
    X_test_t = to_tensor(X_test).to(DEVICE)
    pred_scaled = model(X_test_t).cpu().numpy().flatten()

# La rete predice gia' in mm/giorno (nessuna de-normalizzazione necessaria)
y_pred = np.maximum(pred_scaled, 0.0)   # ET0 >= 0 per coerenza fisica

# Metriche di regressione
r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n=== METRICHE SUL TEST SET ({n_test} giorni) ===")
print(f"  R2 Score  : {r2:.4f}   (1.0 = perfetto)")
print(f"  MAE       : {mae:.3f} mm/giorno")
print(f"  RMSE      : {rmse:.3f} mm/giorno")
print(f"  Errore %  : {(mae / y_test.mean() * 100):.1f}% dell'ET0 media")

# ---------------------------------------------------------------------------
# 5. Visualizzazione - risultati_modello.png
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 10))
fig.suptitle(
    "ET0 Predictor — PyTorch | Pianura Padana 2024",
    fontsize=15, fontweight="bold", y=0.98
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# Palette colori
C_REAL = "#2196F3"    # blu: ET0 reale (Hargreaves-Samani)
C_PRED = "#FF5722"    # arancio: ET0 predetta (Neural Network)
C_LOSS = "#4CAF50"    # verde: curva di loss

# ── Pannello 1: Serie temporale ET0 reale vs predetta ───────────────────────
ax1 = fig.add_subplot(gs[0, :])
x_axis = np.arange(len(y_test))

ax1.fill_between(x_axis, y_test, alpha=0.15, color=C_REAL)
ax1.plot(x_axis, y_test, color=C_REAL, linewidth=1.8,
         label="ET0 reale (Hargreaves-Samani)", zorder=3)
ax1.plot(x_axis, y_pred, color=C_PRED, linewidth=1.8, linestyle="--",
         label="ET0 predetta (Neural Network)", zorder=4)

ax1.set_title(f"Confronto ET0 Reale vs Predetta — {n_test} Giorni di Test (spread su 3 anni)",
              fontsize=12, pad=8)
ax1.set_xlabel("Giorno del periodo di test", fontsize=10)
ax1.set_ylabel("ET0 (mm/giorno)", fontsize=10)
ax1.legend(fontsize=10, loc="upper right")
ax1.grid(True, alpha=0.3)
ax1.annotate(f"R2={r2:.3f}  MAE={mae:.3f} mm  RMSE={rmse:.3f} mm",
             xy=(0.02, 0.92), xycoords="axes fraction",
             fontsize=10, color="black",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", alpha=0.8))

# ── Pannello 2: Scatter ET0 reale vs predetta ───────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.scatter(y_test, y_pred, alpha=0.65, color=C_PRED, edgecolors="white",
            linewidth=0.5, s=45, zorder=3)

lim_min = min(y_test.min(), y_pred.min()) - 0.2
lim_max = max(y_test.max(), y_pred.max()) + 0.2
ax2.plot([lim_min, lim_max], [lim_min, lim_max],
         color=C_REAL, linewidth=2, linestyle="-", label="Predizione perfetta", zorder=2)

ax2.set_title("ET0 Reale vs Predetta (Scatter)", fontsize=12, pad=8)
ax2.set_xlabel("ET0 reale (mm/giorno)", fontsize=10)
ax2.set_ylabel("ET0 predetta (mm/giorno)", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.88, f"R2 = {r2:.4f}", transform=ax2.transAxes,
         fontsize=11, color="black",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

# ── Pannello 3: Curva di Loss ────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(range(1, EPOCHS + 1), history["train_loss"],
         color=C_LOSS, linewidth=2)
ax3.fill_between(range(1, EPOCHS + 1), history["train_loss"],
                 alpha=0.15, color=C_LOSS)
ax3.set_title("Curva di Loss (MSE) durante il Training", fontsize=12, pad=8)
ax3.set_xlabel("Epoca", fontsize=10)
ax3.set_ylabel("MSE Loss (normalizzata)", fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.annotate(f"Loss finale: {history['train_loss'][-1]:.5f}",
             xy=(EPOCHS * 0.6, history["train_loss"][-1] * 1.8),
             fontsize=9, color="black",
             bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
print(f"\nGrafico salvato -> '{OUTPUT_PNG}'")

# ---------------------------------------------------------------------------
# 6. Demo: predizione su nuove letture meteo
# ---------------------------------------------------------------------------

print("\n=== DEMO: Predizione ET0 su Nuove Letture Meteo ===")
print(f"{'Scenario':<30} {'ET0 Reale (HS)':<18} {'ET0 Predetta (NN)'}")
print("-" * 65)

_Ra_demo = calcola_Ra(np.array([197, 15, 105]))

nuovi = pd.DataFrame({
    "T_max_C":               [35.0,  8.0, 22.0],
    "T_min_C":               [22.0,  1.0, 10.0],
    "Umidita_Relativa_%":    [45.0, 82.0, 65.0],
    "Rad_Solare_MJ_m2":      [22.0,  4.0, 14.0],
    "Rad_Extraterr_MJ_m2":   _Ra_demo.tolist(),
    "Velocita_Vento_m_s":    [ 2.5,  1.2,  3.0],
})
scenari = ["Estate calda e secca", "Inverno freddo e umido", "Primavera temperata"]

giorni_ref = np.array([197, 15, 105])   # luglio, gennaio, aprile
Ra_ref = calcola_Ra(giorni_ref)
et0_reale_demo = hargreaves_samani(
    nuovi["T_max_C"].values,
    nuovi["T_min_C"].values,
    Ra_ref
)

X_nuovi = scaler_X.transform(nuovi.values.astype(np.float32))
X_nuovi_t = torch.tensor(X_nuovi, dtype=torch.float32).to(DEVICE)

model.eval()
with torch.no_grad():
    pred_nuovi_sc = model(X_nuovi_t).cpu().numpy().flatten()
et0_pred_demo = np.maximum(pred_nuovi_sc, 0.0)

for sc, er, ep in zip(scenari, et0_reale_demo, et0_pred_demo):
    print(f"  {sc:<28} {er:>6.2f} mm/gg      {ep:>6.2f} mm/gg")

# ---------------------------------------------------------------------------
# 7. Salvataggio modello e scaler (per riuso in app.py)
# ---------------------------------------------------------------------------

print("\n[Salvataggio] Esportazione modello e scaler...")
torch.save({
    "model_state":  model.state_dict(),
    "input_dim":    6,
    "r2":           r2,
    "mae":          mae,
    "rmse":         rmse,
}, MODEL_FILE)
joblib.dump(scaler_X, SCALER_FILE)
print(f"  Modello salvato -> '{MODEL_FILE}'")
print(f"  Scaler salvato  -> '{SCALER_FILE}'")
print("\nPuoi ora avviare la demo con: streamlit run app.py")
