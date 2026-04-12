import torch
import numpy as np
import pandas as pd
# no import needed

# ... wait, let me just write the script standalone
import joblib
from model import ET0Predictor

df = pd.read_excel("dati_meteo_agricoli.xlsx", sheet_name="Dati_Meteo_Giornalieri")
FEATURE_COLS = ["T_max_C", "T_min_C", "Umidita_Relativa_%", "Rad_Solare_MJ_m2", "Rad_Extraterr_MJ_m2", "Velocita_Vento_m_s"]
X = df[FEATURE_COLS].values.astype(np.float32)

scaler_X = joblib.load("scaler_X.pkl")
X_scaled = scaler_X.transform(X)
X_t = torch.tensor(X_scaled, dtype=torch.float32)

model = ET0Predictor(input_dim=6)
checkpoint = torch.load("model_et0.pth")
model.load_state_dict(checkpoint["model_state"])
model.eval()

# Pertubation Feature Importance
baseline_pred = model(X_t).detach().numpy()
importances = {}
for i, col in enumerate(FEATURE_COLS):
    X_perm = X_t.clone()
    # Shuffle column i
    idx = torch.randperm(X_perm.shape[0])
    X_perm[:, i] = X_perm[idx, i]
    pred_perm = model(X_perm).detach().numpy()
    mae_diff = np.mean(np.abs(baseline_pred - pred_perm))
    importances[col] = mae_diff

for col, imp in importances.items():
    print(f"{col}: {imp:.4f} mae impact")
