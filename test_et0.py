import torch
import numpy as np
from model import ET0Predictor
from generate_dataset import calcola_Ra, hargreaves_samani

def test_model_architecture():
    """Verifica l'inizializzazione e le dimensioni I/O del modello PyTorch."""
    model = ET0Predictor(input_dim=6)
    assert isinstance(model, torch.nn.Module), "Il modello deve eredare da nn.Module"
    
    # Test Forward Pass
    dummy_input = torch.randn(10, 6)
    out = model(dummy_input)
    assert out.shape == (10, 1), f"Shape dell'output errata. Prevista (10,1), trovata: {out.shape}"
    assert out.dtype == torch.float32, "Il tipo di output deve essere float32"

def test_hargreaves_samani():
    """Verifica che la stima di Hargreaves-Samani non fallisca e restituisca >= 0."""
    T_max = np.array([30.0, 15.0])
    T_min = np.array([20.0, -5.0])
    Ra = np.array([15.0, 5.0])
    
    et0 = hargreaves_samani(T_max, T_min, Ra)
    assert len(et0) == 2, "La shape di output non corrisponde all'input"
    assert np.all(et0 >= 0), "ET0 Hargreaves non può essere negativa"

def test_calcola_ra():
    """Verifica che i valori di radiazione extraterrestre siano coerenti col periodo."""
    giorni_estate = np.array([180]) # Fine Giugno
    giorni_inverno = np.array([360]) # Fine Dicembre
    
    ra_estate = calcola_Ra(giorni_estate)
    ra_inverno = calcola_Ra(giorni_inverno)
    
    # Ra in estate nell'emisfero Nord (es. 45°N) deve essere molto più alta dell'inverno
    assert ra_estate[0] > ra_inverno[0], "Ra in estate dovrebbe essere maggiore di Ra in inverno a nord"
