"""
model.py
--------
Fase 2: Definizione del Modello PyTorch per Regressione ET0

Rete Neurale Feedforward che apprende la relazione non lineare tra
i parametri meteorologici e l'Evapotraspirazione di riferimento (ET0).

Il modello apprende implicitamente la struttura della formula di
Hargreaves-Samani partendo dai dati grezzi, senza conoscerla a priori.

Input:  6 feature (T_max, T_min, Umidita', Rad_Solare, Rad_Extraterr, Vento)
Output: 1 valore continuo -> ET0 (mm/giorno)
"""

import torch
import torch.nn as nn


class ET0Predictor(nn.Module):
    """
    Rete neurale feed-forward per la predizione dell'ET0 (regressione).

    Architettura:
        Input(6) -> FC(128) -> BN -> ReLU -> Dropout
                 -> FC(64)  -> BN -> ReLU -> Dropout
                 -> FC(32)  -> ReLU
                 -> FC(1)   [output lineare, ET0 >= 0 applicato con np.maximum in post]

    La non-negativita' di ET0 e' garantita in post-processing con np.maximum(pred, 0)
    anziche' con Softplus, per semplicita' e stabilita' del gradiente.
    """

    def __init__(self, input_dim: int = 6, dropout_rate: float = 0.2):
        super(ET0Predictor, self).__init__()

        self.network = nn.Sequential(
            # Primo blocco: cattura relazioni tra variabili meteorologiche
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # Secondo blocco: rappresentazioni intermedie
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # Terzo blocco: compressione
            nn.Linear(64, 32),
            nn.ReLU(),

            # Quarto blocco: output lineare (regressione pura)
            # L'output e' in mm/giorno; il vincolo ET0>=0 viene applicato
            # dopo la predizione con np.maximum(pred, 0)
            nn.Linear(32, 1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Inizializzazione Kaiming per layer con ReLU."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns:
            Tensore di shape (batch, 1) con i valori ET0 predetti (mm/giorno)
        """
        return self.network(x)


if __name__ == "__main__":
    model = ET0Predictor()
    print("Architettura ET0Predictor:")
    print(model)
    print(f"\nParametri totali: {sum(p.numel() for p in model.parameters()):,}")

    dummy = torch.randn(16, 6)
    out = model(dummy)
    print(f"\nInput:  {dummy.shape}")
    print(f"Output: {out.shape}  -> valori ET0 (mm/giorno, non-negativita' applicata con np.maximum in post-processing)")
    print(f"Output esempio: {out[:4].detach().numpy().flatten()}")
