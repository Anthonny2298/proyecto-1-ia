"""
Modulo de Redes Neuronales ("Deep Retention MLP").
- Capa 1: 128 neuronas, BatchNorm, ReLU, Dropout(0.25).
- Capa 2: 64 neuronas, ReLU, Dropout(0.15).
- Capa de salida: 1 neurona (logit) + sigmoide para la probabilidad.
- Desbalance: BCEWithLogitsLoss con pos_weight.
- Early Stopping sobre val_loss con restauracion de los mejores pesos.
- Calibracion Platt (ajustada en validacion) guardada dentro del checkpoint, porque
  pos_weight desplaza las probabilidades y la regla financiera las usa directamente.
"""

import os
import json
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)

DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Fija todas las fuentes de aleatoriedad para que el entrenamiento sea reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DeepRetentionMLP(nn.Module):
    """Perceptron Multicapa para prediccion de riesgo de desercion bancaria."""

    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 64, drop1: float = 0.25, drop2: float = 0.15):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(drop1)

        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(drop2)

        self.out = nn.Linear(hidden2, 1)

        # Calibracion Platt: p = sigmoid(a * logit + b). Identidad hasta que se calibra.
        self.register_buffer('calib_a', torch.tensor(1.0))
        self.register_buffer('calib_b', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve logits sin calibrar (usados por la funcion de perdida y por SHAP)."""
        x = self.drop1(self.relu1(self.bn1(self.fc1(x))))
        x = self.drop2(self.relu2(self.fc2(x)))
        return self.out(x)

    def predict_proba(self, x: torch.Tensor, calibrated: bool = True) -> torch.Tensor:
        """Probabilidad de fuga en [0, 1]. Calibrada por defecto."""
        with torch.no_grad():
            logits = self.forward(x)
            if calibrated:
                logits = self.calib_a * logits + self.calib_b
            return torch.sigmoid(logits)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 120,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 14,
    seed: int = DEFAULT_SEED,
    device: str = None,
    verbose: bool = True,
):
    """
    Entrena la red con Early Stopping, ponderacion de clases y Adam con L2 ligera.
    Devuelve el modelo (en CPU, con los mejores pesos) y el historial.
    """
    set_seed(seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = DeepRetentionMLP(input_dim=X_train.shape[1]).to(device)

    num_neg = float(np.sum(y_train == 0))
    num_pos = float(np.sum(y_train == 1))
    pos_weight_val = num_neg / max(1.0, num_pos)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
    if verbose:
        print(f"Desbalance en Train -> Negativos: {int(num_neg)}, Positivos: {int(num_pos)} | pos_weight: {pos_weight_val:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)),
        batch_size=batch_size, shuffle=True, generator=generator
    )
    t_X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    t_y_val = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    best_val_loss = float('inf')
    best_weights = None
    best_epoch = 0
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_roc_auc': []}

    if verbose:
        print(f"\nEntrenando con Early Stopping (patience={patience})...")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running += loss.item() * batch_x.size(0)
        train_loss = running / len(X_train)

        model.eval()
        with torch.no_grad():
            val_logits = model(t_X_val)
            val_loss = criterion(val_logits, t_y_val).item()
            val_roc = roc_auc_score(y_val, torch.sigmoid(val_logits).cpu().numpy().ravel())

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_roc_auc'].append(float(val_roc))

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoca {epoch:03d}/{epochs:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_roc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"Early Stopping en epoca {epoch}: {patience} epocas sin mejora. Mejor epoca: {best_epoch}.")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    history['best_epoch'] = best_epoch
    model.to('cpu').eval()
    return model, history


def predict_logits(model: DeepRetentionMLP, X: np.ndarray) -> np.ndarray:
    """Logits sin calibrar para un conjunto de datos (en CPU)."""
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).cpu().numpy().ravel()


def predict_proba(model: DeepRetentionMLP, X: np.ndarray, calibrated: bool = True) -> np.ndarray:
    """Probabilidades de fuga para un conjunto de datos (en CPU)."""
    model.eval()
    return model.predict_proba(torch.tensor(X, dtype=torch.float32), calibrated=calibrated).cpu().numpy().ravel()


def calibrate_model(model: DeepRetentionMLP, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    """
    Ajusta Platt scaling sobre los logits de validacion y lo guarda en el modelo.
    Es una transformacion monotona: no cambia el ROC-AUC, solo corrige la escala.
    """
    logits = predict_logits(model, X_val).reshape(-1, 1)
    platt = LogisticRegression(C=1e6).fit(logits, y_val)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])
    model.calib_a.fill_(a)
    model.calib_b.fill_(b)
    return {'a': a, 'b': b}


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """ECE: diferencia promedio (ponderada) entre probabilidad predicha y frecuencia observada."""
    y = np.asarray(y, dtype=float)
    probs = np.asarray(probs, dtype=float)
    idx = np.clip((probs * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += abs(probs[mask].mean() - y[mask].mean()) * mask.mean()
    return float(ece)


def compute_metrics(y: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Metricas de discriminacion y calibracion."""
    preds = (probs >= threshold).astype(int)
    return {
        'roc_auc': float(roc_auc_score(y, probs)),
        'pr_auc': float(average_precision_score(y, probs)),
        'brier_score': float(brier_score_loss(y, probs)),
        'ece': expected_calibration_error(y, probs),
        'mean_predicted': float(np.mean(probs)),
        'observed_rate': float(np.mean(y)),
        'accuracy': float(accuracy_score(y, preds)),
        'precision': float(precision_score(y, preds, zero_division=0)),
        'recall': float(recall_score(y, preds, zero_division=0)),
        'f1_score': float(f1_score(y, preds, zero_division=0)),
    }


def evaluate_model(model: DeepRetentionMLP, X: np.ndarray, y: np.ndarray, calibrated: bool = True):
    """Evalua el modelo y devuelve (metricas, probabilidades)."""
    probs = predict_proba(model, X, calibrated=calibrated)
    return compute_metrics(y, probs), probs


def save_trained_model(model: DeepRetentionMLP, input_dim: int, save_path: str, metrics: dict = None, metrics_path: str = None):
    """Guarda pesos (incluida la calibracion) y metricas."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({'input_dim': input_dim, 'state_dict': model.state_dict()}, save_path)
    print(f"Modelo guardado en: {save_path}")

    if metrics and metrics_path:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metricas guardadas en: {metrics_path}")


def load_trained_model(model_path: str, device: str = 'cpu') -> DeepRetentionMLP:
    """Carga un modelo entrenado (con su calibracion) para el canal de inferencia."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = DeepRetentionMLP(input_dim=checkpoint['input_dim'])
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()
    return model
