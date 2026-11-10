"""
Módulo de Redes Neuronales ("Deep Retention MLP").
Implementa la arquitectura recomendada en Enunciado.txt:
- Capa 1: 128 neuronas, ReLU, BatchNorm, Dropout(0.25).
- Capa 2: 64 neuronas, ReLU, Dropout(0.15).
- Capa de Salida: 1 neurona con Sigmoide.
- Manejo de desbalance mediante BCEWithLogitsLoss con pos_weight.
- Early Stopping con restauración de mejores pesos.
- Métricas: ROC-AUC, PR-AUC, Brier Score (Calibración).
"""

import os
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss
)


class DeepRetentionMLP(nn.Module):
    """
    Perceptrón Multicapa Profundo para predicción de riesgo de deserción bancaria.
    Diseñado específicamente según las especificaciones de Enunciado.txt.
    """
    def __init__(self, input_dim: int, hidden1: int = 128, hidden2: int = 64, drop1: float = 0.25, drop2: float = 0.15):
        super(DeepRetentionMLP, self).__init__()
        
        # Capa 1: Lineal + BatchNorm + ReLU + Dropout
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(drop1)
        
        # Capa 2: Lineal + ReLU + Dropout
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(drop2)
        
        # Capa de Salida: Emite logits (para estabilidad numérica en pérdida)
        self.out = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.drop2(x)
        
        logits = self.out(x)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Emite la probabilidad continua en el rango [0.0, 1.0]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.001,
    patience: int = 12,
    device: str = None
):
    """
    Entrena la red neuronal aplicando Early Stopping, ponderación de clases por desbalance
    y optimizador Adam con regularización L2 ligera.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    input_dim = X_train.shape[1]
    model = DeepRetentionMLP(input_dim=input_dim).to(device)

    # Cálculo del pos_weight para el tratamiento del desbalance
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    pos_weight_val = float(num_neg / max(1, num_pos))
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32).to(device)
    print(f"Desbalance detectado en Train -> Negativos: {num_neg}, Positivos: {num_pos} | pos_weight: {pos_weight_val:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Conversión a tensores y DataLoaders
    t_X_train = torch.tensor(X_train, dtype=torch.float32)
    t_y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(TensorDataset(t_X_train, t_y_train), batch_size=batch_size, shuffle=True)

    t_X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    t_y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    best_val_loss = float('inf')
    best_weights = None
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_roc_auc': []}

    print("\nIniciando entrenamiento con Early Stopping (patience=12)...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * batch_x.size(0)

        epoch_train_loss = running_train_loss / len(X_train)

        # Validación
        model.eval()
        with torch.no_grad():
            val_logits = model(t_X_val)
            val_loss = criterion(val_logits, t_y_val).item()
            val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_roc = roc_auc_score(y_val, val_probs)

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        history['val_roc_auc'].append(val_roc)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Época {epoch:03d}/{epochs:03d} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val ROC-AUC: {val_roc:.4f}")

        # Early Stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early Stopping activado en época {epoch}: {patience} épocas consecutivas sin mejora.")
                break

    # Restaurar mejores pesos
    if best_weights is not None:
        model.load_state_dict(best_weights)
        print("Mejores pesos restaurados con éxito.")

    return model, history


def evaluate_model(model: DeepRetentionMLP, X: np.ndarray, y: np.ndarray, device: str = None) -> dict:
    """
    Evalúa el desempeño predictivo integral y la calibración del modelo en un conjunto de datos.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model.eval()
    t_X = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        probs = model.predict_proba(t_X).cpu().numpy().flatten()

    preds = (probs >= 0.50).astype(int)

    metrics = {
        'roc_auc': float(roc_auc_score(y, probs)),
        'pr_auc': float(average_precision_score(y, probs)),
        'brier_score': float(brier_score_loss(y, probs)),
        'accuracy': float(accuracy_score(y, preds)),
        'precision': float(precision_score(y, preds, zero_division=0)),
        'recall': float(recall_score(y, preds, zero_division=0)),
        'f1_score': float(f1_score(y, preds, zero_division=0)),
    }
    return metrics, probs


def save_trained_model(model: DeepRetentionMLP, input_dim: int, save_path: str, metrics: dict = None, metrics_path: str = None):
    """Guarda el modelo entrenado y sus métricas asociadas."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        'input_dim': input_dim,
        'state_dict': model.state_dict()
    }
    torch.save(checkpoint, save_path)
    print(f"Modelo guardado en: {save_path}")

    if metrics and metrics_path:
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=4)
        print(f"Métricas guardadas en: {metrics_path}")


def load_trained_model(model_path: str, device: str = 'cpu') -> DeepRetentionMLP:
    """Carga un modelo previamente entrenado para canal de inferencia."""
    checkpoint = torch.load(model_path, map_location=device)
    input_dim = checkpoint['input_dim']
    model = DeepRetentionMLP(input_dim=input_dim)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model
