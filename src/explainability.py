"""
Modulo de Explicabilidad con SHAP para Deep Retention.

Se usa shap.GradientExplainer sobre la red (explica el logit de fuga). La calibracion Platt
es una transformacion afin y monotona del logit, por lo que no altera el orden ni el signo
de las contribuciones. Las columnas one-hot se suman para reportar por variable original
(por ejemplo Geography en lugar de Geography_France / Geography_Germany / Geography_Spain).
"""

import numpy as np
import pandas as pd
import shap
import torch

from preprocessing import original_feature_name


def build_explainer(model, background: np.ndarray):
    """Crea el explicador SHAP con un fondo representativo del conjunto de entrenamiento."""
    model.eval()
    return shap.GradientExplainer(model, torch.tensor(background, dtype=torch.float32))


def shap_values(explainer, X: np.ndarray) -> np.ndarray:
    """Valores SHAP con forma (n_clientes, n_variables), sin importar la version de shap."""
    X = np.atleast_2d(X)
    values = explainer.shap_values(torch.tensor(X, dtype=torch.float32))
    if isinstance(values, list):
        values = values[0]
    return np.asarray(values).reshape(len(X), -1)


def aggregate_by_original_feature(values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """Suma las contribuciones de columnas one-hot de una misma variable original."""
    frame = pd.DataFrame(np.atleast_2d(values), columns=feature_names)
    return frame.T.groupby([original_feature_name(n) for n in feature_names], sort=False).sum().T


def global_importance(explainer, X: np.ndarray, feature_names: list) -> pd.Series:
    """Importancia global = media de |SHAP| por variable original, ordenada de mayor a menor."""
    agg = aggregate_by_original_feature(shap_values(explainer, X), feature_names)
    return agg.abs().mean().sort_values(ascending=False)


def explain_single_client(explainer, client_features: np.ndarray, feature_names: list) -> list:
    """Contribucion SHAP local de cada variable original para un cliente, ordenada por magnitud."""
    agg = aggregate_by_original_feature(shap_values(explainer, client_features), feature_names).iloc[0]
    contributions = [
        {'feature': name, 'shap_value': float(val),
         'impact': 'Aumenta Riesgo' if val > 0 else 'Disminuye Riesgo'}
        for name, val in agg.items()
    ]
    return sorted(contributions, key=lambda c: abs(c['shap_value']), reverse=True)
