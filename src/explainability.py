"""
Módulo de Explicabilidad e Interpretabilidad con SHAP para Deep Retention.
Permite identificar los factores determinantes globales de fuga y explicar
predicciones locales para gestores en sucursales.
"""

import os
import joblib
import numpy as np
import pandas as pd
import shap
import torch


def get_feature_importance_ranking(model, background_data: np.ndarray, feature_names: list, n_background: int = 70, n_samples: int = 150):
    """
    Calcula la importancia global de características mediante SHAP.
    Retorna una lista ordenada de (característica, importancia_media_absoluta).
    """
    model.eval()
    
    def model_predict_numpy(data_numpy):
        tensor_data = torch.tensor(data_numpy, dtype=torch.float32)
        with torch.no_grad():
            probs = model.predict_proba(tensor_data).cpu().numpy().flatten()
        return probs

    # Subconjunto de fondo representativo
    bg = background_data[:min(n_background, len(background_data))]
    explainer = shap.KernelExplainer(model_predict_numpy, bg)
    
    # Muestra para calcular valores SHAP
    eval_samples = background_data[:min(n_samples, len(background_data))]
    shap_values = explainer.shap_values(eval_samples, nsamples=100)
    
    # Importancia media absoluta por variable
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    ranking = sorted(zip(feature_names, mean_abs_shap), key=lambda x: x[1], reverse=True)
    
    return ranking, explainer


def explain_single_client(explainer, client_features: np.ndarray, feature_names: list):
    """
    Calcula los valores SHAP locales para un cliente específico.
    Retorna un diccionario estructurado de contribución por variable.
    """
    shap_vals = explainer.shap_values(client_features, nsamples=100)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
        
    vals = shap_vals.flatten()
    contributions = []
    for name, val in zip(feature_names, vals):
        contributions.append({
            'feature': name,
            'shap_value': float(val),
            'impact': 'Aumenta Riesgo' if val > 0 else 'Disminuye Riesgo'
        })
        
    # Ordenar por magnitud de impacto
    contributions = sorted(contributions, key=lambda x: abs(x['shap_value']), reverse=True)
    return contributions

