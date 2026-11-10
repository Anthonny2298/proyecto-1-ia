"""
Pipeline de Entrenamiento y Evaluación de Deep Retention.
Ejecuta el flujo completo de:
1. Carga de datos y partición estratificada.
2. Preprocesamiento (Escalado y OHE) sin fuga.
3. Entrenamiento de Red Neuronal MLP con Early Stopping y balanceo.
4. Evaluación rigurosa (ROC-AUC, PR-AUC, Brier/Calibración).
5. Interpretabilidad SHAP (Factores globales de deserción).
6. Análisis de Cartera Financiera y cálculo de ROI.
7. Exportación de artefactos para el canal de inferencia.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch

# Asegurar importación de módulos locales en src
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import (
    load_raw_data,
    split_data,
    build_and_fit_preprocessor,
    transform_data,
    get_feature_names,
    DROP_COLUMNS,
    TARGET_COLUMN
)
from model import (
    DeepRetentionMLP,
    train_model,
    evaluate_model,
    save_trained_model
)
from financial_logic import analyze_portfolio, DEFAULT_FINANCIAL_PARAMS
from explainability import get_feature_importance_ranking


def run_pipeline():
    print("="*70)
    print("INICIANDO PIPELINE DE ENTRENAMIENTO - SISTEMA DEEP RETENTION")
    print("="*70)

    base_dir = os.path.dirname(__file__)
    data_path = os.path.join(base_dir, 'data', 'Churn_Modelling.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # 1. Carga de Datos
    print(f"\n[1/6] Cargando datos desde {data_path}...")
    df = load_raw_data(data_path)
    print(f"Total registros cargados: {len(df):,} clientes.")

    # 2. Partición Estratificada
    print("\n[2/6] Realizando partición estratificada (70% Train, 15% Val, 15% Test)...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df, test_size=0.15, val_size=0.15, random_state=42)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # 3. Preprocesamiento (Ajustado únicamente en Train)
    print("\n[3/6] Ajustando transformaciones (StandardScaler + OneHotEncoder)...")
    preproc_path = os.path.join(models_dir, 'preprocessor.joblib')
    preprocessor = build_and_fit_preprocessor(X_train, save_path=preproc_path)
    feature_names = get_feature_names(preprocessor)
    print(f"Total variables de entrada generadas: {len(feature_names)}")
    print(f"Variables: {feature_names}")

    X_train_proc = transform_data(preprocessor, X_train)
    X_val_proc = transform_data(preprocessor, X_val)
    X_test_proc = transform_data(preprocessor, X_test)

    # 4. Entrenamiento de la Red Neuronal
    print("\n[4/6] Entrenando Perceptrón Multicapa (MLP) con regularización y Early Stopping...")
    model, history = train_model(
        X_train_proc, y_train,
        X_val_proc, y_val,
        epochs=120,
        batch_size=64,
        lr=0.001,
        patience=14
    )

    # Evaluación en Test Set
    print("\n[5/6] Evaluando métricas diagnósticas en conjunto de Test...")
    test_metrics, test_probs = evaluate_model(model, X_test_proc, y_test)
    print("\n--- RESULTADOS DIAGNÓSTICOS EN TEST SET ---")
    for k, v in test_metrics.items():
        print(f"  {k.upper()}: {v:.4f}")

    # Guardar modelo entrenado y métricas
    model_path = os.path.join(models_dir, 'deep_retention_mlp.pt')
    metrics_path = os.path.join(models_dir, 'test_metrics.json')
    save_trained_model(
        model=model,
        input_dim=len(feature_names),
        save_path=model_path,
        metrics=test_metrics,
        metrics_path=metrics_path
    )

    # Inferencia en el dataset completo para consolidación financiera
    print("\n[6/6] Ejecutando inferencia completa y análisis de rentabilidad financiera...")
    X_full = df.drop(columns=DROP_COLUMNS + [TARGET_COLUMN], errors='ignore')
    X_full_proc = transform_data(preprocessor, X_full)
    
    with torch.no_grad():
        t_full = torch.tensor(X_full_proc, dtype=torch.float32)
        full_probs = model.predict_proba(t_full).cpu().numpy().flatten()

    df_analyzed, financial_summary = analyze_portfolio(df, full_probs)

    print("\n--- RESUMEN FINANCIERO Y DE IMPACTO (CARTERA TOTAL) ---")
    print(f"  Clientes Totales Analizados: {financial_summary['total_clients']:,}")
    print(f"  Capital en Riesgo Crítico: ${financial_summary['total_balance_at_risk']:,.2f}")
    print(f"  Capital Rescatable Estimado: ${financial_summary['rescuable_balance']:,.2f}")
    print(f"  Inversión Total en Retención: ${financial_summary['total_investment']:,.2f}")
    print(f"  Beneficio Neto Esperado: ${financial_summary['total_net_gain']:,.2f}")
    print(f"  Retorno Sobre la Inversión (ROI): {financial_summary['roi_percent']:.2f}%")
    print(f"  Clientes Intervenidos Rentablemente: {financial_summary['num_intervened']:,}")
    print(f"  Clientes con Salida Permitida (Filtro Margen/Deficitarios): {financial_summary['num_allowed_exit']:,}")
    print(f"  Clientes Estables: {financial_summary['num_stable']:,}")

    # Guardar resumen financiero y dataset enriquecido
    fin_summary_path = os.path.join(models_dir, 'financial_summary.json')
    with open(fin_summary_path, 'w', encoding='utf-8') as f:
        json.dump(financial_summary, f, indent=4)

    scored_data_path = os.path.join(base_dir, 'data', 'portfolio_scored.csv')
    df_analyzed.to_csv(scored_data_path, index=False)
    print(f"Dataset consolidado guardado en: {scored_data_path}")

    # Análisis SHAP de Explicabilidad Global
    print("\n[EXTRA] Calculando factores determinantes de abandono con SHAP...")
    bg_samples = X_train_proc[:100]
    ranking, explainer = get_feature_importance_ranking(model, bg_samples, feature_names, n_background=70, n_samples=100)
    
    top_factors = ranking[:5]
    print("\nTop Factores Determinantes de Fuga según SHAP:")
    for rank_idx, (feat, score) in enumerate(top_factors, 1):
        print(f"  {rank_idx}. {feat}: impacto medio |SHAP| = {score:.4f}")

    shap_summary = {
        'feature_names': feature_names,
        'ranking': [{'feature': f, 'importance': float(s)} for f, s in ranking],
        'top_factors': [{'rank': i+1, 'feature': f, 'importance': float(s)} for i, (f, s) in enumerate(top_factors)]
    }
    with open(os.path.join(models_dir, 'shap_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(shap_summary, f, indent=4)

    # Guardar background data para explicabilidad interactiva local en la app
    joblib.dump(bg_samples, os.path.join(models_dir, 'shap_background.joblib'))

    print("\n" + "="*70)
    print("PIPELINE COMPLETADO CON ÉXITO. TODOS LOS ARTEFACTOS GENERADOS.")
    print("="*70)


if __name__ == '__main__':
    run_pipeline()

