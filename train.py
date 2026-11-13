"""
Pipeline de Entrenamiento y Evaluacion de Deep Retention.
1. Carga de datos y particion estratificada (70/15/15).
2. Preprocesamiento (StandardScaler + OneHotEncoder) ajustado solo en Train.
3. Entrenamiento del MLP con pos_weight y Early Stopping (semilla fija).
4. Calibracion Platt en validacion y evaluacion en Test (antes y despues de calibrar).
5. Analisis financiero de la cartera completa y del holdout de Test.
6. Explicabilidad SHAP global.
7. Exportacion de artefactos y figuras para la app y el informe.

Uso:
    python train.py
"""

import os
import sys
import json
import argparse

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from preprocessing import (  # noqa: E402
    DATA_FILENAME, load_raw_data, split_data, build_and_fit_preprocessor,
    transform_data, get_feature_names,
)
from model import (  # noqa: E402
    DEFAULT_SEED, set_seed, train_model, calibrate_model, evaluate_model,
    predict_proba, save_trained_model,
)
from financial_logic import analyze_portfolio, DEFAULT_FINANCIAL_PARAMS  # noqa: E402
from explainability import build_explainer, global_importance  # noqa: E402


def _save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4)


def _plot_training(history, path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], label='Train loss')
    ax.plot(epochs, history['val_loss'], label='Val loss')
    ax.axvline(history['best_epoch'], color='gray', linestyle='--', label=f"Mejor epoca ({history['best_epoch']})")
    ax.set_xlabel('Epoca'); ax.set_ylabel('BCE ponderada'); ax.set_title('Curva de entrenamiento')
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_calibration(y, probs_raw, probs_cal, path):
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    for label, probs in (('Sin calibrar', probs_raw), ('Calibrada (Platt)', probs_cal)):
        frac, mean = calibration_curve(y, probs, n_bins=10, strategy='quantile')
        ax.plot(mean, frac, 'o-', label=label)
    ax.plot([0, 1], [0, 1], 'k--', label='Perfecta')
    ax.set_xlabel('Probabilidad predicha'); ax.set_ylabel('Fuga observada')
    ax.set_title('Calibracion en Test'); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_importance(importance, path):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    importance.sort_values().plot.barh(ax=ax, color='#2563EB')
    ax.set_xlabel('Media |SHAP| (logit de fuga)'); ax.set_title('Factores determinantes de abandono')
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def run_pipeline(data_path=None, models_dir=None, data_out_dir=None, figures_dir=None,
                 epochs=120, patience=14, seed=DEFAULT_SEED, shap_samples=500, verbose=True):
    data_path = data_path or os.path.join(BASE_DIR, 'data', DATA_FILENAME)
    models_dir = models_dir or os.path.join(BASE_DIR, 'models')
    data_out_dir = data_out_dir or os.path.join(BASE_DIR, 'data')
    figures_dir = figures_dir or os.path.join(BASE_DIR, 'docs', 'figures')
    for d in (models_dir, data_out_dir, figures_dir):
        os.makedirs(d, exist_ok=True)
    set_seed(seed)

    print(f"[1/7] Cargando datos desde {data_path}")
    df = load_raw_data(data_path)
    print(f"Registros: {len(df):,} | Tasa de fuga: {df['Exited'].mean():.4f}")

    print("[2/7] Particion estratificada 70/15/15")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df, 0.15, 0.15, random_state=seed)
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    print("[3/7] Ajustando StandardScaler + OneHotEncoder (solo Train)")
    preprocessor = build_and_fit_preprocessor(X_train, save_path=os.path.join(models_dir, 'preprocessor.joblib'))
    feature_names = get_feature_names(preprocessor)
    X_train_p, X_val_p, X_test_p = (transform_data(preprocessor, X) for X in (X_train, X_val, X_test))
    print(f"Variables de entrada ({len(feature_names)}): {feature_names}")

    print("[4/7] Entrenando MLP")
    model, history = train_model(X_train_p, y_train, X_val_p, y_val, epochs=epochs,
                                 patience=patience, seed=seed, verbose=verbose)

    print("[5/7] Calibrando (Platt en Val) y evaluando en Test")
    raw_metrics, test_probs_raw = evaluate_model(model, X_test_p, y_test, calibrated=False)
    calibration = calibrate_model(model, X_val_p, y_val)
    cal_metrics, test_probs = evaluate_model(model, X_test_p, y_test, calibrated=True)
    for name in ('roc_auc', 'pr_auc', 'brier_score', 'ece', 'mean_predicted'):
        print(f"  {name:15s} sin calibrar: {raw_metrics[name]:.4f} | calibrada: {cal_metrics[name]:.4f}")

    metrics = {
        **cal_metrics,
        'uncalibrated': raw_metrics,
        'calibration': calibration,
        'best_epoch': history['best_epoch'],
        'epochs_run': len(history['val_loss']),
        'seed': seed,
        'n_train': len(X_train), 'n_val': len(X_val), 'n_test': len(X_test),
    }
    save_trained_model(model, len(feature_names), os.path.join(models_dir, 'deep_retention_mlp.pt'),
                       metrics, os.path.join(models_dir, 'test_metrics.json'))
    _save_json(history, os.path.join(models_dir, 'training_history.json'))

    print("[6/7] Analisis financiero (cartera completa y holdout de Test)")
    full_probs = predict_proba(model, transform_data(preprocessor, df))
    df_scored, portfolio_summary = analyze_portfolio(df, full_probs)
    _, test_summary = analyze_portfolio(df.loc[X_test.index], test_probs)
    _save_json({'portfolio': portfolio_summary, 'test_holdout': test_summary,
                'params': dict(DEFAULT_FINANCIAL_PARAMS)},
               os.path.join(models_dir, 'financial_summary.json'))
    df_scored.to_csv(os.path.join(data_out_dir, 'portfolio_scored.csv'), index=False)
    for k, v in portfolio_summary.items():
        print(f"  {k:28s} {v:,.2f}" if isinstance(v, float) else f"  {k:28s} {v:,}")

    print("[7/7] SHAP global (GradientExplainer)")
    rng = np.random.RandomState(seed)
    background = X_train_p[rng.choice(len(X_train_p), min(200, len(X_train_p)), replace=False)]
    explainer = build_explainer(model, background)
    importance = global_importance(explainer, X_test_p[:shap_samples], feature_names)
    _save_json({
        'feature_names': feature_names,
        'ranking': [{'feature': f, 'importance': float(v)} for f, v in importance.items()],
        'n_samples': int(min(shap_samples, len(X_test_p))),
    }, os.path.join(models_dir, 'shap_summary.json'))
    joblib.dump(background, os.path.join(models_dir, 'shap_background.joblib'))
    for rank, (feat, val) in enumerate(importance.head(5).items(), 1):
        print(f"  {rank}. {feat}: media |SHAP| = {val:.4f}")

    _plot_training(history, os.path.join(figures_dir, 'training_curve.png'))
    _plot_calibration(y_test, test_probs_raw, test_probs, os.path.join(figures_dir, 'calibration_curve.png'))
    _plot_importance(importance, os.path.join(figures_dir, 'shap_importance.png'))

    print("PIPELINE COMPLETADO.")
    return {'metrics': metrics, 'portfolio': portfolio_summary, 'test_holdout': test_summary,
            'importance': importance.to_dict()}


def main():
    parser = argparse.ArgumentParser(description='Entrena el modelo Deep Retention.')
    parser.add_argument('--data', default=None, help='Ruta al CSV (por defecto data/Customer-Churn-Records.csv)')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    run_pipeline(data_path=args.data, epochs=args.epochs, seed=args.seed)


if __name__ == '__main__':
    main()
