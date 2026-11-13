"""Integracion: pipeline completo sobre una muestra del dataset real, en un directorio temporal."""
import json

import numpy as np
import pandas as pd

from conftest import DATA_PATH
from explainability import build_explainer, explain_single_client
from model import load_trained_model, predict_proba
from preprocessing import transform_data
import joblib


def test_pipeline_end_to_end(tmp_path):
    import train

    sample = pd.read_csv(DATA_PATH).sample(1500, random_state=0)
    data_path = tmp_path / 'sample.csv'
    sample.to_csv(data_path, index=False)
    models, out, figs = tmp_path / 'models', tmp_path / 'data', tmp_path / 'figs'

    result = train.run_pipeline(data_path=str(data_path), models_dir=str(models), data_out_dir=str(out),
                                figures_dir=str(figs), epochs=4, patience=2, shap_samples=50, verbose=False)

    for name in ('deep_retention_mlp.pt', 'preprocessor.joblib', 'test_metrics.json',
                 'financial_summary.json', 'shap_summary.json', 'shap_background.joblib'):
        assert (models / name).exists(), name
    for name in ('training_curve.png', 'calibration_curve.png', 'shap_importance.png'):
        assert (figs / name).exists(), name

    metrics = json.loads((models / 'test_metrics.json').read_text(encoding='utf-8'))
    assert 0.5 < metrics['roc_auc'] <= 1.0
    assert 'uncalibrated' in metrics and 'calibration' in metrics

    scored = pd.read_csv(out / 'portfolio_scored.csv')
    assert len(scored) == len(sample)
    assert result['portfolio']['rescuable_balance'] <= result['portfolio']['total_balance_at_risk']
    assert 'Complain' not in json.loads((models / 'shap_summary.json').read_text())['feature_names']

    # Canal de inferencia separado: solo artefactos guardados.
    model = load_trained_model(str(models / 'deep_retention_mlp.pt'))
    pre = joblib.load(models / 'preprocessor.joblib')
    x = transform_data(pre, sample.head(1))
    p = predict_proba(model, x)
    assert 0.0 <= p[0] <= 1.0
    np.testing.assert_allclose(p[0], scored['p_churn'].iloc[0], rtol=1e-5)

    explainer = build_explainer(model, joblib.load(models / 'shap_background.joblib'))
    names = json.loads((models / 'shap_summary.json').read_text())['feature_names']
    contrib = explain_single_client(explainer, x, names)
    assert {c['feature'] for c in contrib} == {'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                                               'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
                                               'Geography', 'Gender'}
