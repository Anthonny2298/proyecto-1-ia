import numpy as np
import pytest
import torch

from model import (
    DeepRetentionMLP, calibrate_model, expected_calibration_error, load_trained_model,
    predict_proba, save_trained_model, train_model,
)
from preprocessing import build_and_fit_preprocessor, split_data, transform_data


@pytest.fixture
def prepared(synthetic_data):
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = split_data(synthetic_data)
    pre = build_and_fit_preprocessor(X_tr)
    return [transform_data(pre, X) for X in (X_tr, X_va, X_te)], (y_tr, y_va, y_te)


def test_architecture_matches_enunciado():
    m = DeepRetentionMLP(13)
    assert m.fc1.out_features == 128 and m.fc2.out_features == 64 and m.out.out_features == 1
    assert isinstance(m.bn1, torch.nn.BatchNorm1d)
    assert m.drop1.p == 0.25 and m.drop2.p == 0.15


def test_training_is_reproducible(prepared):
    (X_tr, X_va, _), (y_tr, y_va, _) = prepared
    m1, h1 = train_model(X_tr, y_tr, X_va, y_va, epochs=3, seed=7, device='cpu', verbose=False)
    m2, h2 = train_model(X_tr, y_tr, X_va, y_va, epochs=3, seed=7, device='cpu', verbose=False)
    assert h1['val_loss'] == h2['val_loss']
    for k, v in m1.state_dict().items():
        torch.testing.assert_close(v, m2.state_dict()[k])


def test_early_stopping_restores_best_epoch(prepared):
    (X_tr, X_va, _), (y_tr, y_va, _) = prepared
    _, h = train_model(X_tr, y_tr, X_va, y_va, epochs=40, patience=3, device='cpu', verbose=False)
    assert h['best_epoch'] == int(np.argmin(h['val_loss'])) + 1
    assert len(h['val_loss']) <= 40


def test_calibration_fixes_scale_and_survives_save_load(prepared, tmp_path):
    (X_tr, X_va, X_te), (y_tr, y_va, _) = prepared
    model, _ = train_model(X_tr, y_tr, X_va, y_va, epochs=5, device='cpu', verbose=False)
    raw = predict_proba(model, X_va, calibrated=False)
    calibrate_model(model, X_va, y_va)
    cal = predict_proba(model, X_va)
    assert ((cal >= 0) & (cal <= 1)).all()
    # Platt en validacion iguala la probabilidad media con la tasa observada.
    assert abs(cal.mean() - y_va.mean()) < abs(raw.mean() - y_va.mean()) + 1e-6
    assert abs(cal.mean() - y_va.mean()) < 0.03

    path = tmp_path / 'm.pt'
    save_trained_model(model, X_tr.shape[1], str(path))
    loaded = load_trained_model(str(path))
    np.testing.assert_allclose(predict_proba(loaded, X_te), predict_proba(model, X_te), rtol=1e-6)


def test_ece():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, np.array([0.0, 0.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.array([0.95, 0.95, 0.95, 0.95])) == pytest.approx(0.45)
