import numpy as np
import pytest

from preprocessing import (
    CATEGORICAL_FEATURES, LEAKAGE_COLUMNS, NUMERICAL_FEATURES, assert_no_leakage,
    build_and_fit_preprocessor, get_feature_names, original_feature_name, split_data, transform_data,
)


def test_leakage_column_is_rejected():
    with pytest.raises(ValueError):
        assert_no_leakage(['Age', 'Complain'])
    with pytest.raises(ValueError):
        assert_no_leakage(['Age', 'Exited'])


def test_split_excludes_leakage_and_noise_columns(synthetic_data):
    (X_train, _), (X_val, _), (X_test, _) = split_data(synthetic_data)
    expected = list(NUMERICAL_FEATURES) + list(CATEGORICAL_FEATURES)
    for X in (X_train, X_val, X_test):
        assert list(X.columns) == expected
        assert not set(LEAKAGE_COLUMNS) & set(X.columns)


def test_split_sizes_and_stratification(real_data):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(real_data)
    n = len(real_data)
    assert len(X_train) + len(X_val) + len(X_test) == n
    assert len(X_test) == pytest.approx(0.15 * n, abs=1)
    rate = real_data['Exited'].mean()
    for y in (y_train, y_val, y_test):
        assert y.mean() == pytest.approx(rate, abs=0.01)
    assert not set(X_train.index) & set(X_test.index)


def test_preprocessor_fitted_only_on_train(synthetic_data):
    (X_train, _), _, (X_test, _) = split_data(synthetic_data)
    pre = build_and_fit_preprocessor(X_train)
    scaler = pre.named_transformers_['num']
    np.testing.assert_allclose(scaler.mean_, X_train[list(NUMERICAL_FEATURES)].mean().values)
    Xt = transform_data(pre, X_train)
    np.testing.assert_allclose(Xt[:, :len(NUMERICAL_FEATURES)].mean(axis=0), 0, atol=1e-5)
    assert transform_data(pre, X_test).dtype == np.float32


def test_feature_names_and_mapping(synthetic_data):
    (X_train, _), _, _ = split_data(synthetic_data)
    names = get_feature_names(build_and_fit_preprocessor(X_train))
    assert 'Geography_Germany' in names
    assert original_feature_name('Geography_Germany') == 'Geography'
    assert original_feature_name('Age') == 'Age'


def test_transform_ignores_extra_columns(synthetic_data):
    (X_train, _), _, _ = split_data(synthetic_data)
    pre = build_and_fit_preprocessor(X_train)
    a = transform_data(pre, synthetic_data.head(5))
    b = transform_data(pre, synthetic_data.head(5).drop(columns=['Complain', 'Card Type']))
    np.testing.assert_array_equal(a, b)
