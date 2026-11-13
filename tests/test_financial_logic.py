import numpy as np
import pandas as pd
import pytest

from financial_logic import (
    DEFAULT_FINANCIAL_PARAMS, SEGMENT_A, SEGMENT_B, SEGMENT_C, SEGMENT_D,
    DECISION_C, analyze_portfolio, calculate_client_margin, calculate_expected_net_gain,
    decide_retention, evaluate_retention_strategy, resolve_params,
)

P = DEFAULT_FINANCIAL_PARAMS


def test_formula_matches_enunciado():
    assert calculate_expected_net_gain(0.8, 0.5, 1000.0, 200.0) == pytest.approx(0.8 * 0.5 * 1000 - 200)


def test_margin_can_be_negative_for_deficit_client():
    margin = calculate_client_margin(balance=0, estimated_salary=20_000, num_products=1)
    assert margin == pytest.approx(20_000 * P['margin_fee_rate'] - P['cost_to_serve'])
    assert margin < 0


def test_stable_client_gets_no_action():
    r = evaluate_retention_strategy(0.10, 150_000, 100_000, 2)
    assert r['segment'] == SEGMENT_D
    assert not r['action_authorized']
    assert r['incentive_cost'] == 0


def test_high_value_client_gets_vip():
    r = evaluate_retention_strategy(0.80, 150_000, 100_000, 1)
    margin = 150_000 * P['margin_balance_rate'] + 100_000 * P['margin_fee_rate'] - P['cost_to_serve']
    assert r['segment'] == SEGMENT_A
    assert r['action_authorized']
    assert r['expected_net_gain'] == pytest.approx(0.8 * P['effectiveness_vip'] * margin - P['cost_vip'])


def test_medium_client_gets_bonus():
    r = evaluate_retention_strategy(0.70, 40_000, 80_000, 1)
    assert r['segment'] == SEGMENT_B
    assert r['incentive_cost'] == P['cost_auto']


def test_low_balance_but_profitable_client_is_retained():
    """El saldo bajo ya no implica Salida Permitida: decide la formula."""
    r = evaluate_retention_strategy(0.90, 0, 190_000, 2)
    assert r['action_authorized']
    assert r['segment'] == SEGMENT_B


def test_deficit_client_is_allowed_to_exit():
    r = evaluate_retention_strategy(0.95, 0, 15_000, 1)
    assert r['segment'] == SEGMENT_C
    assert r['decision'] == DECISION_C
    assert r['is_deficit']
    assert r['incentive_cost'] == 0
    assert r['expected_net_gain'] == 0


def test_unprofitable_high_balance_falls_back_to_cheaper_incentive():
    """Si el VIP no compensa pero la bonificacion si, se elige la bonificacion."""
    params = {'cost_vip': 5_000.0}
    r = evaluate_retention_strategy(0.80, 90_000, 50_000, 1, params)
    assert r['segment'] == SEGMENT_B


def test_every_authorized_action_has_positive_gain(real_data):
    rng = np.random.RandomState(1)
    df, _ = analyze_portfolio(real_data, rng.uniform(size=len(real_data)))
    auth = df[df['action_authorized']]
    assert (auth['expected_net_gain'] > 0).all()
    assert (df.loc[df['segment'] == SEGMENT_C, 'best_option_gain'] <= 0).all()


def test_kpis_are_coherent(real_data):
    rng = np.random.RandomState(2)
    _, s = analyze_portfolio(real_data, rng.uniform(size=len(real_data)))
    assert s['rescuable_balance'] <= s['total_balance_at_risk']
    assert s['expected_balance_retained'] <= s['rescuable_balance']
    assert s['critical_balance_at_risk'] <= s['total_balance_at_risk']
    assert s['num_vip'] + s['num_auto'] + s['num_allowed_exit'] + s['num_stable'] == s['total_clients']
    assert s['num_intervened'] == s['num_vip'] + s['num_auto']
    assert s['roi_percent'] == pytest.approx(s['total_net_gain'] / s['total_investment'] * 100)


def test_analyze_portfolio_does_not_mutate_input(real_data):
    sample = real_data.head(50).copy()
    before = sample.copy()
    analyze_portfolio(sample, np.full(50, 0.5))
    pd.testing.assert_frame_equal(sample, before)


def test_analyze_portfolio_rejects_length_mismatch(real_data):
    with pytest.raises(ValueError):
        analyze_portfolio(real_data.head(10), np.zeros(9))


def test_params_validation():
    with pytest.raises(ValueError):
        resolve_params({'unknown_param': 1})
    with pytest.raises(ValueError):
        resolve_params({'effectiveness_vip': 1.5})
    with pytest.raises(ValueError):
        resolve_params({'stable_risk_threshold': 0.9, 'high_risk_threshold': 0.5})


def test_default_params_are_immutable():
    with pytest.raises(TypeError):
        DEFAULT_FINANCIAL_PARAMS['cost_vip'] = 1.0


def test_higher_costs_never_increase_interventions(real_data):
    rng = np.random.RandomState(3)
    probs = rng.uniform(size=len(real_data))
    _, base = analyze_portfolio(real_data, probs)
    _, costly = analyze_portfolio(real_data, probs, {'cost_vip': 400.0, 'cost_auto': 120.0})
    assert costly['num_intervened'] <= base['num_intervened']


def test_vectorized_matches_single_client():
    probs = np.array([0.1, 0.5, 0.9])
    bal = np.array([0.0, 50_000.0, 120_000.0])
    sal = np.array([30_000.0, 60_000.0, 90_000.0])
    prod = np.array([1, 2, 1])
    batch = decide_retention(probs, bal, sal, prod)
    for i in range(3):
        single = evaluate_retention_strategy(probs[i], bal[i], sal[i], prod[i])
        assert single['segment'] == batch.loc[i, 'segment']
        assert single['expected_net_gain'] == pytest.approx(batch.loc[i, 'expected_net_gain'])
