"""
Modulo de Logica Financiera y Retencion Rentable ("Deep Retention").

Regla de negocio (enunciado):
    Ganancia Neta Esperada = P(fuga) x Efectividad del Incentivo x Margen del Cliente - Costo del Incentivo
    La accion comercial solo se ejecuta si la Ganancia Neta Esperada es mayor a cero.

Segmentacion resultante:
    D: Estable          -> P(fuga) < umbral estable. Sin accion.
    A: Alto Valor       -> el mejor incentivo es el Gestor VIP y su ganancia es > 0.
    B: Valor Medio      -> el mejor incentivo es la Bonificacion y su ganancia es > 0.
    C: No Rentable      -> ningun incentivo tiene ganancia > 0. "Salida Permitida".
El segmento C lo define exclusivamente la formula, no un umbral de saldo.
"""

from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

DEFAULT_FINANCIAL_PARAMS: Mapping[str, float] = MappingProxyType({
    # Margen anual del cliente
    'margin_balance_rate': 0.025,       # 2.5% de margen financiero neto (NIM) sobre saldo
    'margin_fee_rate': 0.004,           # 0.4% del salario por producto, proxy de comisiones
    'cost_to_serve': 150.0,             # Costo operativo anual de mantener la cuenta ($)

    # Elegibilidad del Gestor VIP
    'high_balance_threshold': 80000.0,  # Saldo Superior

    # Bandas de riesgo (matriz del enunciado)
    'high_risk_threshold': 0.65,
    'stable_risk_threshold': 0.30,

    # Catalogo de incentivos
    'cost_vip': 200.0,                  # Gestor VIP y Tarifa Plana ($ por cliente)
    'effectiveness_vip': 0.50,
    'cost_auto': 60.0,                  # Bonificacion Transaccional ($ por cliente)
    'effectiveness_auto': 0.30,
})

SEGMENT_A = 'A: Alto Valor'
SEGMENT_B = 'B: Valor Medio'
SEGMENT_C = 'C: No Rentable'
SEGMENT_D = 'D: Estable'

DECISION_A = 'Retención Prioritaria'
DECISION_B = 'Retención Automatizada'
DECISION_C = 'Salida Permitida'
DECISION_D = 'Sin Acción Requerida'

INCENTIVE_VIP = 'Gestor VIP y Tarifa Plana'
INCENTIVE_AUTO = 'Bonificación Transaccional'
INCENTIVE_NONE = 'Sin intervención'
INCENTIVE_ORDINARY = 'Mantenimiento Ordinario'

RISK_HIGH = 'Alto'
RISK_MODERATE = 'Moderado'
RISK_LOW = 'Bajo'


def resolve_params(overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, float]:
    """Combina los parametros por defecto con los del usuario y los valida (fail-closed)."""
    params = dict(DEFAULT_FINANCIAL_PARAMS)
    if overrides:
        unknown = sorted(set(overrides) - set(params))
        if unknown:
            raise ValueError(f"Parametros financieros desconocidos: {unknown}")
        params.update({k: float(v) for k, v in overrides.items()})

    for key in ('effectiveness_vip', 'effectiveness_auto', 'high_risk_threshold', 'stable_risk_threshold'):
        if not 0.0 <= params[key] <= 1.0:
            raise ValueError(f"'{key}' debe estar entre 0 y 1 (recibido {params[key]})")
    for key in ('cost_vip', 'cost_auto', 'cost_to_serve', 'high_balance_threshold',
                'margin_balance_rate', 'margin_fee_rate'):
        if params[key] < 0:
            raise ValueError(f"'{key}' no puede ser negativo (recibido {params[key]})")
    if params['stable_risk_threshold'] > params['high_risk_threshold']:
        raise ValueError("El umbral estable no puede superar el umbral de alto riesgo")
    return params


def calculate_client_margin(balance, estimated_salary, num_products, params: Optional[Mapping[str, Any]] = None):
    """
    Margen anual que el cliente aporta al banco. Puede ser negativo (cliente deficitario)
    cuando el costo de servicio supera lo que genera por saldo y comisiones.
    Acepta escalares o arreglos.
    """
    p = resolve_params(params)
    products = np.maximum(1, np.asarray(num_products, dtype=float))
    margin = (np.asarray(balance, dtype=float) * p['margin_balance_rate']
              + np.asarray(estimated_salary, dtype=float) * p['margin_fee_rate'] * products
              - p['cost_to_serve'])
    return float(margin) if np.ndim(margin) == 0 else margin


def calculate_expected_net_gain(p_churn, effectiveness, client_margin, cost):
    """Ganancia Neta Esperada = P(fuga) x Efectividad x Margen - Costo."""
    return np.asarray(p_churn) * effectiveness * np.asarray(client_margin) - cost


def decide_retention(p_churn, balance, estimated_salary, num_products,
                     params: Optional[Mapping[str, Any]] = None) -> pd.DataFrame:
    """Decision comercial vectorizada para uno o muchos clientes."""
    p = resolve_params(params)
    p_churn = np.atleast_1d(np.asarray(p_churn, dtype=float))
    balance = np.atleast_1d(np.asarray(balance, dtype=float))
    salary = np.atleast_1d(np.asarray(estimated_salary, dtype=float))
    products = np.atleast_1d(np.asarray(num_products, dtype=float))

    margin = np.atleast_1d(calculate_client_margin(balance, salary, products, p))
    at_risk = p_churn >= p['stable_risk_threshold']
    vip_eligible = balance >= p['high_balance_threshold']

    gain_vip = np.where(vip_eligible,
                        calculate_expected_net_gain(p_churn, p['effectiveness_vip'], margin, p['cost_vip']),
                        -np.inf)
    gain_auto = calculate_expected_net_gain(p_churn, p['effectiveness_auto'], margin, p['cost_auto'])
    use_vip = gain_vip > gain_auto
    best_gain = np.where(use_vip, gain_vip, gain_auto)

    authorized = at_risk & (best_gain > 0)
    is_a = authorized & use_vip
    is_b = authorized & ~use_vip
    is_c = at_risk & ~authorized

    segment = np.select([is_a, is_b, is_c], [SEGMENT_A, SEGMENT_B, SEGMENT_C], SEGMENT_D)
    decision = np.select([is_a, is_b, is_c], [DECISION_A, DECISION_B, DECISION_C], DECISION_D)
    incentive = np.select([is_a, is_b, is_c], [INCENTIVE_VIP, INCENTIVE_AUTO, INCENTIVE_NONE], INCENTIVE_ORDINARY)
    effectiveness = np.select([is_a, is_b], [p['effectiveness_vip'], p['effectiveness_auto']], 0.0)
    cost = np.select([is_a, is_b], [p['cost_vip'], p['cost_auto']], 0.0)
    # Costo que un esquema indiscriminado habria gastado en los clientes de "Salida Permitida".
    avoided_cost = np.where(is_c, np.where(vip_eligible, p['cost_vip'], p['cost_auto']), 0.0)
    risk_band = np.select([p_churn >= p['high_risk_threshold'], at_risk], [RISK_HIGH, RISK_MODERATE], RISK_LOW)

    return pd.DataFrame({
        'p_churn': p_churn,
        'risk_band': risk_band,
        'segment': segment,
        'decision': decision,
        'action_authorized': authorized,
        'incentive_name': incentive,
        'incentive_cost': cost,
        'incentive_effectiveness': effectiveness,
        'client_margin': margin,
        'is_deficit': margin < 0,
        'expected_net_gain': np.where(authorized, best_gain, 0.0),
        'best_option_gain': np.where(at_risk, best_gain, 0.0),
        'expected_balance_retained': np.where(authorized, p_churn * effectiveness * balance, 0.0),
        'avoided_cost': avoided_cost,
    })


def evaluate_retention_strategy(p_churn: float, balance: float, estimated_salary: float, num_products: int,
                                params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Decision comercial para un unico cliente (consulta del gestor)."""
    row = decide_retention(p_churn, balance, estimated_salary, num_products, params).iloc[0]
    return {k: (v.item() if hasattr(v, 'item') else v) for k, v in row.items()}


def summarize_portfolio(decisions: pd.DataFrame, balance, params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """KPIs consolidados de la cartera (plantilla del informe ejecutivo)."""
    p = resolve_params(params)
    balance = np.asarray(balance, dtype=float)
    p_churn = decisions['p_churn'].to_numpy()
    authorized = decisions['action_authorized'].to_numpy(dtype=bool)
    at_risk = p_churn >= p['stable_risk_threshold']
    critical = p_churn >= p['high_risk_threshold']

    investment = float(decisions.loc[authorized, 'incentive_cost'].sum())
    net_gain = float(decisions.loc[authorized, 'expected_net_gain'].sum())
    return {
        'total_clients': int(len(decisions)),
        'num_at_risk': int(at_risk.sum()),
        'total_balance_at_risk': float(balance[at_risk].sum()),
        'critical_balance_at_risk': float(balance[critical].sum()),
        'rescuable_balance': float(balance[authorized].sum()),
        'expected_balance_retained': float(decisions['expected_balance_retained'].sum()),
        'total_investment': investment,
        'total_net_gain': net_gain,
        'roi_percent': (net_gain / investment * 100.0) if investment > 0 else 0.0,
        'num_intervened': int(authorized.sum()),
        'num_vip': int((decisions['segment'] == SEGMENT_A).sum()),
        'num_auto': int((decisions['segment'] == SEGMENT_B).sum()),
        'num_allowed_exit': int((decisions['segment'] == SEGMENT_C).sum()),
        'num_deficit_allowed_exit': int(((decisions['segment'] == SEGMENT_C) & decisions['is_deficit']).sum()),
        'avoided_spend': float(decisions['avoided_cost'].sum()),
        'num_stable': int((decisions['segment'] == SEGMENT_D).sum()),
    }


def analyze_portfolio(df_portfolio: pd.DataFrame, predictions,
                      params: Optional[Mapping[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analisis financiero de toda la cartera.
    Devuelve el DataFrame enriquecido (sin mutar el original) y los KPIs consolidados.
    """
    predictions = np.asarray(predictions, dtype=float).ravel()
    if len(predictions) != len(df_portfolio):
        raise ValueError(f"Predicciones ({len(predictions)}) y clientes ({len(df_portfolio)}) no coinciden")

    decisions = decide_retention(
        predictions,
        df_portfolio['Balance'].to_numpy(),
        df_portfolio['EstimatedSalary'].to_numpy(),
        df_portfolio['NumOfProducts'].to_numpy(),
        params,
    )
    base = df_portfolio.drop(columns=[c for c in decisions.columns if c in df_portfolio.columns]).reset_index(drop=True)
    combined = pd.concat([base, decisions], axis=1)
    return combined, summarize_portfolio(decisions, df_portfolio['Balance'].to_numpy(), params)
