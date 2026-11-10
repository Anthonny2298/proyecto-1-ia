"""
Módulo de Lógica Financiera y Retención Rentable ("Deep Retention").
Implementa la ecuación de Ganancia Neta Esperada, el catálogo de intervenciones,
y las reglas de negocio para clasificar en Segmentos A, B, C ("Salida Permitida") y D.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


# Parámetros financieros por defecto
DEFAULT_FINANCIAL_PARAMS = {
    # Márgenes bancarios anuales
    'margin_balance_rate': 0.025,       # 2.5% de margen financiero neto sobre saldo
    'margin_fee_rate': 0.004,          # 0.4% comisión anual estimada sobre volumen salarial
    
    # Umbrales patrimoniales de saldo
    'high_balance_threshold': 80000.0,  # Saldo Superior
    'low_balance_threshold': 30000.0,   # Saldo Bajo / Deficitario
    
    # Umbrales de probabilidad de fuga
    'high_risk_threshold': 0.65,        # Alto riesgo
    'moderate_risk_threshold': 0.40,    # Riesgo moderado
    'stable_risk_threshold': 0.30,      # Cliente estable
    
    # Costos y efectividad de campañas
    'cost_vip': 200.0,                  # Costo Gestor VIP y Tarifa Plana ($)
    'effectiveness_vip': 0.50,          # Probabilidad de éxito en retención (50%)
    
    'cost_auto': 60.0,                  # Costo Bonificación Transaccional ($)
    'effectiveness_auto': 0.30,         # Probabilidad de éxito en retención (30%)
}


def calculate_client_margin(balance: float, estimated_salary: float, num_products: int, params: Dict[str, Any] = None) -> float:
    """
    Calcula el margen anual proyectado que el cliente aporta al banco en base
    a su balance patrimonial y flujo transaccional.
    """
    p = params or DEFAULT_FINANCIAL_PARAMS
    margin_bal = balance * p['margin_balance_rate']
    margin_fees = (estimated_salary * p['margin_fee_rate']) * max(1, num_products)
    total_margin = margin_bal + margin_fees
    return float(max(0.0, total_margin))


def calculate_expected_net_gain(p_churn: float, effectiveness: float, client_margin: float, cost: float) -> float:
    """
    Ecuación oficial de Enunciado.txt:
    Ganancia Neta Esperada = (Probabilidad de Fuga × Efectividad del Incentivo × Margen del Cliente) - Costo del Incentivo
    """
    return (p_churn * effectiveness * client_margin) - cost


def evaluate_retention_strategy(
    p_churn: float,
    balance: float,
    estimated_salary: float,
    num_products: int,
    params: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Evalúa integralmente la decisión de retención comercial para un cliente.
    Aplica la matriz de segmentación (A, B, C, D) y la restricción estricta de rentabilidad neta.
    """
    p = params or DEFAULT_FINANCIAL_PARAMS
    margin = calculate_client_margin(balance, estimated_salary, num_products, p)
    
    # 1. Determinación de segmento preliminar según riesgo y balance
    if p_churn < p['stable_risk_threshold']:
        segment = 'D: Estable'
        incentive_name = 'Mantenimiento Ordinario'
        cost = 0.0
        effectiveness = 0.0
        decision = 'Sin Acción Requerida'
        expected_gain = 0.0
        net_saved_margin = 0.0
        action_authorized = False

    elif balance < p['low_balance_threshold']:
        # Saldo bajo / deficitario con riesgo moderado o alto -> Salida Permitida
        segment = 'C: No Rentable'
        incentive_name = 'Sin intervención'
        cost = 0.0
        effectiveness = 0.0
        decision = 'Salida Permitida'
        expected_gain = 0.0
        net_saved_margin = 0.0
        action_authorized = False

    elif balance >= p['high_balance_threshold'] and p_churn >= p['high_risk_threshold']:
        # Segmento A: Alto Valor / Alto Riesgo
        segment = 'A: Alto Valor / Alto Riesgo'
        incentive_name = 'Gestor VIP y Tarifa Plana'
        cost = p['cost_vip']
        effectiveness = p['effectiveness_vip']
        expected_gain = calculate_expected_net_gain(p_churn, effectiveness, margin, cost)
        
        # Filtro estricto: ¿Es la ganancia neta esperada > 0?
        if expected_gain > 0:
            decision = 'Retención Prioritaria'
            action_authorized = True
            net_saved_margin = expected_gain
        else:
            decision = 'Salida Permitida (Filtro Margen)'
            action_authorized = False
            cost = 0.0
            net_saved_margin = 0.0

    else:
        # Segmento B: Medio / Riesgo Moderado
        segment = 'B: Medio / Riesgo Moderado'
        incentive_name = 'Bonificación Transaccional'
        cost = p['cost_auto']
        effectiveness = p['effectiveness_auto']
        expected_gain = calculate_expected_net_gain(p_churn, effectiveness, margin, cost)
        
        # Filtro estricto: ¿Es la ganancia neta esperada > 0?
        if expected_gain > 0:
            decision = 'Retención Automatizada'
            action_authorized = True
            net_saved_margin = expected_gain
        else:
            decision = 'Salida Permitida (Filtro Margen)'
            action_authorized = False
            cost = 0.0
            net_saved_margin = 0.0

    return {
        'segment': segment,
        'decision': decision,
        'action_authorized': action_authorized,
        'incentive_name': incentive_name,
        'incentive_cost': cost,
        'incentive_effectiveness': effectiveness,
        'client_margin': margin,
        'expected_net_gain': expected_gain if action_authorized else 0.0,
        'balance': balance,
        'p_churn': p_churn
    }


def analyze_portfolio(df_portfolio: pd.DataFrame, predictions: np.ndarray, params: Dict[str, Any] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ejecuta el análisis financiero sobre toda la cartera de clientes.
    Retorna el DataFrame enriquecido y las métricas consolidadas del panel de cartera.
    """
    p = params or DEFAULT_FINANCIAL_PARAMS
    results = []
    
    for idx, (_, row) in enumerate(df_portfolio.iterrows()):
        p_churn = float(predictions[idx])
        balance = float(row['Balance'])
        salary = float(row['EstimatedSalary'])
        products = int(row['NumOfProducts'])
        
        eval_res = evaluate_retention_strategy(p_churn, balance, salary, products, p)
        results.append(eval_res)
        
    df_eval = pd.DataFrame(results)
    
    # Evitar duplicación de columnas si df_portfolio ya contiene campos calculados previamente
    overlap_cols = [col for col in df_eval.columns if col in df_portfolio.columns]
    base_df = df_portfolio.drop(columns=overlap_cols, errors='ignore').reset_index(drop=True)
    df_combined = pd.concat([base_df, df_eval], axis=1)
    
    # Asegurar que las series sean 1D para evitar errores de indexación multidimensional
    p_churn_series = df_combined['p_churn'].squeeze()
    balance_series = df_combined['Balance'].squeeze()
    action_series = df_combined['action_authorized'].squeeze()
    decision_series = df_combined['decision'].squeeze()
    cost_series = df_combined['incentive_cost'].squeeze()
    net_gain_series = df_combined['expected_net_gain'].squeeze()
    
    # Métricas consolidadas requeridas en la plantilla ejecutiva
    total_clients = len(df_combined)
    high_risk_mask = p_churn_series >= p['high_risk_threshold']
    intervened_mask = action_series == True
    
    total_balance_at_risk = float(balance_series[high_risk_mask].sum())
    rescuable_balance = float(balance_series[intervened_mask].sum())
    total_investment = float(cost_series[intervened_mask].sum())
    total_net_gain = float(net_gain_series[intervened_mask].sum())
    
    roi_percent = (total_net_gain / total_investment * 100.0) if total_investment > 0 else 0.0
    
    summary = {
        'total_clients': total_clients,
        'total_balance_at_risk': total_balance_at_risk,
        'rescuable_balance': rescuable_balance,
        'total_investment': total_investment,
        'total_net_gain': total_net_gain,
        'roi_percent': roi_percent,
        'num_intervened': int(intervened_mask.sum()),
        'num_allowed_exit': int((decision_series.astype(str).str.contains('Salida Permitida')).sum()),
        'num_stable': int((decision_series == 'Sin Acción Requerida').sum())
    }
    
    return df_combined, summary

