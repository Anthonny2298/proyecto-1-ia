"""
Deep Retention - Aplicacion de Retencion Selectiva de Clientes Bancarios.
1. Panel de Cartera (KPIs de capital en riesgo, capital a intervenir y ROI).
2. Buscador por Cliente (probabilidad de la red, impacto monetario, decision y SHAP local).
3. Simulador de parametros de campana.

Uso:
    streamlit run app/app.py
"""

import os
import sys
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

from preprocessing import transform_data  # noqa: E402
from model import load_trained_model, predict_proba  # noqa: E402
from financial_logic import (  # noqa: E402
    DEFAULT_FINANCIAL_PARAMS, DECISION_A, DECISION_B, DECISION_C, DECISION_D,
    SEGMENT_A, SEGMENT_B, SEGMENT_C, SEGMENT_D, analyze_portfolio, evaluate_retention_strategy,
)
from explainability import build_explainer, explain_single_client  # noqa: E402

MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
REQUIRED_ARTIFACTS = ('deep_retention_mlp.pt', 'preprocessor.joblib', 'shap_background.joblib',
                      'test_metrics.json', 'shap_summary.json')
SCORED_PATH = os.path.join(PROJECT_ROOT, 'data', 'portfolio_scored.csv')

SEGMENT_COLORS = {SEGMENT_A: '#DC2626', SEGMENT_B: '#2563EB', SEGMENT_C: '#D97706', SEGMENT_D: '#10B981'}
DECISION_COLORS = {DECISION_A: '#DC2626', DECISION_B: '#2563EB', DECISION_C: '#D97706', DECISION_D: '#10B981'}

st.set_page_config(page_title="Deep Retention | Retención Selectiva", page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

# Indicador de carga: reemplaza la animacion deportiva de Streamlit por un spinner circular.
st.markdown("""
<style>
[data-testid="stStatusWidgetRunningIcon"] > * { display: none !important; }
[data-testid="stStatusWidgetRunningIcon"]::after {
    content: ""; display: block; width: 18px; height: 18px; box-sizing: border-box;
    border: 3px solid rgba(37, 99, 235, 0.2); border-top-color: #2563EB;
    border-radius: 50%; animation: dr-spin 0.8s linear infinite;
}
@keyframes dr-spin { to { transform: rotate(360deg); } }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Carga modelo, preprocesador, explicador SHAP y metricas. Falla si falta algo."""
    missing = [n for n in REQUIRED_ARTIFACTS if not os.path.exists(os.path.join(MODELS_DIR, n))]
    if missing or not os.path.exists(SCORED_PATH):
        raise FileNotFoundError(f"Faltan artefactos: {missing or ['data/portfolio_scored.csv']}. "
                                "Ejecute 'python train.py' antes de abrir la app.")
    model = load_trained_model(os.path.join(MODELS_DIR, 'deep_retention_mlp.pt'))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.joblib'))
    explainer = build_explainer(model, joblib.load(os.path.join(MODELS_DIR, 'shap_background.joblib')))
    with open(os.path.join(MODELS_DIR, 'test_metrics.json'), encoding='utf-8') as f:
        metrics = json.load(f)
    with open(os.path.join(MODELS_DIR, 'shap_summary.json'), encoding='utf-8') as f:
        shap_meta = json.load(f)
    return model, preprocessor, explainer, metrics, shap_meta


@st.cache_data
def load_portfolio():
    return pd.read_csv(SCORED_PATH)


@st.cache_data
def run_analysis(params_items: tuple):
    """Analisis de cartera cacheado por combinacion de parametros."""
    portfolio = load_portfolio()
    return analyze_portfolio(portfolio, portfolio['p_churn'].to_numpy(), dict(params_items))


try:
    model, preprocessor, explainer, test_metrics, shap_meta = load_artifacts()
    df_portfolio = load_portfolio()
except Exception as exc:  # fail-closed: sin artefactos no hay decisiones
    st.error(f"No se pudo iniciar Deep Retention: {exc}")
    st.stop()

# ---------------------------------------------------------------- Sidebar: parametros
D = DEFAULT_FINANCIAL_PARAMS
st.sidebar.title("🏦 Deep Retention")
st.sidebar.caption("Motor de Retención Selectiva de Clientes Bancarios")
st.sidebar.subheader("⚙️ Catálogo de incentivos")
cost_vip = st.sidebar.slider("Costo Gestor VIP ($)", 50, 600, int(D['cost_vip']), 10)
eff_vip = st.sidebar.slider("Efectividad Gestor VIP (%)", 5, 90, int(D['effectiveness_vip'] * 100), 5) / 100
cost_auto = st.sidebar.slider("Costo Bonificación ($)", 10, 300, int(D['cost_auto']), 5)
eff_auto = st.sidebar.slider("Efectividad Bonificación (%)", 5, 90, int(D['effectiveness_auto'] * 100), 5) / 100

st.sidebar.subheader("💵 Margen del cliente")
nim = st.sidebar.slider("Margen neto sobre saldo, NIM (%)", 0.5, 5.0, D['margin_balance_rate'] * 100, 0.1) / 100
fee = st.sidebar.slider("Comisión anual por producto (% del salario)", 0.0, 1.5, D['margin_fee_rate'] * 100, 0.05) / 100
cost_to_serve = st.sidebar.slider("Costo de servicio anual ($)", 0, 600, int(D['cost_to_serve']), 10)

st.sidebar.subheader("🎯 Segmentación")
high_balance = st.sidebar.number_input("Saldo mínimo para Gestor VIP ($)", 10_000, 250_000,
                                       int(D['high_balance_threshold']), 5_000)
stable_thr, high_thr = st.sidebar.slider("Umbrales de riesgo (estable / alto)", 0.05, 0.95,
                                         (D['stable_risk_threshold'], D['high_risk_threshold']), 0.05)

current_params = {
    'cost_vip': float(cost_vip), 'effectiveness_vip': eff_vip,
    'cost_auto': float(cost_auto), 'effectiveness_auto': eff_auto,
    'margin_balance_rate': nim, 'margin_fee_rate': fee, 'cost_to_serve': float(cost_to_serve),
    'high_balance_threshold': float(high_balance),
    'stable_risk_threshold': float(stable_thr), 'high_risk_threshold': float(high_thr),
}
params_key = tuple(sorted(current_params.items()))

st.sidebar.markdown("---")
st.sidebar.info(
    "**Diagnóstico del modelo (Test)**\n\n"
    f"- ROC-AUC: {test_metrics['roc_auc']:.4f}\n"
    f"- PR-AUC: {test_metrics['pr_auc']:.4f}\n"
    f"- Brier: {test_metrics['brier_score']:.4f}\n"
    f"- ECE (calibración): {test_metrics['ece']:.4f}\n"
    f"- Prob. media vs tasa real: {test_metrics['mean_predicted']:.3f} / {test_metrics['observed_rate']:.3f}"
)

st.title("🏦 Sistema Deep Retention")
st.markdown("##### Retención solo cuando la Ganancia Neta Esperada es positiva")
st.latex(r"\text{Ganancia} = P(\text{fuga}) \times \text{Efectividad} \times \text{Margen} - \text{Costo}")

tab1, tab2, tab3 = st.tabs(["📊 Panel de Cartera", "🔍 Buscador por Cliente", "📈 Simulador de Parámetros"])
df_current, fin = run_analysis(params_key)


def money(x: float) -> str:
    return f"${x / 1e6:,.2f}M" if abs(x) >= 1e6 else f"${x:,.0f}"


# ---------------------------------------------------------------- Tab 1: cartera
with tab1:
    st.subheader("Monitoreo consolidado de la cartera")
    r1 = st.columns(4)
    r1[0].metric("Clientes analizados", f"{fin['total_clients']:,}")
    r1[1].metric("Capital en riesgo", money(fin['total_balance_at_risk']),
                 help=f"Saldo de clientes con P(fuga) >= {stable_thr:.2f}")
    r1[2].metric("Capital crítico", money(fin['critical_balance_at_risk']),
                 help=f"Saldo de clientes con P(fuga) >= {high_thr:.2f}")
    r1[3].metric("Capital de clientes a intervenir", money(fin['rescuable_balance']),
                 help="Saldo de los clientes con Ganancia Neta Esperada > 0 (subconjunto del capital en riesgo)")
    r2 = st.columns(4)
    r2[0].metric("Capital esperado a retener", money(fin['expected_balance_retained']),
                 help="Suma de P(fuga) x Efectividad x Saldo de los clientes intervenidos")
    r2[1].metric("Inversión en retención", money(fin['total_investment']))
    r2[2].metric("Ganancia neta esperada", money(fin['total_net_gain']))
    r2[3].metric("ROI", f"{fin['roi_percent']:.1f}%")
    st.caption(f"Salida Permitida: {fin['num_allowed_exit']:,} clientes "
               f"({fin['num_deficit_allowed_exit']:,} deficitarios). Gasto evitado: {money(fin['avoided_spend'])}. "
               "Nota: la cartera incluye clientes usados en el entrenamiento; las cifras fuera de muestra "
               "están en el informe (holdout de Test).")

    st.markdown("---")
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### Segmentación de la cartera")
        seg = df_current['segment'].value_counts().reset_index()
        seg.columns = ['Segmento', 'Clientes']
        fig = px.pie(seg, values='Clientes', names='Segmento', color='Segmento',
                     color_discrete_map=SEGMENT_COLORS, hole=0.45)
        fig.update_layout(margin=dict(t=20, b=20, l=10, r=10), height=320)
        st.plotly_chart(fig, width='stretch')
    with g2:
        st.markdown("##### Margen del cliente vs probabilidad de fuga")
        sample = df_current.sample(min(1500, len(df_current)), random_state=42)
        fig = px.scatter(sample, x='client_margin', y='p_churn', color='decision',
                         color_discrete_map=DECISION_COLORS,
                         hover_data=['CustomerId', 'Balance', 'EstimatedSalary', 'expected_net_gain'],
                         labels={'client_margin': 'Margen anual ($)', 'p_churn': 'P(fuga)', 'decision': 'Decisión'})
        fig.add_hline(y=stable_thr, line_dash='dot', annotation_text=f'Estable < {stable_thr:.2f}')
        fig.add_hline(y=high_thr, line_dash='dash', line_color='red', annotation_text=f'Alto riesgo >= {high_thr:.2f}')
        fig.add_vline(x=0, line_dash='dot', line_color='orange', annotation_text='Margen 0')
        fig.update_layout(margin=dict(t=20, b=20, l=10, r=10), height=320)
        st.plotly_chart(fig, width='stretch')

    st.markdown("##### Matriz de decisión")
    matrix = df_current.groupby(['segment', 'decision', 'incentive_name']).agg(
        Clientes=('p_churn', 'size'), P_media=('p_churn', 'mean'), Saldo_medio=('Balance', 'mean'),
        Costo_total=('incentive_cost', 'sum'), Ganancia_neta=('expected_net_gain', 'sum')).reset_index()
    st.dataframe(matrix.style.format({'P_media': '{:.2f}', 'Saldo_medio': '${:,.0f}',
                                      'Costo_total': '${:,.0f}', 'Ganancia_neta': '${:,.0f}'}),
                 width='stretch', hide_index=True)

    st.markdown("##### Explorador de clientes")
    f1, f2, f3 = st.columns(3)
    seg_filter = f1.multiselect("Segmento", sorted(df_current['segment'].unique()),
                                default=sorted(df_current['segment'].unique()))
    dec_filter = f2.multiselect("Decisión", sorted(df_current['decision'].unique()),
                                default=sorted(df_current['decision'].unique()))
    min_bal = f3.number_input("Saldo mínimo ($)", value=0, step=10_000)
    view = df_current[df_current['segment'].isin(seg_filter) & df_current['decision'].isin(dec_filter)
                      & (df_current['Balance'] >= min_bal)]
    cols = ['CustomerId', 'Surname', 'Geography', 'Age', 'Balance', 'NumOfProducts', 'p_churn',
            'client_margin', 'segment', 'decision', 'incentive_name', 'incentive_cost', 'expected_net_gain']
    st.dataframe(view[cols].style.format({'Balance': '${:,.2f}', 'p_churn': '{:.2%}', 'client_margin': '${:,.2f}',
                                          'incentive_cost': '${:,.2f}', 'expected_net_gain': '${:,.2f}'}),
                 height=320, width='stretch', hide_index=True)
    st.caption(f"{len(view):,} clientes coincidentes.")

# ---------------------------------------------------------------- Tab 2: cliente
with tab2:
    st.subheader("Consulta individual para gestores de cuenta")
    mode = st.radio("Modo", ["Cliente existente", "Perfil nuevo"], horizontal=True)
    geo_options = list(preprocessor.named_transformers_['cat'].categories_[0])
    gender_options = list(preprocessor.named_transformers_['cat'].categories_[1])

    if mode == "Cliente existente":
        ids = df_portfolio['CustomerId'].tolist()
        selected = st.selectbox("ID del cliente", ids, index=0)
        row = df_portfolio.loc[df_portfolio['CustomerId'] == selected].iloc[0]
        st.info(f"**#{row['CustomerId']} - {row['Surname']}** | {row['Geography']} | {row['Gender']} | "
                f"{row['Age']} años | Antigüedad {row['Tenure']} | Productos {row['NumOfProducts']} | "
                f"Activo: {'Sí' if row['IsActiveMember'] else 'No'}")
        client = {k: row[k] for k in ('CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance',
                                      'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary')}
    else:
        c1, c2, c3 = st.columns(3)
        client = {
            'CreditScore': c1.number_input("CreditScore", 300, 850, 650),
            'Geography': c1.selectbox("País", geo_options),
            'Gender': c1.selectbox("Género", gender_options),
            'Age': c1.slider("Edad", 18, 92, 42),
            'Tenure': c2.slider("Antigüedad (años)", 0, 10, 4),
            'Balance': c2.number_input("Saldo ($)", 0.0, 300_000.0, 95_000.0, step=5_000.0),
            'NumOfProducts': c2.slider("Productos", 1, 4, 1),
            'HasCrCard': c3.selectbox("Tarjeta de crédito", [1, 0], format_func=lambda x: "Sí" if x else "No"),
            'IsActiveMember': c3.selectbox("Miembro activo", [1, 0], format_func=lambda x: "Sí" if x else "No"),
            'EstimatedSalary': c3.number_input("Salario anual ($)", 0.0, 300_000.0, 85_000.0, step=5_000.0),
        }

    features = transform_data(preprocessor, pd.DataFrame([client]))
    p_churn = float(predict_proba(model, features)[0])
    ev = evaluate_retention_strategy(p_churn, client['Balance'], client['EstimatedSalary'],
                                     client['NumOfProducts'], current_params)

    st.markdown("---")
    m = st.columns(4)
    m[0].metric("P(fuga) calibrada", f"{p_churn:.1%}", ev['risk_band'] + " riesgo", delta_color="off")
    m[1].metric("Margen anual del cliente", f"${ev['client_margin']:,.2f}")
    m[2].metric("Mejor ganancia neta esperada", f"${ev['best_option_gain']:,.2f}")
    m[3].metric("Costo del incentivo", f"${ev['incentive_cost']:,.2f}")

    if ev['decision'] in (DECISION_A, DECISION_B):
        st.success(f"**{ev['segment']} - {ev['decision']}:** aplicar **{ev['incentive_name']}** "
                   f"(costo ${ev['incentive_cost']:,.2f}, ganancia neta esperada ${ev['expected_net_gain']:,.2f}).")
    elif ev['decision'] == DECISION_C:
        reason = ("el cliente es deficitario (margen negativo)" if ev['is_deficit']
                  else "ningún incentivo genera ganancia neta positiva")
        st.warning(f"**{ev['segment']} - Salida Permitida:** {reason}. No autorizar gasto de retención.")
    else:
        st.info(f"**{ev['segment']} - {ev['decision']}:** probabilidad de fuga menor a {stable_thr:.2f}. "
                "Mantener atención estándar.")

    st.markdown("##### Explicabilidad local (SHAP)")
    contrib = pd.DataFrame(explain_single_client(explainer, features, shap_meta['feature_names']))
    contrib['Valor del cliente'] = contrib['feature'].map(lambda f: str(client.get(f, '')))
    fig = px.bar(contrib.iloc[::-1], x='shap_value', y='feature', orientation='h', color='impact',
                 color_discrete_map={'Aumenta Riesgo': '#DC2626', 'Disminuye Riesgo': '#10B981'},
                 hover_data=['Valor del cliente'],
                 labels={'shap_value': 'Contribución SHAP al logit de fuga', 'feature': ''})
    fig.update_layout(height=380, margin=dict(t=10, b=10, l=10, r=10))
    st.plotly_chart(fig, width='stretch')
    st.caption("Barras rojas empujan el riesgo hacia arriba y verdes hacia abajo, respecto al cliente promedio.")

# ---------------------------------------------------------------- Tab 3: simulador
with tab3:
    st.subheader("Simulador de costos y rentabilidad neta")
    st.write("Ajuste los parámetros en la barra lateral. La comparación es contra los parámetros por defecto.")
    _, base = run_analysis(tuple(sorted(dict(DEFAULT_FINANCIAL_PARAMS).items())))
    s = st.columns(4)
    s[0].metric("Inversión", money(fin['total_investment']),
                f"{fin['total_investment'] - base['total_investment']:+,.0f} vs base", delta_color="inverse")
    s[1].metric("Ganancia neta", money(fin['total_net_gain']),
                f"{fin['total_net_gain'] - base['total_net_gain']:+,.0f} vs base")
    s[2].metric("ROI", f"{fin['roi_percent']:.1f}%", f"{fin['roi_percent'] - base['roi_percent']:+.1f} pp vs base")
    s[3].metric("Clientes intervenidos", f"{fin['num_intervened']:,}",
                f"{fin['num_intervened'] - base['num_intervened']:+,} vs base")

    st.markdown("##### Curva de retorno al escalar el costo de los incentivos")
    rows = []
    for mult in np.linspace(0.5, 2.0, 16):
        p = dict(current_params, cost_vip=current_params['cost_vip'] * mult,
                 cost_auto=current_params['cost_auto'] * mult)
        _, summ = run_analysis(tuple(sorted(p.items())))
        rows.append({'Multiplicador': f"{mult:.1f}x", 'Inversión ($)': summ['total_investment'],
                     'Ganancia neta ($)': summ['total_net_gain'], 'ROI': f"{summ['roi_percent']:.0f}%",
                     'Intervenidos': summ['num_intervened']})
    curve = pd.DataFrame(rows)
    fig = px.line(curve, x='Inversión ($)', y='Ganancia neta ($)', text='ROI', markers=True,
                  hover_data=['Multiplicador', 'Intervenidos'])
    fig.update_traces(textposition='top center')
    fig.update_layout(height=420)
    st.plotly_chart(fig, width='stretch')
    st.info("Al subir el costo de los incentivos, el filtro de Ganancia Neta Esperada > 0 deja fuera a los "
            "clientes que dejan de compensar: la inversión cae y la ganancia neta se mantiene positiva.")
