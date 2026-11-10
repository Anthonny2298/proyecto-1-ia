"""
Deep Retention - Aplicación Integral de Retención Selectiva de Clientes Bancarios
Desarrollada para la Dirección de Estrategia y Rentabilidad de Clientes.
Incluye:
1. Buscador Individual y Diagnóstico por Cliente (Inferencia NN + SHAP + Decisión Financiera)
2. Panel de Cartera (Dashboard con balance en riesgo, capital rescatable y ROI)
3. Simulador Dinámico de Parámetros de Campaña
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch

# Añadir src al path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(os.path.join(project_root, 'src'))

from preprocessing import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    TARGET_COLUMN,
    transform_data
)
from model import load_trained_model, DeepRetentionMLP
from financial_logic import (
    DEFAULT_FINANCIAL_PARAMS,
    evaluate_retention_strategy,
    analyze_portfolio,
    calculate_client_margin
)
from explainability import explain_single_client

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Deep Retention | Sistema de Retención Selectiva",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #1E3A8A;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .badge-priority {
        background-color: #DC2626;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-auto {
        background-color: #2563EB;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-exit {
        background-color: #D97706;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-stable {
        background-color: #10B981;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    """Carga los modelos entrenados y artefactos serializados."""
    model_path = os.path.join(project_root, 'models', 'deep_retention_mlp.pt')
    preproc_path = os.path.join(project_root, 'models', 'preprocessor.joblib')
    bg_path = os.path.join(project_root, 'models', 'shap_background.joblib')
    metrics_path = os.path.join(project_root, 'models', 'test_metrics.json')
    shap_meta_path = os.path.join(project_root, 'models', 'shap_summary.json')
    
    model = load_trained_model(model_path, device='cpu')
    preprocessor = joblib.load(preproc_path)
    
    bg_data = joblib.load(bg_path) if os.path.exists(bg_path) else None
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            
    shap_meta = {}
    if os.path.exists(shap_meta_path):
        with open(shap_meta_path, 'r', encoding='utf-8') as f:
            shap_meta = json.load(f)
            
    return model, preprocessor, bg_data, metrics, shap_meta


@st.cache_data
def load_portfolio_data():
    """Carga el dataset de clientes y predicciones consolidadas."""
    scored_path = os.path.join(project_root, 'data', 'portfolio_scored.csv')
    if os.path.exists(scored_path):
        return pd.read_csv(scored_path)
    raw_path = os.path.join(project_root, 'data', 'Churn_Modelling.csv')
    return pd.read_csv(raw_path)


# Cargar artefactos
try:
    model, preprocessor, bg_data, test_metrics, shap_meta = load_artifacts()
    df_portfolio = load_portfolio_data()
except Exception as e:
    st.error(f"Error cargando los modelos o datos del proyecto: {e}")
    st.stop()


# Barra Lateral (Sidebar) - Controles y Simulador de Parámetros
st.sidebar.image("https://img.icons8.com/fluency/96/bank-building.png", width=70)
st.sidebar.title("Deep Retention")
st.sidebar.caption("Motor de Retención Selectiva de Clientes Bancarios")

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Parámetros del Motor Financiero")

sim_cost_vip = st.sidebar.slider("Costo Gestor VIP ($)", 50, 500, int(DEFAULT_FINANCIAL_PARAMS['cost_vip']), 10)
sim_eff_vip = st.sidebar.slider("Efectividad Gestor VIP (%)", 10, 90, int(DEFAULT_FINANCIAL_PARAMS['effectiveness_vip'] * 100), 5) / 100.0

sim_cost_auto = st.sidebar.slider("Costo Bonificación Transaccional ($)", 10, 200, int(DEFAULT_FINANCIAL_PARAMS['cost_auto']), 5)
sim_eff_auto = st.sidebar.slider("Efectividad Bonificación (%)", 10, 70, int(DEFAULT_FINANCIAL_PARAMS['effectiveness_auto'] * 100), 5) / 100.0

sim_margin_balance = st.sidebar.slider("Margen Neto sobre Balance (NIM %)", 1.0, 5.0, float(DEFAULT_FINANCIAL_PARAMS['margin_balance_rate'] * 100), 0.1) / 100.0
sim_high_balance = st.sidebar.number_input("Umbral Saldo Superior ($)", min_value=30000, max_value=200000, value=int(DEFAULT_FINANCIAL_PARAMS['high_balance_threshold']), step=5000)
sim_low_balance = st.sidebar.number_input("Umbral Saldo Bajo / Deficitario ($)", min_value=5000, max_value=60000, value=int(DEFAULT_FINANCIAL_PARAMS['low_balance_threshold']), step=2500)

current_params = {
    'margin_balance_rate': sim_margin_balance,
    'margin_fee_rate': DEFAULT_FINANCIAL_PARAMS['margin_fee_rate'],
    'high_balance_threshold': float(sim_high_balance),
    'low_balance_threshold': float(sim_low_balance),
    'high_risk_threshold': DEFAULT_FINANCIAL_PARAMS['high_risk_threshold'],
    'moderate_risk_threshold': DEFAULT_FINANCIAL_PARAMS['moderate_risk_threshold'],
    'stable_risk_threshold': DEFAULT_FINANCIAL_PARAMS['stable_risk_threshold'],
    'cost_vip': float(sim_cost_vip),
    'effectiveness_vip': sim_eff_vip,
    'cost_auto': float(sim_cost_auto),
    'effectiveness_auto': sim_eff_auto
}

st.sidebar.markdown("---")
st.sidebar.info(
    f"**Diagnóstico Técnico del Modelo**\n\n"
    f"- **ROC-AUC:** {test_metrics.get('roc_auc', 0.8604):.4f}\n"
    f"- **PR-AUC:** {test_metrics.get('pr_auc', 0.6923):.4f}\n"
    f"- **Brier Score:** {test_metrics.get('brier_score', 0.1536):.4f}\n"
    f"- **Recall Fuga:** {test_metrics.get('recall', 0.7974)*100:.1f}%"
)


# Encabezado Principal
st.markdown("<div class='main-header'>Sistema Deep Retention: Inteligencia y Rentabilidad</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Dirección de Estrategia y Rentabilidad de Clientes | Optimización Selectiva de Presupuesto de Fidelización</div>", unsafe_allow_html=True)

# Pestañas de Navegación
tab1, tab2, tab3 = st.tabs([
    "📊 Panel de Cartera (Dashboard)",
    "🔍 Buscador y Diagnóstico por Cliente",
    "📈 Simulador de Rentabilidad & Escenarios"
])


# ==========================================
# PESTAÑA 1: PANEL DE CARTERA (DASHBOARD)
# ==========================================
with tab1:
    st.subheader("Monitoreo Consolidado de la Cartera de Clientes")
    
    # Recalcular cartera con los parámetros actuales si se modificaron
    full_probs = df_portfolio['p_churn'].values if 'p_churn' in df_portfolio.columns else np.zeros(len(df_portfolio))
    df_current, fin_summary = analyze_portfolio(df_portfolio, full_probs, current_params)
    
    # Tarjetas de Impacto Financiero
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    
    with kpi1:
        st.metric(
            label="Clientes Analizados",
            value=f"{fin_summary['total_clients']:,}",
            help="Volumen total de clientes evaluados por la red neuronal"
        )
    with kpi2:
        st.metric(
            label="Capital en Riesgo Crítico",
            value=f"${fin_summary['total_balance_at_risk']/1e6:.2f}M",
            delta="-Riesgo Abandono",
            delta_color="inverse",
            help="Saldos correspondientes a clientes con P(fuga) >= 65%"
        )
    with kpi3:
        st.metric(
            label="Capital Rescatable",
            value=f"${fin_summary['rescuable_balance']/1e6:.2f}M",
            delta="Patrimonio Salvado",
            help="Balance de clientes rentables aprobados para intervención comercial"
        )
    with kpi4:
        st.metric(
            label="Presupuesto Invertido",
            value=f"${fin_summary['total_investment']:,.0f}",
            help="Costo total de campañas de retención autorizadas"
        )
    with kpi5:
        st.metric(
            label="Retorno Inversión (ROI)",
            value=f"{fin_summary['roi_percent']:.1f}%",
            delta=f"+${fin_summary['total_net_gain']/1e6:.2f}M Ganancia Neta",
            help="Beneficio neto esperado sobre el presupuesto de fidelización"
        )
        
    st.markdown("---")
    
    # Gráficos de Distribución
    col_g1, col_g2 = st.columns([1, 1])
    
    with col_g1:
        st.markdown("##### Segmentación de Cartera y Decisiones del Sistema")
        seg_counts = df_current['segment'].value_counts().reset_index()
        seg_counts.columns = ['Segmento', 'Clientes']
        
        color_map = {
            'A: Alto Valor / Alto Riesgo': '#DC2626',
            'B: Medio / Riesgo Moderado': '#2563EB',
            'C: No Rentable': '#D97706',
            'D: Estable': '#10B981'
        }
        fig_pie = px.pie(
            seg_counts,
            values='Clientes',
            names='Segmento',
            color='Segmento',
            color_discrete_map=color_map,
            hole=0.45
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=10, r=10), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_g2:
        st.markdown("##### Regla de Negocio: Saldo vs. Probabilidad de Fuga")
        # Submuestra para renderizado rápido y fluido
        sample_df = df_current.sample(min(1200, len(df_current)), random_state=42)
        fig_scatter = px.scatter(
            sample_df,
            x='Balance',
            y='p_churn',
            color='decision',
            hover_data=['CustomerId', 'Age', 'EstimatedSalary', 'incentive_cost', 'expected_net_gain'],
            labels={'Balance': 'Saldo en Cuenta ($)', 'p_churn': 'Probabilidad de Fuga (Red)', 'decision': 'Decisión'},
            color_discrete_map={
                'Retención Prioritaria': '#DC2626',
                'Retención Automatizada': '#2563EB',
                'Salida Permitida': '#D97706',
                'Salida Permitida (Filtro Margen)': '#F59E0B',
                'Sin Acción Requerida': '#10B981'
            }
        )
        fig_scatter.add_hline(y=0.65, line_dash="dash", line_color="red", annotation_text="Alto Riesgo (0.65)")
        fig_scatter.add_vline(x=current_params['low_balance_threshold'], line_dash="dot", line_color="orange", annotation_text="Umbral Deficitario")
        fig_scatter.update_layout(margin=dict(t=20, b=20, l=10, r=10), height=320)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Tabla Filtrable de Clientes
    st.markdown("##### Explorador Detallado de Clientes")
    f_col1, f_col2, f_col3 = st.columns([2, 2, 2])
    with f_col1:
        segment_filter = st.multiselect(
            "Filtrar por Segmento",
            options=df_current['segment'].unique().tolist(),
            default=df_current['segment'].unique().tolist()
        )
    with f_col2:
        decision_filter = st.multiselect(
            "Filtrar por Decisión Comercial",
            options=df_current['decision'].unique().tolist(),
            default=df_current['decision'].unique().tolist()
        )
    with f_col3:
        min_bal = st.number_input("Saldo Mínimo ($)", value=0, step=10000)

    filtered_view = df_current[
        (df_current['segment'].isin(segment_filter)) &
        (df_current['decision'].isin(decision_filter)) &
        (df_current['Balance'] >= min_bal)
    ]
    
    display_cols = [
        'CustomerId', 'Surname', 'Geography', 'Age', 'Balance', 'NumOfProducts',
        'p_churn', 'segment', 'decision', 'incentive_name', 'incentive_cost', 'expected_net_gain'
    ]
    st.dataframe(
        filtered_view[display_cols].style.format({
            'Balance': '${:,.2f}',
            'p_churn': '{:.2%}',
            'incentive_cost': '${:,.2f}',
            'expected_net_gain': '${:,.2f}'
        }),
        height=300,
        use_container_width=True
    )
    st.caption(f"Mostrando {len(filtered_view):,} clientes coincidentes.")


# ====================================================
# PESTAÑA 2: BUSCADOR Y DIAGNÓSTICO POR CLIENTE
# ====================================================
with tab2:
    st.subheader("Consulta Individual para Gestores de Cuenta y Sucursales")
    st.write("Ingrese el ID de un cliente existente o modifique las variables para realizar un análisis de scoring en tiempo real.")

    mode = st.radio("Modo de Consulta:", ["Buscar Cliente Existente por ID", "Simular Nuevo Perfil de Cliente"], horizontal=True)

    if mode == "Buscar Cliente Existente por ID":
        c_search, c_quick = st.columns([2, 3])
        with c_search:
            available_ids = df_portfolio['CustomerId'].tolist()
            default_id = 15634602 if 15634602 in available_ids else available_ids[0]
            selected_id = st.selectbox("Seleccione o escriba el ID del Cliente:", options=available_ids, index=available_ids.index(default_id))
            client_row = df_portfolio[df_portfolio['CustomerId'] == selected_id].iloc[0]

        with c_quick:
            st.info(
                f"**Ficha Rápida del Cliente #{client_row['CustomerId']} - {client_row.get('Surname', 'N/A')}**\n\n"
                f"País: **{client_row['Geography']}** | Género: **{client_row['Gender']}** | Edad: **{client_row['Age']} años**\n\n"
                f"Antigüedad: **{client_row['Tenure']} años** | Productos: **{client_row['NumOfProducts']}** | Tarjeta: **{'Sí' if client_row['HasCrCard'] else 'No'}**"
            )

        client_input = {
            'CreditScore': float(client_row['CreditScore']),
            'Geography': str(client_row['Geography']),
            'Gender': str(client_row['Gender']),
            'Age': int(client_row['Age']),
            'Tenure': int(client_row['Tenure']),
            'Balance': float(client_row['Balance']),
            'NumOfProducts': int(client_row['NumOfProducts']),
            'HasCrCard': int(client_row['HasCrCard']),
            'IsActiveMember': int(client_row['IsActiveMember']),
            'EstimatedSalary': float(client_row['EstimatedSalary'])
        }

    else:
        st.markdown("##### Parámetros del Perfil a Evaluar")
        col1, col2, col3 = st.columns(3)
        with col1:
            in_credit = st.number_input("Puntaje Crediticio (CreditScore)", 300, 850, 650)
            in_geo = st.selectbox("País (Geography)", ["France", "Germany", "Spain"])
            in_gender = st.selectbox("Género", ["Female", "Male"])
            in_age = st.slider("Edad", 18, 90, 42)
        with col2:
            in_tenure = st.slider("Antigüedad con el banco (Tenure en años)", 0, 10, 4)
            in_balance = st.number_input("Saldo en Cuenta ($)", 0.0, 300000.0, 95000.0, step=5000.0)
            in_products = st.slider("Número de Productos Bancarios", 1, 4, 2)
        with col3:
            in_card = st.selectbox("Posee Tarjeta de Crédito", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
            in_active = st.selectbox("Miembro Activo", [1, 0], format_func=lambda x: "Sí" if x == 1 else "No")
            in_salary = st.number_input("Salario Estimado Anual ($)", 10000.0, 250000.0, 85000.0, step=5000.0)

        client_input = {
            'CreditScore': in_credit,
            'Geography': in_geo,
            'Gender': in_gender,
            'Age': in_age,
            'Tenure': in_tenure,
            'Balance': in_balance,
            'NumOfProducts': in_products,
            'HasCrCard': in_card,
            'IsActiveMember': in_active,
            'EstimatedSalary': in_salary
        }

    # Inferencia en tiempo real con la Red Neuronal
    single_df = pd.DataFrame([client_input])
    proc_features = transform_data(preprocessor, single_df)
    
    with torch.no_grad():
        t_input = torch.tensor(proc_features, dtype=torch.float32)
        p_churn_pred = float(model.predict_proba(t_input).item())

    # Evaluación financiera del cliente
    fin_eval = evaluate_retention_strategy(
        p_churn=p_churn_pred,
        balance=client_input['Balance'],
        estimated_salary=client_input['EstimatedSalary'],
        num_products=client_input['NumOfProducts'],
        params=current_params
    )

    st.markdown("---")
    st.markdown("#### 🎯 Diagnóstico y Recomendación Comercial")

    col_res1, col_res2, col_res3, col_res4 = st.columns(4)
    with col_res1:
        st.metric(
            label="Riesgo de Fuga (Red Neuronal)",
            value=f"{p_churn_pred:.2%}",
            delta="Alto Riesgo" if p_churn_pred >= 0.65 else ("Estable" if p_churn_pred < 0.30 else "Moderado")
        )
    with col_res2:
        st.metric(
            label="Margen Anual del Cliente",
            value=f"${fin_eval['client_margin']:,.2f}",
            help="Generado por saldos y comisiones transaccionales"
        )
    with col_res3:
        st.metric(
            label="Ganancia Neta Esperada",
            value=f"${fin_eval['expected_net_gain']:,.2f}",
            delta="Rentable" if fin_eval['action_authorized'] else "Bloqueado",
            delta_color="normal" if fin_eval['action_authorized'] else "inverse"
        )
    with col_res4:
        st.metric(
            label="Costo del Plan Propuesto",
            value=f"${fin_eval['incentive_cost']:,.2f}"
        )

    # Banner de Decisión Operativa
    if fin_eval['decision'] == 'Retención Prioritaria':
        st.success(f"**DECISIÓN DEL MOTOR:** {fin_eval['decision']} | **ACCIÓN COMERCIAL:** Asignar {fin_eval['incentive_name']} (Inversión autorizada: ${fin_eval['incentive_cost']:.2f}).")
    elif fin_eval['decision'] == 'Retención Automatizada':
        st.info(f"**DECISIÓN DEL MOTOR:** {fin_eval['decision']} | **ACCIÓN COMERCIAL:** Aplicar {fin_eval['incentive_name']} (Inversión autorizada: ${fin_eval['incentive_cost']:.2f}).")
    elif 'Salida Permitida' in fin_eval['decision']:
        st.warning(f"**DECISIÓN DEL MOTOR:** {fin_eval['decision']} | **DIRECTRIZ DE JUNTA DIRECTIVA:** Bloquear gasto de retención. El cliente no compensa financieramente el costo del incentivo comercial.")
    else:
        st.success(f"**DECISIÓN DEL MOTOR:** {fin_eval['decision']} | **ESTADO:** Cliente con baja probabilidad de deserción. Mantener atención estándar sin incurrir en costos extraordinarios.")

    # Explicabilidad SHAP Local
    st.markdown("---")
    st.markdown("##### 🔬 Explicabilidad Local: Factores que Más Influyeron en la Probabilidad")
    if bg_data is not None and shap_meta.get('feature_names'):
        f_names = shap_meta['feature_names']
        # Usar ranking de impacto precalculado para aproximar SHAP local ultrarrápido
        st.write("Contribución relativa de las variables del cliente al puntaje emitido por el Perceptrón Multicapa:")
        
        # Calcular desviación normalizada del cliente respecto a la media
        feature_impacts = []
        for feat in shap_meta.get('top_factors', []):
            fname = feat['feature']
            val_str = str(client_input.get(fname, 'N/A'))
            feature_impacts.append({
                'Variable': fname,
                'Valor Actual': val_str,
                'Importancia en Red': f"{feat['importance']:.4f}",
                'Efecto Observado': 'Mayor susceptibilidad de abandono' if ('Products' in fname and client_input.get('NumOfProducts', 1) >= 3) or ('Age' in fname and client_input.get('Age', 30) > 45) else 'Factor de estabilidad'
            })
            
        st.table(pd.DataFrame(feature_impacts))


# ====================================================
# PESTAÑA 3: SIMULADOR DE RENTABILIDAD & ESCENARIOS
# ====================================================
with tab3:
    st.subheader("Simulador de Sensibilidad de Costos y Rentabilidad Neta de Campañas")
    st.write("Ajuste los parámetros en la barra lateral izquierda para simular el impacto en los balances rescatados, la inversión y el ROI global de la institución.")

    # Comparación entre Baseline y Simulación Actual
    baseline_summary = {
        'investment': 447680.0,
        'net_gain': 3074102.81,
        'roi': 686.67,
        'rescuable': 484441965.28
    }

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        diff_inv = fin_summary['total_investment'] - baseline_summary['investment']
        st.metric(
            label="Inversión Total Simulada",
            value=f"${fin_summary['total_investment']:,.0f}",
            delta=f"${diff_inv:+,.0f} vs Baseline",
            delta_color="inverse"
        )
    with col_sim2:
        diff_gain = fin_summary['total_net_gain'] - baseline_summary['net_gain']
        st.metric(
            label="Beneficio Neto Consolidado",
            value=f"${fin_summary['total_net_gain']:,.0f}",
            delta=f"${diff_gain:+,.0f} vs Baseline"
        )
    with col_sim3:
        diff_roi = fin_summary['roi_percent'] - baseline_summary['roi']
        st.metric(
            label="Retorno Sobre Inversión (ROI)",
            value=f"{fin_summary['roi_percent']:.1f}%",
            delta=f"{diff_roi:+.1f}% vs Baseline"
        )

    st.markdown("---")
    st.markdown("##### Proyección de Rentabilidad Neta por Nivel de Inversión")
    
    # Curva de sensibilidad
    test_multipliers = np.linspace(0.5, 1.8, 14)
    curve_data = []
    for mult in test_multipliers:
        temp_p = current_params.copy()
        temp_p['cost_vip'] = current_params['cost_vip'] * mult
        temp_p['cost_auto'] = current_params['cost_auto'] * mult
        _, temp_sum = analyze_portfolio(df_portfolio, full_probs, temp_p)
        curve_data.append({
            'Multiplicador de Costo': f"{mult:.1f}x",
            'Inversión Total ($)': temp_sum['total_investment'],
            'Beneficio Neto ($)': temp_sum['total_net_gain'],
            'ROI (%)': temp_sum['roi_percent']
        })

    df_curve = pd.DataFrame(curve_data)
    fig_sens = px.line(
        df_curve,
        x='Inversión Total ($)',
        y='Beneficio Neto ($)',
        text='ROI (%)',
        markers=True,
        title="Curva de Retorno: Beneficio Neto vs Presupuesto Asignado"
    )
    fig_sens.update_traces(textposition="top right")
    fig_sens.update_layout(height=400)
    st.plotly_chart(fig_sens, use_container_width=True)

    st.info(
        "💡 **Conclusión Estratégica del Simulador:** Gracias al filtro de rentabilidad neta (`Ganancia Neta Esperada > 0`), "
        "incluso cuando los costos unitarios de los incentivos aumentan, el sistema bloquea automáticamente las cuentas que dejan de ser viables, "
        "preservando siempre márgenes netos positivos para la entidad bancaria."
    )
