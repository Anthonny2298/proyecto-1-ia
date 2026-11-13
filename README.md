# Deep Retention: Retención Selectiva y Rentable de Clientes Bancarios

[![Informe Ejecutivo PDF](https://img.shields.io/badge/Informe%20Ejecutivo-PDF-success.svg)](docs/Informe_Ejecutivo_Deep_Retention.pdf)
[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jose30a/proyecto-1-ia/blob/main/notebooks/Deep_Retention_Colab.ipynb)
![Python 3.13](https://img.shields.io/badge/Python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/Framework-PyTorch-EE4C2C.svg)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B.svg)

Motor predictivo con redes neuronales que estima la probabilidad de fuga de cada cliente y solo autoriza un incentivo de retención cuando la **ganancia neta esperada para el banco es positiva**. Los clientes que no compensan el costo del incentivo quedan en **"Salida Permitida"**.

## 📄 Informe ejecutivo

👉 **[Informe Ejecutivo en PDF](docs/Informe_Ejecutivo_Deep_Retention.pdf)** (fuente editable: [Markdown](docs/Informe_Ejecutivo_Deep_Retention.md)).

## 🎯 Resultados

| Indicador | Cartera completa | Test (clientes no vistos) |
|---|---|---|
| Clientes analizados | 10.000 | 1.500 |
| Capital en riesgo (P(fuga) >= 0,30) | $195,4M | $31,2M |
| Capital esperado a retener | $55,1M | $8,8M |
| Inversión en retención | $320.660 | $50.220 |
| Ganancia neta esperada | $1,34M | $214.411 |
| **ROI** | **417,3%** | **426,9%** |
| Intervenidos / Salida Permitida / Estables | 1.877 / 313 / 7.810 | 291 / 49 / 1.160 |

| Métrica (Test) | Valor |
|---|---|
| ROC-AUC | 0,8669 |
| PR-AUC (línea base 0,204) | 0,7039 |
| Brier score (calibrado) | 0,1019 |
| ECE, error de calibración (sin calibrar: 0,204) | 0,0226 |

Factores principales de abandono (SHAP): número de productos, edad y actividad del cliente.

## 🧠 Solución

1. **Datos:** `data/Customer-Churn-Records.csv`. Partición estratificada 70/15/15. `Complain` se excluye por fuga de información (coincide con el abandono en el 99,86% de los casos).
2. **Preprocesamiento:** `StandardScaler` (8 numéricas) y `OneHotEncoder` (país, género), ajustados solo con entrenamiento.
3. **Red neuronal (PyTorch):** 128 (BatchNorm, ReLU, Dropout 0,25) -> 64 (ReLU, Dropout 0,15) -> 1 (sigmoide). `BCEWithLogitsLoss` con `pos_weight`, Adam y Early Stopping (paciencia 14).
4. **Calibración:** Platt scaling en validación, guardado dentro del modelo, para que la probabilidad sea usable en la fórmula financiera.
5. **Regla financiera:**

   `Ganancia Neta Esperada = P(fuga) x Efectividad x Margen del cliente - Costo del incentivo`

   Se evalúa cada incentivo elegible (Gestor VIP para saldo >= $80k, Bonificación Transaccional para todos), se elige el de mayor ganancia y solo se ejecuta si es > 0. El margen descuenta un costo de servicio, así que un cliente puede ser deficitario.
6. **Explicabilidad:** SHAP (`GradientExplainer`) global y por cliente.

| Segmento | Condición | Decisión | Incentivo |
|---|---|---|---|
| A: Alto Valor | Ganancia > 0 con Gestor VIP | Retención Prioritaria | Gestor VIP y Tarifa Plana ($200) |
| B: Valor Medio | Ganancia > 0 con Bonificación | Retención Automatizada | Bonificación Transaccional ($60) |
| C: No Rentable | Ningún incentivo con ganancia > 0 | Salida Permitida | Ninguno |
| D: Estable | P(fuga) < 0,30 | Sin Acción Requerida | Mantenimiento ordinario |

Supuestos financieros y detalle de cambios: [docs/CAMBIOS_AUDITORIA.md](docs/CAMBIOS_AUDITORIA.md).

## 🖥️ Aplicación (Streamlit)

* **Panel de Cartera:** capital en riesgo, capital a intervenir, inversión, ganancia neta, ROI, matriz de decisión y explorador de clientes.
* **Buscador por Cliente:** probabilidad calibrada, margen, ganancia neta esperada, decisión y gráfico SHAP del cliente. Acepta clientes existentes o perfiles nuevos.
* **Simulador de Parámetros:** costos, efectividades, márgenes, costo de servicio y umbrales; compara contra los valores por defecto y muestra la curva inversión vs ganancia.

## 🚀 Ejecución

### Google Colab (recomendado)

Abra [`notebooks/Deep_Retention_Colab.ipynb`](notebooks/Deep_Retention_Colab.ipynb) con el botón **Abrir en Colab** y ejecute todo. El notebook instala dependencias, corre los tests, entrena, muestra métricas y figuras y levanta la app con un link público temporal.

### Local

```bash
git clone https://github.com/jose30a/proyecto-1-ia.git
cd proyecto-1-ia
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python -m pytest -q tests       # tests unitarios + integración
python train.py                 # reentrena (reproducible, semilla 42)
streamlit run app/app.py        # abre http://localhost:8501
python docs/generate_pdf.py     # regenera el PDF desde el Markdown
```

El repositorio ya incluye el modelo entrenado, así que la app funciona sin reentrenar.

## 📂 Estructura

```text
app/app.py                       Aplicación Streamlit (panel, buscador, simulador)
data/Customer-Churn-Records.csv  Dataset del proyecto
data/portfolio_scored.csv        Cartera con probabilidades y decisiones
docs/                            Informe (MD + PDF), figuras, generador del PDF, spec de cambios
models/                          Pesos del MLP (con calibración), preprocesador, métricas y SHAP
notebooks/                       Notebook de ejecución en Google Colab
src/preprocessing.py             Carga, partición y transformaciones sin fuga
src/model.py                     MLP, entrenamiento, calibración y métricas
src/financial_logic.py           Ganancia neta esperada, segmentación y KPIs
src/explainability.py            SHAP global y local
tests/                           Tests unitarios y de integración (pytest)
train.py                         Pipeline de entrenamiento de punta a punta
requirements.txt                 Dependencias con versiones fijadas (Colab)
```

## ✅ Correspondencia con la rúbrica

| Criterio | Dónde se cubre |
|---|---|
| 1. Arquitectura y entrenamiento | `src/model.py`, `src/preprocessing.py`: MLP, escalado, one-hot, `pos_weight`, Dropout, BatchNorm, Early Stopping, exclusión de `Complain` |
| 2. Lógica financiera | `src/financial_logic.py`: fórmula de ganancia neta por incentivo; "Salida Permitida" definida por la fórmula |
| 3. Aplicación | `app/app.py`: buscador por cliente, panel de cartera y simulador |
| 4. Informe y explicabilidad | `docs/Informe_Ejecutivo_Deep_Retention.pdf`: ROI, capital, matriz, SHAP |
| 5. Reproducibilidad | `requirements.txt` fijado, semilla, inferencia separada del entrenamiento, pesos guardados, `tests/` |
