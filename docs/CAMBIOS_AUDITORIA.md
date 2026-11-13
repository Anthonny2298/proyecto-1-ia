# Spec de correcciones: auditoria Deep Retention v2

Rama: `fix/auditoria-deep-retention`. Cada punto lista el problema detectado en la auditoria
(verificado en Google Colab), el cambio y como se valida.

## 1. Datos y leakage

- **Problema:** el repo entrenaba con `Churn_Modelling.csv` y no con el dataset entregado
  (`Customer-Churn-Records.csv`). Ese dataset trae `Complain`, que coincide con `Exited` en el
  99.86% de los clientes (con ella el ROC-AUC sube a 0.9987: fuga de informacion).
- **Cambio:** se entrena con `data/Customer-Churn-Records.csv`. `Complain` se excluye como
  `LEAKAGE_COLUMNS`. `Satisfaction Score`, `Card Type` y `Point Earned` se excluyen como
  `NON_INFORMATIVE_COLUMNS` (correlacion con `Exited` ~0 y la ablacion no mejora el AUC).
- **Validacion:** test que falla si una columna de leakage llega al preprocesador.

## 2. Reproducibilidad

- **Problema:** sin semilla, cada `python train.py` daba metricas y ROI distintos a los publicados.
- **Cambio:** `set_seed(42)` (Python, NumPy, Torch, generador del DataLoader). Dependencias
  fijadas a las versiones de Google Colab. El modelo se mueve a CPU antes de la inferencia
  (antes fallaba con GPU).
- **Validacion:** test que entrena dos veces con la misma semilla y exige pesos identicos.

## 3. Calibracion de probabilidades

- **Problema:** `pos_weight` infla las probabilidades (media predicha 0.38 vs tasa real 0.20,
  ECE 0.17). La formula financiera usa la probabilidad directamente, asi que el ROI salia inflado.
- **Cambio:** Platt scaling ajustado en validacion y guardado dentro del checkpoint
  (buffers `calib_a`, `calib_b`). `predict_proba` devuelve la probabilidad calibrada.
  Se reportan Brier y ECE antes y despues de calibrar.
- **Validacion:** tests de rango [0, 1], de que la calibracion sobrevive al guardar/cargar y de ECE.

## 4. Logica financiera

- **Problema:** el Segmento C se asignaba por un umbral fijo de saldo (< $30k), no por la formula:
  bloqueaba 694 clientes rentables. El margen se forzaba a >= 0, asi que ningun cliente podia ser
  deficitario. "Capital rescatable" ($460M) > "capital en riesgo" ($184M).
- **Cambio:**
  - Margen anual = saldo x NIM + salario x comision x productos - **costo de servicio**
    (puede ser negativo: cliente deficitario).
  - Para todo cliente con p >= umbral estable se calcula
    `Ganancia = p x Efectividad x Margen - Costo` para cada incentivo elegible
    (Gestor VIP solo con saldo superior; Bonificacion para todos) y se elige el de mayor ganancia.
  - Ganancia > 0 -> A (VIP) o B (Bonificacion). Ganancia <= 0 -> C "Salida Permitida".
    p < umbral estable -> D. El segmento C queda definido exclusivamente por la formula.
  - KPIs coherentes: capital en riesgo = saldo con p >= umbral estable; capital de clientes
    intervenidos es un subconjunto; se agrega el capital esperado a retener (sum p x efect x saldo)
    y el ahorro por "Salida Permitida". Se reportan cartera completa y holdout de test.
  - Calculo vectorizado (sin `iterrows`), parametros inmutables y validados.
- **Validacion:** tests unitarios de la formula, de cada segmento, de cliente deficitario y de
  coherencia de KPIs.

## 5. Explicabilidad

- **Problema:** la "SHAP local" de la app era una regla fija (Age > 45, Products >= 3).
- **Cambio:** `shap.GradientExplainer` sobre la red, con importancia global y local agregada por
  variable original (las columnas one-hot se suman). La app muestra la explicacion real del cliente.

## 6. Aplicacion

- Sin valores hardcodeados: el baseline del simulador se calcula con los parametros por defecto y
  las metricas se leen de `models/test_metrics.json`.
- Si faltan artefactos la app se detiene con un mensaje claro (fail-closed).
- Nuevos controles: costo de servicio, comision, umbrales de riesgo.
- Analisis cacheado y vectorizado.

## 7. Entrega

- `notebooks/Deep_Retention_Colab.ipynb`: instala, corre tests, entrena y levanta la app en Colab.
- `docs/generate_pdf.py` genera el PDF desde el Markdown del informe (una sola fuente de verdad).
- Tests: `pytest` (unitarios + integracion del pipeline sobre una muestra del dataset real).

## Supuestos documentados

| Parametro | Valor | Justificacion |
|---|---|---|
| NIM sobre saldo | 2.5% anual | Margen financiero neto tipico de banca minorista |
| Comision sobre salario | 0.4% x producto | Proxy de comisiones transaccionales |
| Costo de servicio | $150 anual | Costo operativo de mantener la cuenta; permite margenes negativos |
| Gestor VIP | $200, 50% efectividad | Solo saldo >= $80k |
| Bonificacion transaccional | $60, 30% efectividad | Cualquier saldo |
| Umbral estable / alto riesgo | 0.30 / 0.65 | Matriz del enunciado |

La banda 0.30-0.65 no esta definida en la matriz del enunciado; se le aplica la misma formula.
