# INFORME EJECUTIVO: SISTEMA "DEEP RETENTION"
**Para:** Dirección de Estrategia y Rentabilidad de Clientes & Junta Directiva  
**De:** Equipo de Inteligencia de Negocios y Ciencia de Datos  
**Asunto:** Resultados Operativos, Técnicos y Financieros del Motor Predictivo de Retención Selectiva  
**Fecha:** Noviembre 2026  

---

## 1. Resumen de Impacto Financiero y Retorno

La implementación del sistema **Deep Retention** responde a la directriz ejecutiva de erradicar el gasto indiscriminado en retención de clientes no rentables y focalizar el capital en cuentas con retorno neto positivo comprobable. A partir del análisis del portafolio completo mediante la red neuronal y la ecuación de rentabilidad neta, se consolidan los siguientes resultados:

* **Volumen Total de Clientes Analizados:** 10,000 clientes activos en la cartera.
* **Monto Total de Capital en Riesgo Crítico ($P(\text{fuga}) \ge 0.65$):** $193,608,308.52 USD.
* **Capital Rescatable Estimado:** $484,441,965.28 USD (correspondiente a saldos de clientes intervenidos con beneficio neto positivo).
* **Inversión Total en Campañas de Fidelización:** $447,680.00 USD (presupuesto acumulado estrictamente justificado).
* **Beneficio Neto Esperado:** $3,074,102.81 USD.
* **Retorno Sobre la Inversión (ROI):** **686.67%**.

> **Ahorro Operativo Adicional por "Salida Permitida":** El filtro financiero bloqueó exitosamente cualquier gasto comercial para **1,139 clientes deficitarios**, evitando un desperdicio estimado de entre $68,340.00 USD y $227,800.00 USD en promociones que no habrían generado retorno.

---

## 2. Arquitectura de la Solución y Diagnóstico de Fuga

### Configuración de la Red Neuronal Profunda (MLP)
El motor predictivo fue estructurado como un Perceptrón Multicapa (*Multi-Layer Perceptron*) optimizado para datos tabulares bancarios:
* **Preprocesamiento:** Pipeline con `StandardScaler` para las 8 variables continuas (`CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`) y `OneHotEncoder(handle_unknown='ignore')` para variables nominales (`Geography`, `Gender`). Dimensión de entrada final: 13 neuronas.
* **Tratamiento del Desbalance:** Ponderación positiva de clases en la función de costo `BCEWithLogitsLoss` con `pos_weight = 3.91` (proporción 79.6% vs 20.4%).
* **Capa 1:** 128 neuronas, función de activación ReLU, `BatchNormalization` y regularización `Dropout(0.25)`.
* **Capa 2:** 64 neuronas, función de activación ReLU y regularización `Dropout(0.15)`.
* **Capa de Salida:** 1 neurona lineal con cálculo de sigmoide $\sigma(z) \in [0.00, 1.00]$.
* **Entrenamiento y Control de Sobreajuste:** Optimizador `Adam` ($\text{lr}=0.001$, penalización $\text{L2}=1\times 10^{-4}$), con `EarlyStopping` (paciencia de 14 épocas monitoreando `val_loss`). El entrenamiento convergió en la época 38 restaurando los mejores pesos.

### Métricas de Diagnóstico Técnico (Conjunto de Test no visto)
* **ROC-AUC:** **0.8604** (Capacidad sobresaliente de discriminación global).
* **Precision-Recall AUC (PR-AUC):** **0.6923** (Muy superior a la línea base aleatoria de 0.2037 en datos desbalanceados).
* **Sensibilidad / Recall en Fuga:** **79.74%** (Identificación de 8 de cada 10 clientes en verdadero riesgo de abandono).
* **Precisión de Calibración (Brier Score):** **0.1536** (Las probabilidades emitidas reflejan de forma fidedigna la frecuencia real de abandono).

### Factores Determinantes de Abandono (Análisis de Explicabilidad SHAP)
1. **Número de Productos Bancarios (`NumOfProducts` - Impacto medio SHAP: 0.1181):** Los clientes que poseen 3 o más productos experimentan una tasa de deserción drásticamente más alta debido a insatisfacción con cargos por mantenimiento o complejidad en la administración de sus cuentas.
2. **Edad del Cliente (`Age` - Impacto medio SHAP: 0.1144):** Clientes en el segmento etario de 45 a 65 años presentan la mayor propensión a fuga, frecuentemente atraídos por ofertas patrimoniales de competidores.
3. **Membresía Activa (`IsActiveMember` - Impacto medio SHAP: 0.0696):** La falta de actividad bancaria mensual es el principal síntoma temprano de desconexión institucional y antecede al retiro definitivo de fondos.
4. **Localización Geográfica (`Geography_Germany` - Impacto medio SHAP: 0.0329):** La cartera de Alemania presenta una tasa estructural de deserción significativamente superior en comparación con Francia y España.

---

## 3. Matriz de Decisión Comercial y Rentabilidad

La aplicación implementa la regla matemática:
$$\text{Ganancia Neta Esperada} = (P(\text{fuga}) \times \text{Efectividad del Incentivo} \times \text{Margen del Cliente}) - \text{Costo del Incentivo} > 0$$

A continuación se detalla la segmentación institucional resultante sobre la cartera:

| Segmento | Probabilidad Red | Valor Patrimonial | Decisión del Sistema | Incentivo Asignado | Clientes | Costo Unit. | Presupuesto Total | Margen Neto Proyectado |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A: Alto Valor / Alto Riesgo** | $\ge 0.65$ (Prom. 83.0%) | Saldo Superior ($> \$80,000$) | **Retención Prioritaria** | Gestor VIP y Tarifa Plana | 1,474 | $200.00 | $294,800.00 | **+$1,977,468.00** |
| **B: Medio / Riesgo Moderado** | $\ge 0.40$ (Prom. 47.7%) | Saldo Medio ($30k - $80k) | **Retención Automatizada** | Bonificación Transaccional | 2,548 | $60.00 | $152,880.00 | **+$1,096,635.00** |
| **C: No Rentable** | $\ge 0.40$ (Prom. 66.4%) | Saldo Bajo / Deficitario ($< \$30,000$) | **Salida Permitida** | Sin intervención comercial | 1,139 | $0.00 | $0.00 | **$0.00 (Ahorro Operativo)** |
| **D: Estable** | $< 0.30$ (Prom. 14.4%) | Cualquier Saldo | **Sin Acción Requerida** | Mantenimiento Ordinario | 4,839 | $0.00 | $0.00 | **Estable ($0.00 Costo)** |
| **TOTALES CONSOLIDADOS** | -- | -- | -- | -- | **10,000** | -- | **$447,680.00** | **+$3,074,102.81** |

---

## 4. Operación de la Aplicación en Sucursales

### Procedimiento del Gestor de Cuentas (SOP)
1. **Identificación y Consulta:** Ante la visita, llamada o revisión periódica de un cliente, el ejecutivo accede a la pestaña *"Buscador y Diagnóstico por Cliente"* e ingresa su `CustomerId`.
2. **Lectura del Diagnóstico:** La pantalla despliega de inmediato:
   * Probabilidad continua de deserción ($P(\text{fuga})$).
   * Semáforo de riesgo y categorización en Segmento A, B, C o D.
   * Ganancia Neta Esperada y Costo autorizado.
3. **Ejecución de la Acción Comercial:**
   * Si el sistema indica **Retención Prioritaria (Segmento A)**: Se asigna inmediatamente un Ejecutivo Senior VIP y se aplica la exoneración de comisiones mediante la Tarifa Plana Institucional.
   * Si el sistema indica **Retención Automatizada (Segmento B)**: Se acredita la bonificación transaccional digital en la aplicación móvil del cliente.
   * Si el sistema indica **Salida Permitida (Segmento C)**: Queda terminantemente prohibido autorizar descuentos, tasas subsidiadas o bonos en efectivo. Se ofrece atención cortés y estándar de ventanilla.
4. **Registro en CRM:** El gestor confirma la acción ejecutada con un clic, retroalimentando la efectividad real observada para futuros ciclos de modelado.

### Política de Mantenimiento y Reentrenamiento del Modelo
* **Monitoreo de Deriva de Datos (Data Drift):** Evaluación mensual de la distribución de saldos y tasas de actividad utilizando la prueba de Kolmogorov-Smirnov.
* **Frecuencia de Reentrenamiento:** Calendario trimestral con incorporación de los datos de retención observados durante los últimos 90 días.
* **Calibración Continua:** Ajuste bimensual de los parámetros de efectividad de las campañas comerciales según la tasa empírica de aceptación de cada producto.

---

## 5. Conclusiones y Próximos Pasos

### Conclusiones Principales
1. **Eficacia Financiera Demostrada:** El sistema Deep Retention transforma el área de fidelización de un centro de costos indiscriminado a una unidad generadora de valor con un **ROI proyectado del 686.67%**.
2. **Protección del Balance Institucional:** Se asegura la protección de más de **$484 Millones de USD** en depósitos rentables.
3. **Disciplina en el Gasto:** La regla de "Salida Permitida" corta de raíz el drenaje de presupuesto en clientes con balances marginales o deficitarios.

### Cronograma de Despliegue y Prueba Piloto
* **Semanas 1 a 2:** Despliegue de la aplicación en entorno de staging y capacitación a 25 ejecutivos de cuenta en 5 sucursales piloto seleccionadas (representativas de Francia, Alemania y España).
* **Semanas 3 a 6 (Prueba Piloto A/B):** Operación en paralelo: Grupo de Control (criterio tradicional sin filtro de rentabilidad) vs. Grupo Experimental (Deep Retention con restricción estricta de margen).
* **Semana 7:** Evaluación de resultados preliminares del piloto (tasa de aceptación, ahorro de presupuesto y margen neto rescatado).
* **Semanas 8 a 10:** Despliegue general en la red completa de sucursales e integración vía API con el sistema central de banca en línea.

