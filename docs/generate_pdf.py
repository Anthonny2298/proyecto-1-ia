"""
Generador automático del Informe Ejecutivo en formato PDF profesional
utilizando ReportLab.
"""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether
)

def build_pdf():
    pdf_path = os.path.join(os.path.dirname(__file__), 'Informe_Ejecutivo_Deep_Retention.pdf')
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Estilos personalizados
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1E3A8A'),
        fontName='Helvetica-Bold',
        spaceAfter=4
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#4B5563'),
        spaceAfter=10
    )

    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#1E3A8A'),
        fontName='Helvetica-Bold',
        spaceBefore=12,
        spaceAfter=6
    )

    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#1F2937'),
        fontName='Helvetica-Bold',
        spaceBefore=8,
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#111827'),
        spaceAfter=5
    )

    bullet_style = ParagraphStyle(
        'BulletStyle',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#1F2937'),
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=3
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        fontName='Helvetica-Bold',
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#111827'),
        alignment=1
    )

    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#1E3A8A'),
        alignment=1
    )

    story = []

    # Encabezado
    story.append(Paragraph("INFORME EJECUTIVO: SISTEMA \"DEEP RETENTION\"", title_style))
    story.append(Paragraph("<b>Para:</b> Dirección de Estrategia y Rentabilidad &nbsp;|&nbsp; <b>De:</b> Equipo de BI & Data Science &nbsp;|&nbsp; <b>Fecha:</b> Noviembre 2026", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceAfter=10))

    # 1. Resumen de Impacto Financiero y Retorno
    story.append(Paragraph("1. Resumen de Impacto Financiero y Retorno", h1_style))
    story.append(Paragraph(
        "Por disposición de la Junta Directiva, el sistema <b>Deep Retention</b> automatiza la selección de clientes propensos a fuga "
        "condicionando cada beneficio a la obtención de una ganancia neta positiva comprobable. Los resultados consolidados de cartera son:",
        body_style
    ))

    kpi_data = [
        [Paragraph("<b>Métrica de Negocio</b>", table_header_style), Paragraph("<b>Resultado Institucional</b>", table_header_style), Paragraph("<b>Impacto / Significado Estratégico</b>", table_header_style)],
        [Paragraph("Volumen Total Analizado", table_cell_style), Paragraph("10,000 clientes", table_cell_bold), Paragraph("100% de la cartera evaluada en tiempo real", table_cell_style)],
        [Paragraph("Capital en Riesgo Crítico (P ≥ 0.65)", table_cell_style), Paragraph("$193,608,308.52 USD", table_cell_bold), Paragraph("Balance en cuentas con inminente fuga", table_cell_style)],
        [Paragraph("Capital Rescatable Estimado", table_cell_style), Paragraph("$484,441,965.28 USD", table_cell_bold), Paragraph("Patrimonio en cuentas rentables aprobadas para retención", table_cell_style)],
        [Paragraph("Inversión Total en Fidelización", table_cell_style), Paragraph("$447,680.00 USD", table_cell_bold), Paragraph("Presupuesto comercial estrictamente justificado", table_cell_style)],
        [Paragraph("Beneficio Neto Esperado", table_cell_style), Paragraph("+$3,074,102.81 USD", table_cell_bold), Paragraph("Ganancia líquida deduciendo el costo de campañas", table_cell_style)],
        [Paragraph("Retorno de Inversión (ROI)", table_cell_style), Paragraph("686.67%", table_cell_bold), Paragraph("Por cada $1.00 invertido se rescatan $6.87 netos", table_cell_style)],
    ]
    t_kpi = Table(kpi_data, colWidths=[150, 130, 260])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#F8FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        "<b>Ahorro Operativo por 'Salida Permitida':</b> El filtro bloqueó gastos para <b>1,139 clientes deficitarios</b>, "
        "evitando un desperdicio estimado de entre <b>$68,340.00</b> y <b>$227,800.00 USD</b>.",
        body_style
    ))

    # 2. Arquitectura de la Solución y Diagnóstico de Fuga
    story.append(Paragraph("2. Arquitectura de la Solución y Diagnóstico de Fuga", h1_style))
    story.append(Paragraph(
        "<b>Configuración de la Red Neuronal:</b> Perceptrón Multicapa (MLP) de 3 capas. "
        "Preprocesamiento modular con <code>StandardScaler</code> (8 variables continuas) y <code>OneHotEncoder</code> (2 categóricas). "
        "Ponderación por desbalance con <code>pos_weight=3.91</code>. "
        "Capa 1: 128 neuronas (ReLU, BatchNorm, Dropout 0.25). Capa 2: 64 neuronas (ReLU, Dropout 0.15). "
        "Capa de Salida: 1 neurona lineal con función sigmoide continua. "
        "Entrenado con optimizador <code>Adam</code> (lr=0.001) y <code>EarlyStopping</code> (patience=14) convergiendo en la época 38.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Métricas de Diagnóstico Técnico (Test Set):</b> "
        "<b>ROC-AUC: 0.8604</b> | <b>Precision-Recall AUC: 0.6923</b> | <b>Sensibilidad / Recall: 79.74%</b> | <b>Precisión de Calibración (Brier): 0.1536</b>.",
        body_style
    ))
    story.append(Paragraph("<b>Factores Determinantes de Abandono (Interpretabilidad SHAP):</b>", h2_style))
    story.append(Paragraph("• <b>1. Número de Productos Bancarios (NumOfProducts, impacto |SHAP|=0.1181):</b> Clientes con ≥3 productos muestran alta propensión de deserción por fricción de mantenimiento o costos acumulados.", bullet_style))
    story.append(Paragraph("• <b>2. Edad del Cliente (Age, impacto |SHAP|=0.1144):</b> El segmento de 45 a 65 años registra la mayor fuga patrimonial atraído por ofertas externas.", bullet_style))
    story.append(Paragraph("• <b>3. Actividad Transaccional (IsActiveMember, impacto |SHAP|=0.0696):</b> La ausencia de actividad mensual es el síntoma precursor más confiable de abandono.", bullet_style))
    story.append(Paragraph("• <b>4. Segmento Geográfico (Geography_Germany, impacto |SHAP|=0.0329):</b> Mayor tasa estructural de deserción en comparación con Francia y España.", bullet_style))

    # 3. Matriz de Decisión Comercial y Rentabilidad
    story.append(Paragraph("3. Matriz de Decisión Comercial y Rentabilidad", h1_style))
    story.append(Paragraph(
        "Regla de decisión matemática: <i>Ganancia Neta Esperada = (P(fuga) × Efectividad × Margen Cliente) - Costo Incentivo > 0</i>",
        body_style
    ))

    mat_data = [
        [Paragraph("<b>Segmento</b>", table_header_style), Paragraph("<b>Probabilidad</b>", table_header_style), Paragraph("<b>Saldo</b>", table_header_style), Paragraph("<b>Decisión</b>", table_header_style), Paragraph("<b>Incentivo</b>", table_header_style), Paragraph("<b>Clientes</b>", table_header_style), Paragraph("<b>Costo Unit.</b>", table_header_style), Paragraph("<b>Presupuesto</b>", table_header_style), Paragraph("<b>Retorno Neto</b>", table_header_style)],
        [Paragraph("A: Alto Valor", table_cell_style), Paragraph("≥ 0.65", table_cell_style), Paragraph("> $80k", table_cell_style), Paragraph("Retención Prioritaria", table_cell_bold), Paragraph("Gestor VIP + Tarifa", table_cell_style), Paragraph("1,474", table_cell_style), Paragraph("$200", table_cell_style), Paragraph("$294,800", table_cell_style), Paragraph("+$1,977,468", table_cell_bold)],
        [Paragraph("B: Medio", table_cell_style), Paragraph("≥ 0.40", table_cell_style), Paragraph("$30k - $80k", table_cell_style), Paragraph("Retención Automatizada", table_cell_bold), Paragraph("Bono Transaccional", table_cell_style), Paragraph("2,548", table_cell_style), Paragraph("$60", table_cell_style), Paragraph("$152,880", table_cell_style), Paragraph("+$1,096,635", table_cell_bold)],
        [Paragraph("C: No Rentable", table_cell_style), Paragraph("≥ 0.40", table_cell_style), Paragraph("< $30k", table_cell_style), Paragraph("Salida Permitida", table_cell_bold), Paragraph("Sin intervención", table_cell_style), Paragraph("1,139", table_cell_style), Paragraph("$0", table_cell_style), Paragraph("$0", table_cell_style), Paragraph("Ahorro Operat.", table_cell_bold)],
        [Paragraph("D: Estable", table_cell_style), Paragraph("< 0.30", table_cell_style), Paragraph("Cualquiera", table_cell_style), Paragraph("Sin Acción Requerida", table_cell_bold), Paragraph("Mantenimiento", table_cell_style), Paragraph("4,839", table_cell_style), Paragraph("$0", table_cell_style), Paragraph("$0", table_cell_style), Paragraph("Estable ($0)", table_cell_bold)],
        [Paragraph("<b>TOTALES</b>", table_header_style), Paragraph("--", table_header_style), Paragraph("--", table_header_style), Paragraph("--", table_header_style), Paragraph("--", table_header_style), Paragraph("<b>10,000</b>", table_header_style), Paragraph("--", table_header_style), Paragraph("<b>$447,680</b>", table_header_style), Paragraph("<b>+$3,074,103</b>", table_header_style)],
    ]
    t_mat = Table(mat_data, colWidths=[65, 55, 60, 95, 85, 45, 45, 55, 65])
    t_mat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E3A8A')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#1E3A8A')),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.HexColor('#F8FAFC'), colors.white]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_mat)
    story.append(Spacer(1, 6))

    # 4. Operación de la Aplicación en Sucursales
    story.append(Paragraph("4. Operación de la Aplicación en Sucursales", h1_style))
    story.append(Paragraph(
        "• <b>Protocolo del Gestor de Cuentas:</b> El ejecutivo ingresa a la aplicación Streamlit en la sucursal, consulta al cliente por su <code>CustomerId</code> o simula su perfil actual. "
        "El sistema despliega el puntaje continuo, el semáforo y la decisión comercial recomendada. Si el resultado es 'Salida Permitida', el sistema bloquea cualquier beneficio presupuestario. "
        "Si califica en A o B, se activa el paquete autorizado con registro en CRM.",
        body_style
    ))
    story.append(Paragraph(
        "• <b>Mantenimiento y Reentrenamiento del Modelo:</b> Evaluación mensual de deriva de variables (*data drift*) con Kolmogorov-Smirnov. "
        "Ciclo de reentrenamiento programado trimestralmente para incorporar nuevas dinámicas macroeconómicas y tasas de efectividad reales observadas.",
        body_style
    ))

    # 5. Conclusiones y Próximos Pasos
    story.append(Paragraph("5. Conclusiones y Próximos Pasos", h1_style))
    story.append(Paragraph(
        "1. <b>Viabilidad Financiera:</b> El sistema Deep Retention maximiza la rentabilidad al elevar el ROI de retención a <b>686.67%</b> y asegurar <b>+$3.07M USD</b> de margen neto.<br/>"
        "2. <b>Prueba Piloto A/B:</b> Se recomienda una prueba de 4 semanas en 5 sucursales clave (Grupo Control tradicional vs. Grupo Deep Retention).<br/>"
        "3. <b>Despliegue General:</b> Concluido el piloto, se habilitará la API para todas las sucursales y la banca digital.",
        body_style
    ))

    doc.build(story)
    print(f"PDF generado con éxito en: {pdf_path}")

if __name__ == '__main__':
    build_pdf()
