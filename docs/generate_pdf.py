"""
Genera docs/Informe_Ejecutivo_Deep_Retention.pdf a partir de
docs/Informe_Ejecutivo_Deep_Retention.md (unica fuente de verdad del informe).

Soporta el subconjunto de Markdown que usa el informe: titulos (#, ##, ###), parrafos,
listas (* , -, 1.), tablas, imagenes ![texto](ruta), **negrita**, `codigo` y separadores (---).

Uso:
    python docs/generate_pdf.py
"""

import os
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, Image, ListFlowable, ListItem, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
MD_PATH = os.path.join(DOCS_DIR, 'Informe_Ejecutivo_Deep_Retention.md')
PDF_PATH = os.path.join(DOCS_DIR, 'Informe_Ejecutivo_Deep_Retention.pdf')
BLUE = colors.HexColor('#1E3A8A')

_styles = getSampleStyleSheet()
STYLES = {
    'h1': ParagraphStyle('h1', parent=_styles['Heading1'], fontSize=18, leading=22, textColor=BLUE, spaceAfter=6),
    'h2': ParagraphStyle('h2', parent=_styles['Heading2'], fontSize=13, leading=17, textColor=BLUE,
                         spaceBefore=10, spaceAfter=4),
    'h3': ParagraphStyle('h3', parent=_styles['Heading3'], fontSize=11, leading=14, spaceBefore=6, spaceAfter=3),
    'body': ParagraphStyle('body', parent=_styles['Normal'], fontSize=9.5, leading=13, spaceAfter=4),
    'cell': ParagraphStyle('cell', parent=_styles['Normal'], fontSize=7.5, leading=9.5),
    'head': ParagraphStyle('head', parent=_styles['Normal'], fontSize=7.5, leading=9.5,
                           textColor=colors.white, fontName='Helvetica-Bold'),
    'caption': ParagraphStyle('caption', parent=_styles['Normal'], fontSize=8, leading=10,
                              textColor=colors.HexColor('#4B5563'), alignment=1, spaceAfter=6),
}


def inline(text: str) -> str:
    """Convierte negrita, cursiva y codigo en linea al markup de ReportLab."""
    text = escape(text.strip())
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)', r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
    return text.replace('\\$', '$')


def build_table(rows):
    cells = [[c.strip() for c in r.strip().strip('|').split('|')] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r':?-{2,}:?', c) for c in r)]
    data = [[Paragraph(inline(c), STYLES['head'] if i == 0 else STYLES['cell']) for c in r]
            for i, r in enumerate(cells)]
    table = Table(data, repeatRows=1, hAlign='LEFT', colWidths=[7.5 * inch / len(cells[0])] * len(cells[0]))
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), BLUE),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F1F5F9')]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#CBD5E1')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 3), ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    return [table, Spacer(1, 6)]


def build_list(items, ordered):
    flow = [ListItem(Paragraph(inline(t), STYLES['body']), leftIndent=12) for t in items]
    return [ListFlowable(flow, bulletType='1' if ordered else 'bullet', start='1' if ordered else None,
                         leftIndent=14, bulletFontSize=8)]


def build_image(alt, rel_path):
    path = os.path.join(DOCS_DIR, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Imagen del informe no encontrada: {path}. Ejecute 'python train.py'.")
    img = Image(path)
    scale = min(4.6 * inch / img.imageWidth, 3.0 * inch / img.imageHeight)
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    return [img, Paragraph(inline(alt), STYLES['caption'])]


def markdown_to_flowables(markdown: str):
    lines = markdown.splitlines()
    story, i = [], 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped == '---':
            story += [Spacer(1, 4), HRFlowable(width='100%', color=colors.HexColor('#CBD5E1')), Spacer(1, 4)]
            i += 1
        elif stripped.startswith('#'):
            level = len(stripped) - len(stripped.lstrip('#'))
            story.append(Paragraph(inline(stripped[level:]), STYLES[f'h{min(level, 3)}']))
            i += 1
        elif stripped.startswith('!['):
            m = re.match(r'!\[(.*?)\]\((.+?)\)', stripped)
            story += build_image(m.group(1), m.group(2))
            i += 1
        elif stripped.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                rows.append(lines[i])
                i += 1
            story += build_table(rows)
        elif re.match(r'^(\*|-|\d+\.)\s', stripped):
            ordered = bool(re.match(r'^\d+\.', stripped))
            items = []
            while i < len(lines) and re.match(r'^\s*(\*|-|\d+\.)\s', lines[i]):
                items.append(re.sub(r'^\s*(\*|-|\d+\.)\s+', '', lines[i]))
                i += 1
            story += build_list(items, ordered)
        else:
            para = [stripped.lstrip('> ')]
            i += 1
            while i < len(lines) and lines[i].strip() and not re.match(r'^\s*(#|\||!\[|\*\s|-\s|\d+\.\s|---)', lines[i]):
                para.append(lines[i].strip())
                i += 1
            story.append(Paragraph(inline(' '.join(para)), STYLES['body']))
    return story


def build_pdf(md_path: str = MD_PATH, pdf_path: str = PDF_PATH) -> str:
    with open(md_path, encoding='utf-8') as f:
        story = markdown_to_flowables(f.read())
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, leftMargin=0.5 * inch, rightMargin=0.5 * inch,
                            topMargin=0.5 * inch, bottomMargin=0.5 * inch,
                            title='Informe Ejecutivo - Deep Retention')
    doc.build(story)
    print(f"PDF generado en: {pdf_path}")
    return pdf_path


if __name__ == '__main__':
    build_pdf()
