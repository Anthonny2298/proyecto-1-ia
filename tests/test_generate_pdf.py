import os
import sys

import pytest
from reportlab.platypus import Image, ListFlowable, Paragraph, Table

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'docs'))
import generate_pdf  # noqa: E402


def test_inline_markup():
    assert generate_pdf.inline('**ROI** de `x` < 5') == '<b>ROI</b> de <font face="Courier">x</font> &lt; 5'


def test_markdown_blocks_are_parsed():
    md = "# Titulo\n\nParrafo con **negrita**.\n\n* uno\n* dos\n\n| A | B |\n|---|---|\n| 1 | 2 |\n"
    story = generate_pdf.markdown_to_flowables(md)
    kinds = [type(f) for f in story]
    assert Paragraph in kinds and ListFlowable in kinds and Table in kinds
    table = next(f for f in story if isinstance(f, Table))
    assert len(table._cellvalues) == 2  # encabezado + 1 fila (el separador se descarta)


def test_missing_image_fails_closed():
    with pytest.raises(FileNotFoundError):
        generate_pdf.markdown_to_flowables("![fig](figures/no_existe.png)")


def test_build_pdf(tmp_path):
    fig = tmp_path / 'fig.png'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.plot([0, 1]); plt.savefig(fig); plt.close()
    md = tmp_path / 'inf.md'
    md.write_text(f"# Informe\n\n![Figura]({fig})\n\nTexto.\n", encoding='utf-8')
    out = tmp_path / 'inf.pdf'
    generate_pdf.build_pdf(str(md), str(out))
    assert out.read_bytes()[:4] == b'%PDF'
    assert any(isinstance(f, Image) for f in generate_pdf.markdown_to_flowables(md.read_text(encoding='utf-8')))
