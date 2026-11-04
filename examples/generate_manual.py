"""Genera examples/Manual_Motor.pdf.

DOCUMENTO DE EJEMPLO, INVENTADO PARA LA PRUEBA: no es un manual real de
ningun fabricante. Texto tecnico realista, una tabla de especificaciones,
una tabla de pares de apriete y un diagrama dibujado (PNG con Pillow)
incrustado, para que el parser lo extraiga como imagen con pagina y bbox.
"""
import os

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image as RLImage, PageBreak, Paragraph, Spacer,
                                Table, TableStyle)

HERE = os.path.dirname(os.path.abspath(__file__))


def make_diagram(path: str) -> None:
    img = Image.new("RGB", (900, 500), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        small = font
    d.rectangle([50, 150, 850, 380], outline="black", width=3)
    d.ellipse([180, 200, 330, 350], outline="black", width=3)
    d.text((215, 260), "Piston", fill="black", font=font)
    d.rectangle([480, 220, 700, 280], outline="black", width=3)
    d.text((520, 235), "Ciguenal", fill="black", font=font)
    d.line([330, 275, 480, 250], fill="black", width=3)
    d.text((350, 180), "Biela", fill="black", font=small)
    d.text((50, 420), "Figura 1. Diagrama del motor: piston, biela y ciguenal.",
           fill="black", font=small)
    d.text((50, 40), "DIAGRAMA DEL MOTOR", fill="black", font=font)
    img.save(path)


def build(output: str) -> None:
    from reportlab.platypus import SimpleDocTemplate

    diagram = os.path.join(HERE, "diagrama_motor.png")
    make_diagram(diagram)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Manual de Mantenimiento del Motor XM-200", styles["Title"]))
    story.append(Paragraph(
        "Documento de ejemplo creado para la prueba tecnica RAG Multimodal. "
        "No corresponde a un fabricante real.", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("1. Introduccion", styles["Heading1"]))
    story.append(Paragraph(
        "El motor XM-200 es un motor de combustion interna de cuatro tiempos, "
        "disenado para uso en maquinaria ligera. Este manual describe su "
        "mantenimiento preventivo, sus especificaciones y su diagrama interno. "
        "El mantenimiento preventivo debe realizarse cada 250 horas de uso.",
        styles["Normal"]))
    story.append(PageBreak())

    story.append(Paragraph("2. Especificaciones tecnicas", styles["Heading1"]))
    data = [
        ["Parametro", "Valor"],
        ["Potencia maxima", "12 kW a 3600 rpm"],
        ["Cilindrada", "420 cc"],
        ["Capacidad de aceite", "1,1 litros"],
        ["Tipo de combustible", "Gasolina sin plomo"],
        ["Sistema de arranque", "Electrico y manual"],
    ]
    table = Table(data, colWidths=[7 * cm, 7 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEEEEE")]),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "La potencia maxima del motor XM-200 es de 12 kW a 3600 rpm y su "
        "cilindrada es de 420 cc. La capacidad de aceite es de 1,1 litros.",
        styles["Normal"]))
    story.append(PageBreak())

    story.append(Paragraph("3. Mantenimiento y pares de apriete", styles["Heading1"]))
    story.append(Paragraph(
        "Antes de cualquier intervencion, apague el motor y dejelo enfriar "
        "durante 30 minutos. Cambie el aceite cada 250 horas de uso y revise "
        "el filtro de aire cada 100 horas.", styles["Normal"]))
    torque = [
        ["Componente", "Par de apriete"],
        ["Tornillos de culata", "24 Nm"],
        ["Bujia", "18 Nm"],
        ["Tapon de drenaje de aceite", "15 Nm"],
    ]
    t2 = Table(torque, colWidths=[7 * cm, 7 * cm])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    story.append(t2)
    story.append(Paragraph(
        "El par de apriete de los tornillos de culata es de 24 Nm. "
        "El par de apriete de la bujia es de 18 Nm.", styles["Normal"]))
    story.append(PageBreak())

    story.append(Paragraph("4. Diagrama del motor", styles["Heading1"]))
    story.append(Paragraph(
        "La Figura 1 muestra el diagrama del motor XM-200. En el diagrama se "
        "observan el piston, la biela y el ciguenal. El piston transmite el "
        "movimiento a la biela y la biela lo transmite al ciguenal, que "
        "convierte el movimiento lineal en movimiento rotativo.",
        styles["Normal"]))
    story.append(RLImage(diagram, width=15 * cm, height=8.3 * cm))
    story.append(PageBreak())

    story.append(Paragraph("5. Solucion de problemas", styles["Heading1"]))
    story.append(Paragraph(
        "Si el motor no arranca, revise primero el combustible, luego la bujia "
        "y por ultimo el filtro de aire. Si el motor pierde potencia, revise "
        "el filtro de aire y la calidad del combustible.", styles["Normal"]))

    SimpleDocTemplate(output, pagesize=A4).build(story)
    print(f"PDF generado: {output}")


if __name__ == "__main__":
    build(os.path.join(HERE, "Manual_Motor.pdf"))
