"""Genera examples/Manual_Motor.pdf (12 paginas).

DOCUMENTO DE EJEMPLO, INVENTADO PARA LA PRUEBA: no es un manual real de
ningun fabricante. Texto tecnico realista, tablas de especificaciones,
pares de apriete, mantenimiento y diagnostico, y CUATRO diagramas
dibujados (PNG con Pillow) incrustados como imagenes, para que el parser
los extraiga como imagenes con pagina y bbox (multimodal real: texto,
tablas e imagenes). Los datos clave del manual anterior se conservan
(culata 24 Nm, bujia 18 Nm, tapon de drenaje 15 Nm, 12 kW, 420 cc,
aceite 1,1 litros, mantenimiento cada 250 horas).
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


def _fonts():
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
        small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
        small = font
    return font, small


def _canvas(title):
    img = Image.new("RGB", (1000, 620), "white")
    d = ImageDraw.Draw(img)
    font, small = _fonts()
    d.text((40, 20), title, fill="black", font=font)
    return img, d, font, small


def _box(d, xy, label, font, fill=None):
    d.rectangle(xy, outline="black", width=3, fill=fill)
    x0, y0, x1, y1 = xy
    d.text(((x0 + x1) / 2 - 8 * len(label) / 2, (y0 + y1) / 2 - 12),
           label, fill="black", font=font)


def make_explotada(path: str) -> None:
    """Vista explotada: piezas apiladas con su nombre."""
    img, d, font, small = _canvas("VISTA EXPLOTADA DEL MOTOR XM-200")
    _box(d, [350, 90, 650, 150], "Culata", font)
    _box(d, [350, 180, 650, 230], "Valvulas", font)
    d.ellipse([420, 260, 580, 360], outline="black", width=3)
    d.text((465, 300), "Piston", fill="black", font=font)
    d.line([500, 360, 500, 440], fill="black", width=3)
    d.text((510, 390), "Biela", fill="black", font=small)
    _box(d, [380, 440, 620, 500], "Ciguenal", font)
    _box(d, [300, 520, 700, 570], "Bloque y carter", font)
    d.text((40, 595), "Figura 1. Vista explotada: culata, valvulas, piston, biela y ciguenal.",
           fill="black", font=small)
    img.save(path)


def make_refrigeracion(path: str) -> None:
    """Circuito de refrigeracion liquida en bucle."""
    img, d, font, small = _canvas("CIRCUITO DE REFRIGERACION")
    _box(d, [80, 120, 320, 200], "Radiador", font)
    _box(d, [680, 120, 920, 200], "Bomba de agua", font)
    _box(d, [80, 400, 320, 480], "Bloque motor", font)
    _box(d, [680, 400, 920, 480], "Termostato 82 C", font)
    d.line([320, 160, 680, 160], fill="black", width=3)
    d.line([800, 200, 800, 400], fill="black", width=3)
    d.line([680, 440, 320, 440], fill="black", width=3)
    d.line([200, 400, 200, 200], fill="black", width=3)
    d.text((380, 260), "Refrigerante 2,0 litros", fill="black", font=small)
    d.text((40, 595), "Figura 2. Circuito de refrigeracion: radiador, bomba, bloque y termostato.",
           fill="black", font=small)
    img.save(path)


def make_lubricacion(path: str) -> None:
    """Circuito de lubricacion: carter, bomba, filtro, cojinetes."""
    img, d, font, small = _canvas("CIRCUITO DE LUBRICACION")
    _box(d, [80, 120, 320, 200], "Carter", font)
    _box(d, [680, 120, 920, 200], "Bomba de aceite", font)
    _box(d, [80, 400, 320, 480], "Filtro de aceite", font)
    _box(d, [680, 400, 920, 480], "Cojinetes", font)
    d.line([320, 160, 680, 160], fill="black", width=3)
    d.line([800, 200, 800, 400], fill="black", width=3)
    d.line([680, 440, 320, 440], fill="black", width=3)
    d.line([200, 400, 200, 200], fill="black", width=3)
    d.text((380, 260), "Presion 2,5 - 4,0 bar en caliente", fill="black", font=small)
    d.text((40, 595), "Figura 3. Circuito de lubricacion: carter, bomba, filtro y cojinetes.",
           fill="black", font=small)
    img.save(path)


def make_electrico(path: str) -> None:
    """Esquema electrico simplificado: bateria, arranque, bobina, bujia."""
    img, d, font, small = _canvas("ESQUEMA ELECTRICO Y ARRANQUE")
    d.ellipse([80, 120, 200, 240], outline="black", width=3)
    d.text((105, 170), "12 V", fill="black", font=font)
    d.text((90, 250), "Bateria 18 Ah", fill="black", font=small)
    _box(d, [400, 120, 620, 200], "Motor arranque", font)
    _box(d, [400, 400, 620, 480], "Bobina", font)
    _box(d, [760, 400, 920, 480], "Bujia", font)
    d.line([200, 180, 400, 160], fill="black", width=3)
    d.line([140, 240, 140, 440], fill="black", width=3)
    d.line([140, 440, 400, 440], fill="black", width=3)
    d.line([620, 440, 760, 440], fill="black", width=3)
    d.text((380, 260), "Alternador 12 V 15 A", fill="black", font=small)
    d.text((40, 595), "Figura 4. Esquema electrico: bateria, arranque, bobina y bujia.",
           fill="black", font=small)
    img.save(path)


def _table(data, widths=(7 * cm, 7 * cm)):
    t = Table(data, colWidths=list(widths))
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#333333")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEEEEE")]),
    ]))
    return t


def build(output: str) -> None:
    from reportlab.platypus import SimpleDocTemplate

    diagrams = {
        "explotada": os.path.join(HERE, "diagrama_explotada.png"),
        "refrigeracion": os.path.join(HERE, "diagrama_refrigeracion.png"),
        "lubricacion": os.path.join(HERE, "diagrama_lubricacion.png"),
        "electrico": os.path.join(HERE, "diagrama_electrico.png"),
    }
    make_explotada(diagrams["explotada"])
    make_refrigeracion(diagrams["refrigeracion"])
    make_lubricacion(diagrams["lubricacion"])
    make_electrico(diagrams["electrico"])
    legacy = os.path.join(HERE, "diagrama_motor.png")
    if os.path.exists(legacy):
        os.remove(legacy)  # sustituido por los 4 diagramas nuevos

    styles = getSampleStyleSheet()
    N, H1 = styles["Normal"], styles["Heading1"]
    story = []

    # Pagina 1: portada e introduccion
    story.append(Paragraph("Manual de Mantenimiento del Motor XM-200", styles["Title"]))
    story.append(Paragraph(
        "Documento de ejemplo creado para la prueba tecnica RAG Multimodal. "
        "No corresponde a un fabricante real.", N))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("1. Introduccion", H1))
    story.append(Paragraph(
        "El motor XM-200 es un motor de combustion interna de cuatro tiempos, "
        "disenado para uso en maquinaria ligera. Este manual describe su "
        "mantenimiento preventivo, sus especificaciones, sus circuitos de "
        "refrigeracion, lubricacion y arranque electrico, y el diagnostico "
        "de fallas mas frecuentes. El mantenimiento preventivo debe "
        "realizarse cada 250 horas de uso.", N))
    story.append(Paragraph(
        "Antes de cualquier intervencion, apague el motor y dejelo enfriar "
        "durante 30 minutos. Trabaje en un lugar ventilado, sin fumar y "
        "lejos de llamas abiertas.", N))
    story.append(PageBreak())

    # Pagina 2: especificaciones tecnicas
    story.append(Paragraph("2. Especificaciones tecnicas", H1))
    story.append(_table([
        ["Parametro", "Valor"],
        ["Potencia maxima", "12 kW a 3600 rpm"],
        ["Cilindrada", "420 cc"],
        ["Relacion de compresion", "9,5:1"],
        ["Capacidad de aceite", "1,1 litros"],
        ["Capacidad de refrigerante", "2,0 litros"],
        ["Tipo de combustible", "Gasolina sin plomo"],
        ["Sistema de arranque", "Electrico y manual"],
        ["Bateria", "12 V 18 Ah"],
        ["Peso en seco", "32 kg"],
    ]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "La potencia maxima del motor XM-200 es de 12 kW a 3600 rpm y su "
        "cilindrada es de 420 cc. La capacidad de aceite es de 1,1 litros "
        "y la capacidad de refrigerante es de 2,0 litros.", N))
    story.append(PageBreak())

    # Pagina 3: componentes + vista explotada
    story.append(Paragraph("3. Componentes principales", H1))
    story.append(Paragraph(
        "La Figura 1 muestra la vista explotada del motor XM-200. En el "
        "diagrama se observan la culata, las valvulas, el piston, la biela "
        "y el ciguenal. El piston transmite el movimiento a la biela y la "
        "biela lo transmite al ciguenal, que convierte el movimiento lineal "
        "en movimiento rotativo. La culata cierra la camara de combustion y "
        "alberga las valvulas de admision y escape.", N))
    story.append(RLImage(diagrams["explotada"], width=15 * cm, height=9.3 * cm))
    story.append(PageBreak())

    # Pagina 4: pares de apriete
    story.append(Paragraph("4. Pares de apriete", H1))
    story.append(Paragraph(
        "Apriete siempre con torquimetro y en el orden indicado. Un apriete "
        "excesivo dana las roscas y un apriete insuficiente provoca fugas.", N))
    story.append(_table([
        ["Componente", "Par de apriete"],
        ["Tornillos de culata", "24 Nm"],
        ["Bujia", "18 Nm"],
        ["Tapon de drenaje de aceite", "15 Nm"],
        ["Tapa de valvulas", "10 Nm"],
        ["Carter", "12 Nm"],
        ["Soporte del motor", "32 Nm"],
        ["Volante de inercia", "45 Nm"],
    ]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "El par de apriete de los tornillos de culata es de 24 Nm. "
        "El par de apriete de la bujia es de 18 Nm.", N))
    story.append(PageBreak())

    # Pagina 5: plan de mantenimiento
    story.append(Paragraph("5. Plan de mantenimiento", H1))
    story.append(_table([
        ["Tarea", "Intervalo"],
        ["Cambio de aceite", "Cada 250 horas"],
        ["Revision del filtro de aire", "Cada 100 horas"],
        ["Cambio de bujia", "Cada 200 horas"],
        ["Cambio de refrigerante", "Cada 500 horas"],
        ["Revision de holgura de valvulas", "Cada 500 horas"],
        ["Cambio de filtro de combustible", "Cada 300 horas"],
        ["Revision de la correa", "Cada 1000 horas"],
    ]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Cambie el aceite cada 250 horas de uso y revise el filtro de aire "
        "cada 100 horas. El mantenimiento preventivo debe realizarse cada "
        "250 horas de uso.", N))
    story.append(PageBreak())

    # Pagina 6: refrigeracion
    story.append(Paragraph("6. Sistema de refrigeracion", H1))
    story.append(Paragraph(
        "La Figura 2 muestra el circuito de refrigeracion del motor XM-200. "
        "El refrigerante circula del radiador a la bomba de agua, de ahi al "
        "bloque motor y vuelve por el termostato, que abre a 82 grados "
        "centigrados. La capacidad de refrigerante es de 2,0 litros. Si el "
        "motor se sobrecalienta, revise el nivel de refrigerante, el "
        "termostato y que el radiador no este obstruido.", N))
    story.append(RLImage(diagrams["refrigeracion"], width=15 * cm, height=9.3 * cm))
    story.append(PageBreak())

    # Pagina 7: lubricacion
    story.append(Paragraph("7. Sistema de lubricacion", H1))
    story.append(Paragraph(
        "La Figura 3 muestra el circuito de lubricacion del motor XM-200. "
        "La bomba toma el aceite del carter, lo pasa por el filtro y lo "
        "envia a los cojinetes del ciguenal y del arbol de levas. La presion "
        "de aceite en caliente debe estar entre 2,5 y 4,0 bar. Use aceite "
        "SAE 10W-30 y respete la capacidad de 1,1 litros: un nivel bajo "
        "dana los cojinetes y un nivel alto provoca humo azul.", N))
    story.append(RLImage(diagrams["lubricacion"], width=15 * cm, height=9.3 * cm))
    story.append(PageBreak())

    # Pagina 8: electrico y arranque
    story.append(Paragraph("8. Sistema electrico y arranque", H1))
    story.append(Paragraph(
        "La Figura 4 muestra el esquema electrico del motor XM-200. La "
        "bateria de 12 V y 18 Ah alimenta el motor de arranque y, a traves "
        "de la bobina, la bujia. El alternador de 12 V y 15 A recarga la "
        "bateria con el motor en marcha. Si el motor no arranca, revise "
        "primero el combustible, luego la bujia y por ultimo el filtro de "
        "aire.", N))
    story.append(RLImage(diagrams["electrico"], width=15 * cm, height=9.3 * cm))
    story.append(PageBreak())

    # Pagina 9: cambio de aceite
    story.append(Paragraph("9. Procedimiento: cambio de aceite", H1))
    story.append(Paragraph(
        "1. Apague el motor y dejelo enfriar durante 30 minutos.<br/>"
        "2. Coloque un recipiente bajo el carter y retire el tapon de "
        "drenaje.<br/>"
        "3. Deje escurrir todo el aceite usado y limpie el tapon.<br/>"
        "4. Apriete el tapon de drenaje de aceite a 15 Nm.<br/>"
        "5. Cambie el filtro de aceite y llene con 1,1 litros de aceite "
        "SAE 10W-30.<br/>"
        "6. Arranque el motor 2 minutos, apaguelo y verifique el nivel y "
        "que no haya fugas.", N))
    story.append(PageBreak())

    # Pagina 10: diagnostico de fallas
    story.append(Paragraph("10. Diagnostico de fallas", H1))
    story.append(_table([
        ["Sintoma", "Causa probable"],
        ["El motor no arranca", "Sin combustible, bujia sucia o filtro tapado"],
        ["Perdida de potencia", "Filtro de aire sucio o combustible viejo"],
        ["Sobrecalentamiento", "Poco refrigerante o termostato atascado"],
        ["Humo azul por el escape", "Exceso de aceite o anillos gastados"],
        ["Vibracion excesiva", "Soporte del motor flojo o helice danada"],
        ["Consumo alto de combustible", "Filtro de aire sucio o bujia gastada"],
    ]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Si el motor no arranca, revise primero el combustible, luego la "
        "bujia y por ultimo el filtro de aire. Si el motor pierde potencia, "
        "revise el filtro de aire y la calidad del combustible.", N))
    story.append(PageBreak())

    # Pagina 11: seguridad y almacenamiento
    story.append(Paragraph("11. Seguridad y almacenamiento", H1))
    story.append(Paragraph(
        "Use guantes y gafas en toda intervencion. No fume cerca del "
        "combustible y no guarde trapos con aceite junto al motor. Para "
        "almacenar el motor mas de 30 dias, vacie el combustible, cambie el "
        "aceite, limpie el filtro de aire y guardelo en un lugar seco y "
        "cubierto. Desconecte la bateria de 12 V antes de guardarlo.", N))
    story.append(Paragraph(
        "El combustible viejo forma depositos que tapan el carburador: "
        "si el motor estuvo guardado, drene el deposito antes de arrancar.",
        N))
    story.append(PageBreak())

    # Pagina 12: registro y glosario
    story.append(Paragraph("12. Registro de mantenimiento y glosario", H1))
    story.append(_table([
        ["Fecha", "Horas", "Tarea realizada"],
        ["", ""],
        ["", ""],
        ["", ""],
    ]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "Glosario: culata es la tapa superior que cierra la camara de "
        "combustion; ciguenal es el eje que convierte el movimiento lineal "
        "del piston en rotativo; holgura de valvulas es la separacion entre "
        "la valvula y el balancin, que se revisa cada 500 horas.", N))

    SimpleDocTemplate(output, pagesize=A4).build(story)
    print(f"PDF generado: {output}")


if __name__ == "__main__":
    build(os.path.join(HERE, "Manual_Motor.pdf"))
