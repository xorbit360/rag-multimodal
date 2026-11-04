import fitz

from app.adapters.pdf_parser import PyMuPDFParser


def _make_pdf(tmp_path):
    path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. Mantenimiento")
    page.insert_text((72, 100), "Cambie el aceite cada 250 horas.")
    # Imagen raster pequena
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 10, 10))
    pix.set_rect(pix.irect, (255, 0, 0))
    img_path = tmp_path / "red.png"
    pix.save(str(img_path))
    page.insert_image(fitz.Rect(72, 150, 172, 250), filename=str(img_path))
    doc.save(str(path))
    doc.close()
    return str(path)


def test_parser_extracts_text_with_bbox_and_image(tmp_path):
    parser = PyMuPDFParser(images_dir=str(tmp_path / "images"))
    parsed = parser.parse(_make_pdf(tmp_path), "d1", "test.pdf")
    assert any("250 horas" in b.text for b in parsed.blocks)
    text_block = next(b for b in parsed.blocks if "250 horas" in b.text)
    assert text_block.page == 1
    assert text_block.bbox.x0 >= 0 and text_block.bbox.y0 >= 0
    assert len(parsed.images) == 1
    assert parsed.images[0].page == 1
