import time

import fitz
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _pdf_bytes(tmp_path):
    path = tmp_path / "api.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. Mantenimiento")
    page.insert_text((72, 100), "El par de apriete de los tornillos de culata es de 24 Nm.")
    doc.save(str(path))
    doc.close()
    return path.read_bytes()


def test_health_ok():
    assert client.get("/health").status_code == 200


def test_ingest_rejects_non_pdf():
    resp = client.post("/ingest", files={"file": ("x.txt", b"hola", "text/plain")})
    assert resp.status_code == 400


def test_ingest_then_ask(tmp_path):
    resp = client.post("/ingest", files={"file": ("api.pdf", _pdf_bytes(tmp_path),
                                                 "application/pdf")})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    for _ in range(50):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    assert job["status"] == "completed", job
    assert job["chunks_count"] >= 1
    answer = client.post("/ask", json={"question": "Cual es el par de apriete de los tornillos de culata?"}).json()
    assert "24 Nm" in answer["answer"]
    assert answer["sources"]


def test_ask_without_context_refuses(tmp_path):
    # Pregunta totalmente ajena a cualquier documento indexado en este test
    answer = client.post("/ask", json={"question": "Cual es la receta secreta de la paella valenciana con azafran de otro planeta?"}).json()
    assert answer["answer"] == "No tengo información suficiente en los documentos para responder eso."
    assert answer["sources"] == []
