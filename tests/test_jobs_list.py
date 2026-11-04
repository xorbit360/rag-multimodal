"""GET /jobs: lista solo lectura, sin pdf_path ni secretos.

El menu aparte 'Trabajos' del frontend usa este endpoint: muestra
documento, estado, progreso, chunks e imagenes, y nunca la ruta del
PDF guardado ni ningun secreto.
"""
import time

import fitz
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _pdf_bytes(tmp_path):
    path = tmp_path / "jobs.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "1. Prueba")
    page.insert_text((72, 100), "El par de apriete de los tornillos de culata es de 24 Nm.")
    doc.save(str(path))
    doc.close()
    return path.read_bytes()


def test_jobs_list_is_read_only_and_hides_pdf_path(tmp_path):
    resp = client.post("/ingest", files={"file": ("jobs.pdf", _pdf_bytes(tmp_path),
                                                 "application/pdf")})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    for _ in range(50):
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("completed", "failed"):
            break
        time.sleep(0.1)
    listing = client.get("/jobs")
    assert listing.status_code == 200
    jobs = listing.json()
    assert isinstance(jobs, list)
    found = [j for j in jobs if j["job_id"] == job_id]
    assert found, jobs
    assert found[0]["status"] == "completed"
    assert "pdf_path" not in found[0]
    assert "api_key" not in str(found[0]).lower()
