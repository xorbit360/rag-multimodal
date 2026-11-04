from app.adapters.embeddings import HashEmbedding
from app.adapters.llm import FakeLLM
from app.adapters.vector_store import InMemoryVectorStore
from app.domain.models import INSUFFICIENT_ANSWER, Chunk
from app.domain.use_cases import AskUseCase


def _ask_with_chunks(chunks):
    store = InMemoryVectorStore()
    emb = HashEmbedding()
    if chunks:
        store.upsert(chunks, emb.embed([c.text for c in chunks]))
    return AskUseCase(FakeLLM(), emb, store)


def test_no_context_returns_explicit_refusal():
    ask = _ask_with_chunks([])
    answer = ask.ask("Cual es la capital de Francia?")
    assert answer.text == INSUFFICIENT_ANSWER
    assert answer.sources == []


def test_out_of_context_question_returns_refusal_even_with_docs():
    chunks = [Chunk(id="c1", document_id="d", document_name="Manual_Motor.pdf",
                    page=3, text="El par de apriete de los tornillos de culata es de 24 Nm.",
                    bbox=None)]
    ask = _ask_with_chunks(chunks)
    answer = ask.ask("Cual es la receta de la paella valenciana?")
    assert answer.text == INSUFFICIENT_ANSWER


def test_in_context_question_returns_answer_with_source():
    chunks = [Chunk(id="c1", document_id="d", document_name="Manual_Motor.pdf",
                    page=3, text="El par de apriete de los tornillos de culata es de 24 Nm.",
                    bbox=None, image_id="img1")]
    ask = _ask_with_chunks(chunks)
    answer = ask.ask("Cual es el par de apriete de los tornillos de culata?")
    assert "24 Nm" in answer.text
    assert answer.sources[0].document_name == "Manual_Motor.pdf"
    assert answer.sources[0].page == 3
    assert answer.image_id == "img1"
