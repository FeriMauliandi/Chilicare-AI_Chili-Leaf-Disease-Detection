from langchain_core.documents import Document

from src.ingestion.labeler import (
    GENERAL_LABEL,
    assign_labels,
    infer_label,
    score_labels,
)


def test_infer_label_bercak():
    text = "Bercak daun cercospora disebabkan spora jamur yang menyebar lewat angin."
    assert infer_label(text) == "Bercak"


def test_infer_label_keriting():
    text = "Daun mengeriting ke atas akibat serangan thrips dan tungau."
    assert infer_label(text) == "Keriting"


def test_infer_label_kuning():
    text = "Daun menguning karena virus gemini yang ditularkan kutu kebul."
    assert infer_label(text) == "Kuning"


def test_infer_label_whitefly():
    text = "Kutu kebul (Bemisia tabaci) menghasilkan embun madu dan memicu jelaga."
    assert infer_label(text) == "Whitefly"


def test_infer_label_returns_umum_when_text_is_ambiguous():
    assert infer_label("Selamat datang di situs resmi kami.") == GENERAL_LABEL


def test_score_labels_returns_score_per_canonical_label():
    scores = score_labels("bercak daun cercospora")
    assert scores["Bercak"] > scores["Kuning"]
    assert set(scores) == {"Bercak", "Keriting", "Kuning", "Whitefly", "Sehat"}


def test_assign_labels_preserves_curated_label():
    doc = Document(
        page_content="Bercak daun cercospora dan spora jamur.",
        metadata={"label": "Sehat"},
    )
    result = assign_labels([doc])
    assert result[0].metadata["label"] == "Sehat"


def test_assign_labels_tags_documents_without_label():
    doc = Document(page_content="Serangan thrips membuat daun mengeriting.", metadata={})
    result = assign_labels([doc])
    assert result[0].metadata["label"] == "Keriting"
