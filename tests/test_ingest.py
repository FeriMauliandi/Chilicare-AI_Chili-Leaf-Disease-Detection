import pytest

from ingest import _looks_like_noise, _retry_delay_seconds


def test_noise_filter_drops_web_navigation():
    text = (
        "Home\nAbout Company\nProduct Menu Toggle\n"
        "NPK PREMIUM\nNPK COMPACTION\nOrganic Fertilizer\n"
    )
    assert _looks_like_noise(text) is True


def test_noise_filter_drops_pdf_table_of_contents():
    text = (
        "DAFTAR ISI\n"
        "Kata Pengantar ........................................................... i\n"
        "Daftar Isi ................................................................. ii\n"
        "BAB I Pendahuluan ......................................................... 1\n"
        "BAB II Penyakit Penting ................................................... 11\n"
    )
    assert _looks_like_noise(text) is True


def test_noise_filter_keeps_prose():
    text = (
        "Bercak daun cercospora disebabkan oleh jamur Cercospora capsici. "
        "Spora menyebar melalui angin dan percikan air hujan. Petik daun yang "
        "terinfeksi lalu bakar agar tidak menular ke tanaman lain."
    )
    assert _looks_like_noise(text) is False


def test_retry_delay_uses_server_hint():
    error = Exception("429 RESOURCE_EXHAUSTED. Please retry in 27.4s.")
    assert _retry_delay_seconds(error, attempt=1) == pytest.approx(29.4)


def test_retry_delay_falls_back_to_exponential_backoff():
    error = Exception("connection reset")
    assert _retry_delay_seconds(error, attempt=1) == 10.0
    assert _retry_delay_seconds(error, attempt=3) == 40.0
    assert _retry_delay_seconds(error, attempt=10) == 60.0
