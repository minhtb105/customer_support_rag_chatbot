"""Voice endpoints — mocked providers, no real audio/keys in CI.

Covers: transcribe auth + 422 empty/oversize + red_flag on transcript,
speak truncate + fail-open text-only, daily-check missing/full + prefs,
no-key determinism (factory -> mock), frontend grep keys == 0.
"""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

AUDIO = ("audio.webm", b"\x1a\x45\xdf\xa3" + b"\x00" * 64, "audio/webm")


def _login(client, username, password="Test@123"):
    client.cookies.clear()
    r = client.post("/v1/auth/register", json={"username": username, "password": password})
    assert r.status_code in (200, 201), r.text
    r2 = client.post("/v1/auth/login-json", json={"username": username, "password": password})
    assert r2.status_code == 200, r2.text
    return r2.json()["user"]


@pytest.fixture
def voice_iso(monkeypatch, tmp_path):
    import src.voice.prefs as prefs
    monkeypatch.setattr(prefs, "VOICE_PREFS_PATH", tmp_path / "voice_prefs.json")
    monkeypatch.setenv("VOICE_PREFS_PATH", str(tmp_path / "voice_prefs.json"))
    import src.voice.providers as prov

    class FixedSTT:
        def transcribe(self, audio_bytes, content_type=""):
            return "mot tram hai muoi sau luc doi"

    monkeypatch.setattr(prov, "get_stt_provider", lambda: FixedSTT())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Force-mock TTS (D7): real models installed must never change CI.
    monkeypatch.setenv("VOICE_TTS_PROVIDER", "none")
    monkeypatch.setattr(prov.PiperProvider, "available", False)
    return tmp_path


def test_transcribe_401_no_auth(client, voice_iso):
    client.cookies.clear()
    r = client.post("/v1/voice/transcribe", files={"audio": AUDIO})
    assert r.status_code == 401, r.text


def test_transcribe_ok_and_no_redflag(client, voice_iso):
    import uuid
    u = _login(client, f"vv_{uuid.uuid4().hex[:6]}")
    r = client.post("/v1/voice/transcribe", files={"audio": AUDIO})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "mot tram hai muoi sau luc doi"
    assert body["emergency"] is False
    assert body["parse"]["value"] == 126
    assert body["parse"]["context"] == "fasting"
    assert body["parse"]["ambiguous"] is False
    assert "126" in body["parse"]["confirm_text"]


def test_transcribe_redflag_before_confirm(client, voice_iso, monkeypatch):
    import uuid
    import src.voice.providers as prov

    class PanicSTT:
        def transcribe(self, audio_bytes, content_type=""):
            return "vet thuong chay mu, sot cao"  # foot_infection red-flag

    monkeypatch.setattr(prov, "get_stt_provider", lambda: PanicSTT())
    _login(client, f"vv_{uuid.uuid4().hex[:6]}")
    r = client.post("/v1/voice/transcribe", files={"audio": AUDIO})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["emergency"] is True
    assert "115" in body.get("message", "")


def test_transcribe_422_empty_and_oversize(client, voice_iso):
    import uuid
    _login(client, f"vv_{uuid.uuid4().hex[:6]}")
    r = client.post("/v1/voice/transcribe",
                    files={"audio": ("empty.webm", b"", "audio/webm")})
    assert r.status_code == 422, r.text
    big = ("big.webm", b"\x00" * (5 * 1024 * 1024 + 1), "audio/webm")
    r = client.post("/v1/voice/transcribe", files={"audio": big})
    assert r.status_code == 422, r.text


def test_speak_truncate_fail_open(client, voice_iso, monkeypatch):
    import uuid
    import src.voice.providers as prov
    monkeypatch.setattr(prov, "synthesize", lambda text, region="Bac": None)
    _login(client, f"vv_{uuid.uuid4().hex[:6]}")
    r = client.post("/v1/voice/speak", json={"text": "x" * 800})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["audio_available"] is False
    assert len(body["text"]) == 500  # truncated ~500 (D6)


def test_daily_check_missing_then_full(client, voice_iso):
    import uuid
    u = _login(client, f"vv_{uuid.uuid4().hex[:6]}")
    r = client.get("/v1/voice/daily-check", params={"user_id": u["id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["missing_slots"] == ["fasting", "post_meal_2h"]
    assert "thiếu" in body["greeting"]
    # log fasting today -> only post_meal_2h missing
    from src.diabetes.glucose_tracker import add_log
    add_log(user_id=u["id"], value_mgdl=110, context="fasting")
    r = client.get("/v1/voice/daily-check", params={"user_id": u["id"]})
    assert r.json()["missing_slots"] == ["post_meal_2h"]
    add_log(user_id=u["id"], value_mgdl=140, context="post_meal_2h")
    r = client.get("/v1/voice/daily-check", params={"user_id": u["id"]})
    body = r.json()
    assert body["missing_slots"] == []
    assert "đủ" in body["greeting"]


def test_prefs_get_set_fail_open(client, voice_iso):
    import uuid
    u = _login(client, f"vv_{uuid.uuid4().hex[:6]}")
    r = client.get("/v1/voice/prefs", params={"user_id": u["id"]})
    assert r.status_code == 200, r.text
    assert r.json()["required_slots"] == ["fasting", "post_meal_2h"]
    assert r.json()["tts_voice_region"] == "Bac"
    r = client.post("/v1/voice/prefs", params={"user_id": u["id"]},
                    json={"required_slots": ["fasting"], "tts_voice_region": "Nam"})
    assert r.status_code == 200, r.text
    assert r.json()["required_slots"] == ["fasting"]
    assert r.json()["tts_voice_region"] == "Nam"


def test_no_key_factory_is_mock():
    import os
    os.environ.pop("OPENAI_API_KEY", None)
    from src.voice.providers import get_stt_provider, MockSTTProvider
    assert isinstance(get_stt_provider(), MockSTTProvider)


def test_frontend_has_no_server_keys():
    repo = Path(__file__).resolve().parents[1] / "frontend"
    hits = []
    for p in list(repo.rglob("*.ts")) + list(repo.rglob("*.tsx")):
        if "node_modules" in p.parts or ".next" in p.parts:
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for key in ("OPENAI_API_KEY", "TYPESAFE", "VinBase"):
            if key in txt:
                hits.append(f"{p.name}:{key}")
    assert hits == [], f"server keys leaked to frontend: {hits}"


class TestTTSProviderRouting:
    """Server TTS default (D4/D6/D7): env seam + region map + force-mock."""

    def test_none_provider_returns_none(self, monkeypatch):
        import src.voice.providers as prov
        monkeypatch.setenv("VOICE_TTS_PROVIDER", "none")
        assert prov.synthesize("xin chao", "Bac") is None

    def test_piper_unavailable_fail_open(self, monkeypatch):
        import src.voice.providers as prov
        monkeypatch.setenv("VOICE_TTS_PROVIDER", "piper")
        monkeypatch.setattr(prov.PiperProvider, "available", False)
        assert prov.synthesize("xin chao", "Bac") is None

    def test_unknown_region_falls_back_bac(self, monkeypatch):
        import src.voice.providers as prov
        assert prov._norm_region("Hanoi") == "Bac"
        assert prov._norm_region("") == "Bac"
        assert prov._norm_region("Nam") == "Nam"
        seen = {}
        monkeypatch.setenv("VOICE_TTS_PROVIDER", "piper")
        monkeypatch.setattr(prov.PiperProvider, "available", True)
        monkeypatch.setattr(
            prov.PiperProvider, "speak",
            lambda self, text, region="Bac": seen.setdefault("region", region))
        prov.synthesize("xin chao", "Hanoi")
        assert seen["region"] == "Hanoi"  # passthrough; provider norms to Bac
        assert prov._norm_region(seen["region"]) == "Bac"

    def test_region_voices_cover_all_regions(self):
        import src.voice.providers as prov
        for d in (prov.PIPER_REGION_VOICES, prov.KOKORO_REGION_VOICES):
            assert set(d) == {"Bac", "Trung", "Nam"}

    def test_kokoro_missing_deps_fail_open(self, monkeypatch):
        import src.voice.providers as prov
        monkeypatch.setenv("VOICE_TTS_PROVIDER", "kokoro")
        assert prov.synthesize("xin chao", "Bac") is None
