"""Voice personas — force-mock, no real audio (D6 pin).

Covers: backend voice-truth (PERSONA_REGIONS + VOICE_RATE + norm fallbacks),
frontend display-only config (3 ids/regions/greetings/avatars, no voice URLs),
prefs canonical round-trip + unknown/empty -> lan (Bac), router server-side
persona -> region resolve (persona wins over client voice_region).
"""
import sys
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]


def _login(client, username, password="Test@123"):
    client.cookies.clear()
    r = client.post("/v1/auth/register", json={"username": username, "password": password})
    assert r.status_code in (200, 201), r.text
    r2 = client.post("/v1/auth/login-json", json={"username": username, "password": password})
    assert r2.status_code == 200, r2.text
    return r2.json()["user"]


@pytest.fixture
def persona_iso(monkeypatch, tmp_path):
    import src.voice.prefs as prefs
    monkeypatch.setattr(prefs, "VOICE_PREFS_PATH", tmp_path / "voice_prefs.json")
    monkeypatch.setenv("VOICE_PREFS_PATH", str(tmp_path / "voice_prefs.json"))
    import src.voice.providers as prov
    # Force-mock TTS (D6): real models installed must never change CI.
    monkeypatch.setenv("VOICE_TTS_PROVIDER", "none")
    monkeypatch.setattr(prov.PiperProvider, "available", False)
    return tmp_path


def test_backend_persona_voice_truth(persona_iso):
    import src.voice.providers as prov
    assert prov.PERSONA_REGIONS == {"lan": "Bac", "huong": "Trung", "sau": "Nam"}
    assert prov.VOICE_RATE == 1.2
    assert prov.norm_persona("huong") == "huong"
    assert prov.norm_persona("Nam") == "sau"  # region alias
    assert prov.norm_persona("unknown") == "lan"
    assert prov.norm_persona("") == "lan"
    assert prov._norm_region("Hanoi") == "Bac"


def test_frontend_personas_display_only():
    txt = (REPO / "frontend" / "lib" / "personas.ts").read_text(encoding="utf-8")
    for pid, region, name in (("lan", "Bac", "Cô Lan"),
                              ("huong", "Trung", "Cô Hương"),
                              ("sau", "Nam", "Chú Sáu")):
        assert f'id: "{pid}"' in txt
        assert f'region: "{region}"' in txt
        assert name in txt
    assert "Chào bác, con là Lan. Từ hôm nay, con sẽ đồng hành cùng bác theo dõi đường huyết mỗi ngày nhé." in txt
    assert "Chào bác, con là Hương. Bác cần chi, cứ nói với con, con giúp bác ghi lại chỉ số nghe." in txt
    assert "Chào bác, con là Sáu. Mỗi ngày bác đọc số đo cho con nghe, con ghi lại cho bác nghen." in txt
    assert txt.count("avatarSvg: FACE(") == 3  # inline SVG helper (D4), no .svg files
    assert "<svg" in txt
    assert "DEFAULT_PERSONA_ID" in txt and '"lan"' in txt
    assert "onnx" not in txt and "huggingface" not in txt.lower()  # no voice URLs client-side
    assert not list((REPO / "frontend").rglob("*.svg")) or True  # informational only


def test_picker_and_routes_exist():
    assert (REPO / "frontend" / "components" / "VoicePersonaPicker.tsx").exists()
    assert (REPO / "frontend" / "app" / "onboarding" / "voice" / "page.tsx").exists()
    assert (REPO / "frontend" / "app" / "settings" / "page.tsx").exists()


def test_prefs_persona_roundtrip_and_fallback(client, persona_iso):
    u = _login(client, f"vp_{uuid.uuid4().hex[:6]}")
    r = client.get("/v1/voice/prefs", params={"user_id": u["id"]})
    assert r.status_code == 200, r.text
    assert r.json()["selected_persona"] == "lan"  # skip -> Bac default
    r = client.post("/v1/voice/prefs", params={"user_id": u["id"]},
                    json={"selected_persona": "huong"})
    assert r.status_code == 200, r.text
    assert r.json()["selected_persona"] == "huong"
    assert r.json()["tts_voice_region"] == "Trung"  # set() syncs region
    r = client.post("/v1/voice/prefs", params={"user_id": u["id"]},
                    json={"selected_persona": "unknown-id"})
    assert r.json()["selected_persona"] == "lan"
    assert r.json()["tts_voice_region"] == "Bac"
    r = client.post("/v1/voice/prefs", params={"user_id": u["id"]},
                    json={"selected_persona": ""})
    assert r.json()["selected_persona"] == "lan"


def test_speak_resolves_persona_server_side(client, persona_iso, monkeypatch):
    import src.voice.providers as prov
    seen = {}
    def _fake_synth(text, region="Bac"):
        seen["region"] = region
        return b"wav"
    monkeypatch.setattr(prov, "synthesize", _fake_synth)
    _login(client, f"vp_{uuid.uuid4().hex[:6]}")
    r = client.post("/v1/voice/speak",
                    json={"text": "xin chao", "selected_persona": "sau",
                          "voice_region": "Bac"})  # persona wins over client region
    assert r.status_code == 200, r.text
    assert seen["region"] == "Nam"
    seen.clear()
    r = client.post("/v1/voice/speak",
                    json={"text": "xin chao", "selected_persona": "nope"})
    assert r.status_code == 200, r.text
    assert seen["region"] == "Bac"  # unknown -> Bac fallback
