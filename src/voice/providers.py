"""STT/TTS provider abstraction — merged (D8) + Server TTS Default (2026-09-26).

Step-0 spike (no code, web research 2026-09-26):
- Piper-vi: 3 official voices in rhasspy/piper-voices (vi_VN): vais1000-medium
  (female, ~64MB), 25hours_single-low (male, ~63MB — speaches-ai mirror shows
  63.1MB model.onnx), vivos-x_low (female, ~32MB). All use espeak voice "vi".
  piper-tts 1.2.0 wheel embeds espeak-ng via espeakbridge (no separate
  espeak-ng install needed on Windows). Community bucket bonelag/voice has
  40 VI models with explicit Bac/Nam tags (espeak vi / vi-vn-x-south) —
  alternative for certified region accents. Fits 60-120MB budget with lazy
  per-region download (one voice at a time, not all 3).
- Kokoro-VN: REAL (contextboxai/Kokoro-Vietnamese: kokoro_vi.onnx + named
  voicepacks diem_trinh/mai_linh/... + vig2p G2P, no espeak-ng needed).
  BUT: base 82M fp32 ~326MB / fp16 ~163MB / int8 ~88MB + voicepack + custom
  package chain (kokoro-vietnamese + vig2p + onnxruntime) → over the
  60-120MB budget; voicepacks are speaker-named (no Bac/Trung/Nam mapping).
  Verdict: NOT viable within budget → Piper-only wiring (per abort clause);
  kokoro kept as env-switched optional seam, NOT forced.

TTS default: server Piper (VOICE_TTS_PROVIDER=piper) with lazy per-region
download; browser speechSynthesis stays last-resort fallback in the widget.
No new requirements deps (lazy import + optional-install docs only).
"""
from __future__ import annotations

import io
import os
import wave
from pathlib import Path
from typing import Dict, Optional, Tuple


def _model_dir() -> Path:
    return Path(os.getenv("VOICE_MODEL_DIR", "metadata/voice_models"))


# Official Piper VI voices (rhasspy/piper-voices, espeak voice "vi").
# Speakers differ (not certified region accents) — mapping is best-effort
# demo default; certified Bac/Nam accents -> bonelag/voice bucket (see docs).
_HF = "https://huggingface.co"
PIPER_REGION_VOICES: Dict[str, Tuple[str, str]] = {
    "Bac": (
        f"{_HF}/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx",
        f"{_HF}/rhasspy/piper-voices/resolve/main/vi/vi_VN/vais1000/medium/vi_VN-vais1000-medium.onnx.json",
    ),
    "Trung": (
        f"{_HF}/rhasspy/piper-voices/resolve/main/vi/vi_VN/vivos/x_low/vi_VN-vivos-x_low.onnx",
        f"{_HF}/rhasspy/piper-voices/resolve/main/vi/vi_VN/vivos/x_low/vi_VN-vivos-x_low.onnx.json",
    ),
    "Nam": (
        f"{_HF}/speaches-ai/piper-vi_VN-25hours_single-low/resolve/main/model.onnx",
        f"{_HF}/speaches-ai/piper-vi_VN-25hours_single-low/resolve/main/config.json",
    ),
}

# Kokoro-VN voicepacks are speaker-named (no certified Bac/Trung/Nam mapping).
# Distinct packs listed for A/B by ear; unknown -> Bac pack.
KOKORO_REGION_VOICES: Dict[str, str] = {
    "Bac": "diem_trinh",
    "Trung": "mai_linh",
    "Nam": "hung_thinh",
}
# Back-compat alias (D6 pin: REGION_VOICES dict + unknown->Bac fallback).
REGION_VOICES = {"piper": PIPER_REGION_VOICES, "kokoro": KOKORO_REGION_VOICES}

# Voice personas (2026-09-27) — backend is voice-truth (D1 pin).
# Frontend personas.ts is display-only {id,region,name,avatarSvg,greeting};
# server resolves selected_persona -> region, then synthesize(text, region).
# Single slower rate for elderly (D3 pin: 1.2, no per-persona rates).
VOICE_RATE = 1.2
PERSONA_REGIONS = {"lan": "Bac", "huong": "Trung", "sau": "Nam"}


def norm_persona(pid: str = "lan") -> str:
    """Canonical persona id; unknown/empty -> lan (Bac default, D5 pin)."""
    s = (pid or "").strip().lower()
    if s in PERSONA_REGIONS:
        return s
    alias = {"bac": "lan", "north": "lan",
             "trung": "huong", "central": "huong",
             "nam": "sau", "south": "sau"}
    return alias.get(s, "lan")


def _norm_region(region: str = "Bac") -> str:
    s = (region or "").strip().lower()
    table = {"bac": "Bac", "north": "Bac",
             "trung": "Trung", "central": "Trung",
             "nam": "Nam", "south": "Nam"}
    return table.get(s, "Bac")  # unknown -> Bac fallback


class MockSTTProvider:
    """Deterministic stub for CI / no-key envs."""

    def __init__(self, text: str = "mot tram hai muoi sau luc doi"):
        self._text = text

    def transcribe(self, audio_bytes: bytes, content_type: str = "") -> str:
        return self._text


class WhisperAPIProvider:
    """OpenAI Whisper API — key server-side only, never leaves the backend."""

    def __init__(self, api_key: Optional[str] = None):
        self._key = api_key or os.getenv("OPENAI_API_KEY", "")

    def transcribe(self, audio_bytes: bytes, content_type: str = "") -> str:
        if not self._key:
            raise RuntimeError("OPENAI_API_KEY missing")
        import tempfile
        from openai import OpenAI  # local import: no cost when mocked
        suffix = ".webm"
        if "wav" in (content_type or ""):
            suffix = ".wav"
        elif "mpeg" in (content_type or "") or "mp3" in (content_type or ""):
            suffix = ".mp3"
        client = OpenAI(api_key=self._key)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
            f.write(audio_bytes)
            f.flush()
            with open(f.name, "rb") as fh:
                resp = client.audio.transcriptions.create(
                    model="whisper-1", file=fh, language="vi")
        return (resp.text or "").strip()


class PhoSTTStub:
    """phoSTT (Zipformer-vi, CPU-only) — interface + stub only this cycle.

    To enable: install phoSTT deps + download Zipformer-vi model (see
    docs/VOICE_DESIGN.md), then wire into get_stt_provider().
    """

    def transcribe(self, audio_bytes: bytes, content_type: str = "") -> str:
        raise NotImplementedError(
            "phoSTT not wired in this cycle (see docs/VOICE_DESIGN.md: "
            "install Zipformer-vi + deps, then enable in get_stt_provider)")


def get_stt_provider():
    """Factory (D2 pin): key -> Whisper, no key -> deterministic mock."""
    if os.getenv("OPENAI_API_KEY"):
        return WhisperAPIProvider()
    return MockSTTProvider()


class PiperProvider:
    """Server TTS default (viPiper, CPU). Lazy import + lazy per-region
    download; unavailable/missing model -> None (fail-open text-only)."""

    def __init__(self):
        self._ok = False
        try:
            import piper  # noqa: F401
            self._ok = True
        except Exception:
            self._ok = False

    @property
    def available(self) -> bool:
        return self._ok

    def speak(self, text: str, voice_region: str = "Bac") -> Optional[bytes]:
        """Same speak() interface; None -> caller fails open to text-only."""
        text = (text or "")[:500]
        if not text.strip() or not self._ok:
            return None
        try:
            voice = _load_piper_voice(voice_region)
            try:
                # Slower for elderly (D3 pin: single VOICE_RATE, length_scale>1 = slower).
                chunks = list(voice.synthesize(text, length_scale=VOICE_RATE))
            except TypeError:
                chunks = list(voice.synthesize(text))
            raw = b"".join(
                getattr(c, "audio_int16_bytes", b"") or b"" for c in chunks)
            if not raw:
                return None
            rate = int(getattr(getattr(voice, "config", None),
                               "sample_rate", 22050) or 22050)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(raw)
            return buf.getvalue()
        except Exception:
            return None


_voices: Dict[str, object] = {}


def _ensure_piper_model(region: str) -> Tuple[Path, Path]:
    """Lazy per-region download (one voice at a time, not all 3)."""
    key = _norm_region(region)
    onnx_url, cfg_url = PIPER_REGION_VOICES[key]
    d = _model_dir()
    d.mkdir(parents=True, exist_ok=True)
    onnx_p, cfg_p = d / f"vi-{key}.onnx", d / f"vi-{key}.onnx.json"
    if not onnx_p.exists() or not cfg_p.exists():
        import urllib.request
        urllib.request.urlretrieve(onnx_url, str(onnx_p))  # noqa: S310
        urllib.request.urlretrieve(cfg_url, str(cfg_p))  # noqa: S310
    return onnx_p, cfg_p


def _load_piper_voice(region: str):
    key = _norm_region(region)
    if key not in _voices:
        onnx_p, cfg_p = _ensure_piper_model(key)
        from piper import PiperVoice  # lazy: no cost when mocked/absent
        _voices[key] = PiperVoice.load(str(onnx_p), str(cfg_p))
    return _voices[key]


class KokoroProvider:
    """Kokoro-VN optional seam (NOT wired to inference this cycle).

    Needs: pip install onnxruntime + kokoro-vietnamese model files
    (kokoro_vi.onnx + voicepack, ~300MB — over demo budget, see module
    docstring). Files missing -> None (fail-open). Inference wire lands
    here only if the user A/B-picks kokoro.
    """

    def speak(self, text: str, voice_region: str = "Bac") -> Optional[bytes]:
        _ = KOKORO_REGION_VOICES.get(_norm_region(voice_region), "diem_trinh")
        return None


def synthesize(text: str, voice_region: str = "Bac") -> Optional[bytes]:
    """Server TTS entry. Seam VOICE_TTS_PROVIDER (piper|kokoro|none,
    default piper); unknown provider -> piper. None -> browser fallback."""
    provider = os.getenv("VOICE_TTS_PROVIDER", "piper").strip().lower()
    if provider in ("none", "off", "browser", ""):
        return None
    if provider == "kokoro":
        return KokoroProvider().speak((text or "")[:500], voice_region)
    return PiperProvider().speak((text or "")[:500], voice_region)
