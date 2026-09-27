"""Voice subsystem — VinGroup voicebot interim (Whisper STT + browser/Piper TTS).

Interim stack (VOICE_DESIGN.md): OpenAI Whisper API primary (server-side key),
phoSTT stub (local fallback, doc-only), browser speechSynthesis default TTS,
Piper optional same speak() interface. Dialog brain reused (/labs/ask,
/triage, FQG) — voice is only a channel, never a separate write path.
"""
