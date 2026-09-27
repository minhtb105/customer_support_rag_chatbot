"""Learn audio guard (F3): lesson scripts (~565-695 chars) exceed /speak [:500].

Chunked playback splits on sentence boundaries and packs to <=480 chars,
so no single /speak call ever cuts mid-sentence. This test mirrors the
frontend splitScript() rule and enforces it on every lesson.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(__file__).resolve().parents[1] / "data" / "curriculum" / "lessons.json"
LIMIT = 480  # under /speak [:500] with margin


def split_script(text: str) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    parts = re.findall(r"[^.!?…]+[.!?…]+", t)
    sents = [s.strip() for s in (parts if parts and len("".join(parts)) >= len(t) - 2 else [t])]
    sents = [s for s in sents if s]
    chunks: list[str] = []
    cur = ""

    def push_long(s: str) -> str:
        words = s.split()
        c = ""
        for w in words:
            cand = (c + " " + w).strip()
            if len(cand) <= LIMIT:
                c = cand
            else:
                if c:
                    chunks.append(c)
                c = w
        return c

    for s in sents:
        cand = (cur + " " + s).strip()
        if len(cand) <= LIMIT:
            cur = cand
            continue
        if cur:
            chunks.append(cur)
        cur = push_long(s) if len(s) > LIMIT else s
    if cur:
        chunks.append(cur)
    return chunks


def _load():
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)["lessons"]


class TestLearnAudioChunks:
    def test_every_chunk_within_speak_limit(self):
        for l in _load():
            for c in split_script(str(l["script_vi"])):
                assert len(c) <= 500, f'{l["id"]}: chunk {len(c)} chars'

    def test_no_mid_sentence_cut_and_lossless(self):
        for l in _load():
            orig = str(l["script_vi"]).strip()
            chunks = split_script(orig)
            assert chunks, f'{l["id"]}: no chunks'
            # Lossless: reassembly covers the full script (whitespace-normalized).
            norm = lambda s: re.sub(r"\s+", " ", s).strip()
            assert norm(" ".join(chunks)) == norm(orig), f'{l["id"]}: lossy split'
            # No mid-sentence cut: every chunk except the last ends with a
            # sentence terminator (long-sentence word-splits are the only
            # exception, and no current lesson needs one).
            for c in chunks[:-1]:
                assert c.rstrip()[-1] in ".!?…", f'{l["id"]}: cut mid-sentence: {c[-40:]}'

    def test_long_lessons_need_chunking(self):
        lessons = _load()
        multi = [l["id"] for l in lessons if len(split_script(str(l["script_vi"]))) > 1]
        assert multi, "expected at least one lesson to need >1 chunk"
        # Single-call playback would truncate: longest script exceeds /speak [:500].
        assert max(len(str(l["script_vi"])) for l in lessons) > 500
