"""Learn content validation: schema, units (mg/dL primary, no mmol/L leftovers),
quiz shape, script length (~150 words, elderly plain Vietnamese).
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATA = Path(__file__).resolve().parents[1] / "data" / "curriculum" / "lessons.json"

EXPECTED_COUNTS = {"hieu-benh": 5, "dinh-duong": 10, "van-dong": 5,
                   "thuoc-theo-doi": 5, "bien-chung": 3, "tam-ly": 2}


def _load():
    with open(DATA, encoding="utf-8") as f:
        return json.load(f)["lessons"]


class TestLearnContent:
    def test_count_and_groups(self):
        lessons = _load()
        assert len(lessons) == 30
        counts: dict[str, int] = {}
        for l in lessons:
            counts[l["group"]] = counts.get(l["group"], 0) + 1
        assert counts == EXPECTED_COUNTS

    def test_schema(self):
        for l in _load():
            for k in ("id", "group", "title", "goal", "script_vi",
                      "short", "quiz", "status"):
                assert k in l, f"{l.get('id')} missing {k}"
            assert isinstance(l["short"], bool)
            assert l["status"] == "draft"
            q = l["quiz"]
            assert q["q"] and len(q["options"]) == 3
            assert all(isinstance(o, str) and o.strip() for o in q["options"])
            assert q["correct"] in (0, 1, 2)
            assert q["explanation"] and q["explanation"].strip()

    def test_script_length(self):
        for l in _load():
            n = len(str(l["script_vi"]).split())
            assert 60 <= n <= 180, f'{l["id"]}: {n} words'

    def test_no_mmol_leftovers(self):
        blob = json.dumps(_load(), ensure_ascii=False)
        assert "mmol" not in blob.lower()
        for stray in ("4.0-5.5", "4,0", "<7.8", "<7,8", "13.9", "16.7",
                      "13,9", "16,7"):
            assert stray not in blob, f"stray mmol-era value {stray}"

    def test_mgdl_primary_and_safety_numbers(self):
        by_id = {l["id"]: l for l in _load()}
        assert "mg/dL" in by_id["1.2"]["script_vi"]
        for lid in ("1.2", "5.1"):
            assert "70" in by_id[lid]["script_vi"]
            assert "250" in by_id[lid]["script_vi"]
        assert "115" in by_id["5.1"]["script_vi"]

    def test_no_dose_advice(self):
        # Scripts + explanations must never advise dosing; quiz *distractors*
        # may name wrong behaviors (that is the point of the question).
        # Negation-aware: "không tự tăng liều" (prohibition) is allowed.
        import unicodedata

        def unaccent(s: str) -> str:
            s = unicodedata.normalize("NFD", s)
            s = "".join(c for c in s if unicodedata.category(c) != "Mn")
            return s.replace("đ", "d").replace("Đ", "D")

        for l in _load():
            blob = unaccent((str(l["script_vi"]) + " " +
                             str(l["quiz"]["explanation"])).lower())
            for banned in ("tang lieu", "giam lieu", "uong gap doi",
                           "ngung thuoc", "doi lieu"):
                for m in re.finditer(re.escape(banned), blob):
                    before = blob[max(0, m.start() - 30):m.start()]
                    assert re.search(r"(khong|dung|tuyet|cho|tuyet doi)[\w\s]*$",
                                     before), \
                        f'{l["id"]}: unnegated dose-advice phrase: {banned}'
