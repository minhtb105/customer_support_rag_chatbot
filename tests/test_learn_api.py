"""Learn API: onboarding/profile, progress, quiz scoring, adaptive next,
server-driven today (VN wall-clock, missed-day no-skip), map_quiz_answer,
auth isolation. No-key deterministic (pure logic, mock TTS — no audio).
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

VN = timezone(timedelta(hours=7))


@pytest.fixture
def iso_learn(monkeypatch, tmp_path):
    import src.learn.store as st
    monkeypatch.setattr(st, "LEARN_DB_PATH", tmp_path / "learn.db")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return tmp_path


def _auth(client, make_user):
    user, tok = make_user(role="user")
    return user, {"Authorization": f"Bearer {tok}"}


class TestLearnApi:
    def test_401_anon(self, client, iso_learn):
        client.cookies.clear()
        assert client.get("/v1/lessons").status_code == 401

    def test_onboarding_and_list(self, client, make_user, iso_learn):
        user, h = _auth(client, make_user)
        r = client.post("/v1/lessons/onboarding?user_id=" + user["id"],
                        json={"age": 65, "meds": "thuoc tieu duong",
                              "has_meter": 1}, headers=h)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["day1_date"] == datetime.now(VN).date().isoformat()
        r = client.get("/v1/lessons?user_id=" + user["id"], headers=h)
        assert r.status_code == 200
        assert len(r.json()["lessons"]) == 30

    def test_403_cross_user(self, client, make_user, iso_learn):
        victim, _ = make_user(role="user")
        _, h2 = _auth(client, make_user)
        r = client.get("/v1/lessons?user_id=" + victim["id"], headers=h2)
        assert r.status_code == 403

    def test_quiz_scoring_and_mastery(self, client, make_user, iso_learn):
        user, h = _auth(client, make_user)
        client.post("/v1/lessons/onboarding?user_id=" + user["id"],
                    json={}, headers=h)
        import src.learn.store as st
        r = client.post("/v1/lessons/1.1/quiz?user_id=" + user["id"],
                        json={"answer": 1}, headers=h)
        assert r.status_code == 200, r.text
        assert r.json()["correct"] is True
        prof = st.get_profile(user["id"])
        assert prof["mastery"].get("hieu-benh", 0) > 0
        r = client.post("/v1/lessons/1.2/quiz?user_id=" + user["id"],
                        json={"answer": 2}, headers=h)
        assert r.json()["correct"] is False  # 1.2 correct is 0

    def test_quiz_answer_text_mapping(self, client, make_user, iso_learn):
        user, h = _auth(client, make_user)
        client.post("/v1/lessons/onboarding?user_id=" + user["id"],
                    json={}, headers=h)
        r = client.post("/v1/lessons/1.1/quiz?user_id=" + user["id"],
                        json={"answer_text": "đáp án B"}, headers=h)
        assert r.json()["correct"] is True
        r = client.post("/v1/lessons/1.1/quiz?user_id=" + user["id"],
                        json={"answer_text": "một"}, headers=h)
        assert r.json()["mapped"] is None  # no-guess -> buttons

    def test_next_coldstart_and_adaptive(self, client, make_user, iso_learn):
        user, h = _auth(client, make_user)
        client.post("/v1/lessons/onboarding?user_id=" + user["id"],
                    json={}, headers=h)
        r = client.get("/v1/lessons/next?user_id=" + user["id"], headers=h)
        assert r.json()["lesson"]["id"] == "1.1"
        # Finish all hieu-benh -> weakest group shifts elsewhere.
        import src.learn.store as st
        by_id = {l["id"]: l for l in
                 client.get("/v1/lessons?user_id=" + user["id"],
                            headers=h).json()["lessons"]}
        for lid, l in by_id.items():
            if l["group"] == "hieu-benh":
                st.record_quiz(user["id"], lid, True)
        r = client.get("/v1/lessons/next?user_id=" + user["id"], headers=h)
        assert r.json()["lesson"]["group"] != "hieu-benh"

    def test_engagement_low_prefers_short(self, client, make_user, iso_learn):
        user, h = _auth(client, make_user)
        client.post("/v1/lessons/onboarding?user_id=" + user["id"],
                    json={}, headers=h)
        import src.learn.store as st
        st.upsert_profile(user["id"], {"engagement": "low"})
        r = client.get("/v1/lessons/next?user_id=" + user["id"], headers=h)
        assert r.json()["lesson"]["id"] == "1.1"  # cold-start wins first
        st.record_quiz(user["id"], "1.1", False)
        st.record_quiz(user["id"], "1.2", False)  # hieu-benh 0.0, ties others
        # tie-break smallest group -> hieu-benh; engagement low -> short:true
        r = client.get("/v1/lessons/next?user_id=" + user["id"], headers=h)
        nxt = r.json()["lesson"]
        assert nxt["group"] == "hieu-benh" and nxt["short"] is True
        assert nxt["id"] == "1.4"  # first short:true in weakest group

    def test_today_server_driven_and_resame_day(self, client, make_user,
                                               iso_learn):
        user, h = _auth(client, make_user)
        client.post("/v1/lessons/onboarding?user_id=" + user["id"],
                    json={}, headers=h)
        r1 = client.get("/v1/lessons/today?user_id=" + user["id"], headers=h)
        r2 = client.get("/v1/lessons/today?user_id=" + user["id"], headers=h)
        assert r1.json()["day"] == 1
        assert r1.json()["lesson"]["id"] == r2.json()["lesson"]["id"] == "1.1"

    def test_today_missed_day_no_skip(self, client, make_user, iso_learn):
        user, h = _auth(client, make_user)
        client.post("/v1/lessons/onboarding?user_id=" + user["id"],
                    json={}, headers=h)
        import src.learn.store as st
        three_ago = (datetime.now(VN).date() - timedelta(days=3)).isoformat()
        st.upsert_profile(user["id"], {"day1_date": three_ago})
        r = client.get("/v1/lessons/today?user_id=" + user["id"], headers=h)
        assert r.json()["day"] == 4  # current day kept, no skipping
        # 1.1 done -> earliest unfinished in sequence (1.2), not day-4 lesson.
        st.record_quiz(user["id"], "1.1", True)
        r = client.get("/v1/lessons/today?user_id=" + user["id"], headers=h)
        assert r.json()["lesson"]["id"] == "1.2"


class TestMapQuizAnswer:
    def test_letters_only(self):
        from src.learn.recommend import map_quiz_answer as m
        assert m("đáp án A") == "A"
        assert m("cau b") == "B"
        assert m("chọn C") == "C"
        assert m("B") == "B"

    def test_no_guess(self):
        from src.learn.recommend import map_quiz_answer as m
        assert m("một") is None
        assert m("1") is None
        assert m("") is None
        assert m("tôi không biết") is None
