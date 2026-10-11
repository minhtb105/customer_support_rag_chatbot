# UI Review — Multi-Role Audit (desktop, light mode)

- Date: 2026-10-07. Scope: full function walkthrough as patient, doctor, admin (pharmacist excluded — admin covers the only pharmacist-gated screen, `/monitor/alerts`).
- Accounts (from `scripts/seed_demo_users.py --force`, password `Demo@123` demo-only): `demo_patient_01` (user), `demo_doctor_01` (doctor, verified), `demo_admin_01` (admin, verified).
- Data: glucose baseline 203 logs (89 days) + 27-log trend cascade (100→114→126 fasting, 3 consecutive days) seeded under the patient auth UUID; logins verified 200 for all 3 roles.
- Method: real browsers flows (login form → submit readings → FQG chat → safeguard question → SOAP generate → quiz → queue click-through → tracing), one viewport screenshot per screen in `docs/demo/ui-review/` (1440×900, light).
- This cycle is **report-only**: no UI code was changed for any finding below.

## Severity scale

- **Critical** — flow completely broken (crash, dead end, data loss, safety-relevant misdirection).
- **Major** — works but misleads, erodes trust, or defeats the feature's purpose.
- **Minor** — polish, copy, dev-text leakage, demo hygiene.

## Summary (3 Critical, 7 Major, 12 Minor)

| ID | Sev | Location (screenshot) | Issue | Proposal |
|----|-----|------------------------|-------|----------|
| C1 | Critical | Patient → lesson detail (`patient-08-lesson.png`) | `/hoc/[id]` crashes: `use(params)` is a Next-15 pattern; on Next 14.2 `params` is a plain object, so `use()` throws. Both client overlay and SSR 500 (`GET /hoc/1.1 500`). Lesson detail is 100% unusable; `tsc` also reports a `.next/types` error for this file. Quiz + voice-answer + 60s audio unreachable. | Read `params.id` directly (same as `expert/patients/[id]`, which works). Add one browser smoke test for lesson detail. |
| C2 | Critical | Doctor → Duyệt queue (`doctor-04-queue.png`) | `/expert/queue` crashes on real data: `JSON.parse(r.failed_metrics)` throws (`"faithfulness"... is not valid JSON`) — backend sends a plain string/list, not JSON. The entire HILT review queue is broken exactly when reviews exist (58+ pending). | Parse defensively (try/catch + handle string/array) or align API contract to always send an array. |
| C3 | Critical | Doctor clicks `[Xem log #id]` (`doctor-03-loglink.png`) | Cross-role audit trail is a dead end: the link lands on `/tracker?highlight=id`, which is strictly self-view, so the doctor sees their **own empty tracker** (0 logs, "Chưa có dữ liệu") instead of the patient's log. The flagship click-through story is false for doctors; worse, the form invites doctors to submit glucose data as themselves. | Render patient context on that route for experts (read-only patient logs + highlight), or route to `expert/patients/[id]` anchored at the log. Gate the entry form to patient role. |
| M1 | Major | Patient chat, history, memory (`patient-05-tracker-safeguard.png`, `patient-13-reviews.png`, `admin-04-memory.png`) | Assistant answers render in **English** inside a Vietnamese-only elderly-patient product ("Eating 2 pieces of sweet bread…", "I'm not sure based on the provided information…"), although the diabetes prompt mandates Vietnamese. | Force `language: vi` (or a VI-only FQG reply template) on the patient chat path; add a language assertion to tests. |
| M2 | Major | Header for expert/admin (`doctor-01-queue.png`, `admin-01-tracing.png`) | Nav wraps mid-word at 1440px ("Tổng/quan", "Nhật ký/ĐH", "WHO-/RAG API", "Đăng/xuất") once role items accumulate — unreadable for the primary expert user. | Collapse role sections into a dropdown / hamburger below `lg`, shorten labels ("Nhật ký", "Duyệt (58)"). |
| M3 | Major | Doctor/patient previsit header (`doctor-02-detail.png`, `patient-06-previsit.png`) | Patient shown as raw UUID prefix ("Hồ sơ bệnh nhân — 500ddbf9", "Patient: 500ddbf9") — a doctor cannot tell who this is. | Join username (as the queue endpoint already does) and show it everywhere a patient appears. |
| M4 | Major | Objective card (`doctor-02-detail.png`) | Raw Python dict repr leaks to doctors (`Phân bố: {'high': 11, 'normal': 97, 'elevated': 124}`); the frequency figure does not reconcile (57 logs / 14 days shown as "trung bình 21.5/tuần" = 28.5 expected). | Render a human-readable breakdown; verify the per-week formula against the same window the count uses. |
| M5 | Major | Playground + reviews + queue badge (`patient-12-playground.png`, `patient-13-reviews.png`) | Every sampled query fails **all 4** HILT metrics → 58–65 perpetually pending reviews, 15 identical spam notifications, and "low confidence" banners on good grounded answers. Thresholds look miscalibrated, eroding user trust in the safety gate itself. | Re-measure thresholds on golden queries; group/dedupe notifications; show per-metric bars before routing everything to doctors. |
| M6 | Major | Memory table (`admin-04-memory.png`) | All facts `type=general`, entities empty, confidence/weight identical (0.8/0.878), and repeats are stored verbatim (5 identical "bánh ngọt" rows, no dedup). Classification/entities extraction appears non-functional. | Fix classifier/entity extractor (or drop the columns), add content-hash dedup on write. |
| M7 | Major | S — Subjective card (`patient-06-previsit.png`, `doctor-02-detail.png`) | An editable textarea duplicates the S text with no visible save action — edits go nowhere (dead edit) and the duplication confuses. | Either wire save explicitly (button + confirmation) or render read-only. |
| m1 | Minor | All Recharts (`patient-02-tracker.png`, `patient-06-previsit.png`) | X-axis tick labels overlap into an unreadable strip on 14d/90d charts. | `interval="preserveStartEnd"` (as sparkline does) + fewer ticks / angled labels on the 14d chart too. |
| m2 | Minor | Landing, playground, tracing/chunks/memory, reviews | Raw backend URL (`http://127.0.0.1:8000`) printed to end users in chips and subtitles. | Show relative labels ("POST /v1/query") and hide the host outside admin views. |
| m3 | Minor | Reviews, users, guidelines, prompts | Dev-facing copy leaks to end users ("Push notification polling 7s", "python scripts/seed_admin.py", "Xem docs/monitoring.md", raw version/hash rows, "Kiểm tra ngay (HEAD)"). | Move to tooltips/admin-only or plain language. |
| m4 | Minor | Tracker entry (`patient-02-tracker.png`) | Value input ships prefilled with magic default `110` — one accidental tap logs a false reading. | Start empty with placeholder; keep 110 only as a hint. |
| m5 | Minor | Reviews, previsit | Raw/mixed timestamps ("2026-10-07T03:21", "10:20:38 7/10/2026"). | Single `vi-VN` format everywhere. |
| m6 | Minor | Trace detail (`admin-02-detail.png`) | Only "Đang tải trace..." after 2.5s — no data, no error, unknown whether slow or failed. | Skeleton + timeout + retry/error message. |
| m7 | Minor | Chunks (`admin-03-chunks.png`) | Source/File columns truncate to "who_classification_diabetes_..." (which document is unknowable); Trang/Cập nhật are "—" for every row. | Tooltip/full column + index page metadata so citations carry page numbers. |
| m8 | Minor | Alerts tab (`doctor-06-alerts.png`) | "Cảnh báo thuốc" tab is offered to doctors but always dead-ends at the role message (RBAC itself is correct). | Hide the tab for roles without access. |
| m9 | Minor | Users (`admin-06-users.png`) | Verified ✅ for every row (incl. plain users), Active is a cryptic "•", only action is Unverify. | Text statuses + verify/deactivate actions. |
| m10 | Minor | Queue, guidelines (`doctor-01-queue.png`, `doctor-05-guidelines.png`) | Test junk pollutes demo views (`u_d86b93` 350-critical above the demo patient; `[gold] Superseded Test` fixture in the monitor list). | Wipe `test_*`/gold scopes before demos or hide them behind a demo filter. |
| m11 | Minor | Sparkline 90d (`patient-06-previsit.png`) | Sparse/flat rendering; anomaly dots hard to see at 90d scale. | Larger dots + y-zoom to data range. |
| m12 | Minor | Previsit bottom (`patient-06-previsit.png`) | Raw JSON dump block shown to patients. | Move behind a `<details>` or admin-only. |

## What works (verified live, keep as-is)

- Login form for all 3 roles; httpOnly cookie session; correct RBAC denials (doctor→alerts message).
- Spike red banner with exact safety text + chat/RAG lock; unlock on safe log.
- Trend cascade detection + proactive FQG + episodic save ("Đã lưu ngữ cảnh."); safeguard canned refusal, no follow-up call.
- Queue sort critical → trend → watch; trend badge on demo patient.
- SOAP: S notes + `[Xem log #id]` links, O numbers, A HbA1c<7% BYT5481/ADA, P empty + "Dành cho bác sĩ chỉ định".
- Persona picker (onboarding + settings reuse) — excellent for elderly users.
- Tracing table search/filter/pagination; memory user filter + table; prompts draft/dry-run layout; triage severity highlighting (KHẨN CẤP red).
- Review pipeline end-to-end (query → pending → notification → history).

## Environment notes (for reproducibility)

- Backend ran natively (`.\.venv\Scripts\python -m uvicorn ... :8000`); WSL was tried first per request but its system Python 3.12 has a corrupt `_ctypes` (stdlib reinstall unavailable), so the venv route failed. Fallback per approved plan.
- Frontend ran natively (`npm run dev :3000`); WSL serves 200 inside WSL but Windows↔WSL TCP for :3000 was blocked on this host.
- Two env-only adjustments (both untracked/ignored, not product changes): `frontend/.env.local` `localhost`→`127.0.0.1` (browser resolved `localhost` to ::1 while uvicorn binds IPv4); BE started with `FRONTEND_URL` including both origins (default CORS allowlist covers only `localhost:3000`, browsers on `127.0.0.1:3000` got "Failed to fetch"). Revert/revisit when demo host changes.
- `.venv` needed `pip install -r requirements.txt` (was missing `python-jose`, which silently disabled auth/admin/chat/diabetes/triage/labs/voice/learn/vitals/monitors routers via fail-open import). No repo files changed.
- Screenshots are honest captures of one audit pass; repeated runs accumulate demo logs (trend window holds ~10 consecutive runs before the limit-30 cascade math drifts — wipe + reseed per the iron rule before any re-shoot).

## Out of scope (not covered)

- Pharmacist role (admin covers `/monitor/alerts`); dark mode; mobile 375px; loan/streaming paths beyond one playground query; lesson audio playback (blocked by C1); review approve/decision click-through (blocked by C2).
