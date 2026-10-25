# AGENTS.md — Diabetes Companion (customer_support_rag_chatbot)

## Task workflow

1. Đọc yêu cầu, `README.md`, phần gần nhất có liên quan trong `WORKFLOW_STATE.md` và tài liệu/hợp đồng trong `docs/`. File trạng thái có nhiều giai đoạn lịch sử; không coi yêu cầu cũ là nhiệm vụ hiện tại.
2. Ghi mục tiêu, phạm vi thay đổi và tiêu chí hoàn thành có thể kiểm tra. Với luồng sức khỏe, nêu rõ trường hợp an toàn, lỗi, nguồn dẫn và ranh giới không chẩn đoán/điều trị. Sửa từng phần nhỏ, giữ nguyên thay đổi đã có trong working tree.
3. Chạy phép kiểm tra liên quan bên dưới; kết thúc bằng bản bàn giao gồm thay đổi, lệnh/kết quả thực tế, kiểm tra chưa chạy, tác động dữ liệu và bước tiếp theo. Không nói “xong” nếu chưa kiểm tra được hành vi chính.

## Quick Start (Windows PowerShell 5.1)

```powershell
$env:PYTHONIOENCODING='utf-8'
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"  # backend + pytest
# Chỉ khi cần embedding local: .\.venv\Scripts\python.exe -m pip install -e ".[local,test]"
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --port 8000
# Terminal khác, từ repo root:
Set-Location frontend
npm ci
npm run dev
```

Backend ở http://localhost:8000, frontend ở http://localhost:3000. Dùng `.venv\Scripts\python.exe` của repo cho mọi lệnh Python; không dùng Python hệ thống. Trong `frontend/` dùng `npm ci` để theo lockfile. Nếu venv đã tồn tại, bỏ bước tạo mới.

Env (gitignored, never commit): `.env` chạy NVIDIA (`LLM_PROVIDER=nvidia`, `EMBEDDING_PROVIDER=nvidia`, keys `NVDIA_API_KEY/2/3` — giữ đúng chính tả này); `frontend/.env.local` needs `NEXT_PUBLIC_API_URL=http://localhost:8000`.

## Architecture

- `src/api/main.py` — FastAPI composition root (routers mounted here).
- Domain packages: `src/{chat,diabetes,labs,learn,monitors,reviews,scheduling,triage,vitals,voice,auth,admin}/` — each owns `router.py` + store/tracker logic.
- `src/shared/` — shared kernel. Reuse before writing: `text.py:unaccent`, `memory/_extract.py` (single entity extractor), `prompt_templates.build_strict()`, `monitors` use `run_job()`/`stage_version()`.
- `scripts/` — one-shot tooling. Shared helpers: `nutrition/_fetch.py` (download), `nutrition/_parse_io.py`, `nutrition/_ingest.py --src md|json`, `reembed.py --provider local|nvidia`. Keep CLIs working when editing.
- `frontend/app/` — Next.js routes (`tracker`, `previsit`, `monitor/*`, `admin/*`, `expert/*`); shared UI in `frontend/components/` (`AdminTable`, `GlucoseChart`, `MonitorList`); API wrappers in `frontend/lib/api.ts` via `apiGet`/`apiPost`.
- `tests/` — pytest; shared fixtures in `tests/_helpers.py` (extend it, don't copy-paste cases).
- Chunking: `src/shared/chunk_strategies.py::VARIANTS` (registry `tên → (base, overrides)`; unknown name thì raise — không đặt tên tự do). Serving default single-source `src/shared/config.py::DEFAULT_CHUNK_STRATEGY` (hiện `hybrid_section_semantic`, winner B1 2026-10-12). Index tách theo provider: `embeddings/pdf_db_nvidia/<strategy>/` (2048d).

## Test-DB isolation

- `tests/conftest.py:client` (session fixture) redirects all 5 DBs (glucose/auth/vitals/monitoring/tracing) to temp files — tests never touch dev DBs. Always write API tests against this `client` fixture.
- Why module-namespace patching: trackers do `from config import X_DB_PATH` at import time, so setting `cfg.X_DB_PATH` alone is a no-op. Patch the tracker's own namespace (`bp_tracker.VITALS_DB_PATH`, `glucose_tracker.GLUCOSE_DB_PATH`, …) **before** importing the app — see `conftest.py` lines 39-60.
- Auth helpers: `make_user`, `auth_header`, `doctor_user`, `admin_client` — reuse them instead of hand-rolling register/login.
- Re-init pattern: after redirecting a DB path, call the tracker's `init_*_db()` (e.g. `init_bp_db()`, `init_glucose_db()`).

## Benchmark workflow

```bash
python scripts/run_benchmarks.py [--skip-judge] [--latency-n 10]   # B1-B5 RAG over data/evaluation/
BENCH_STRATEGIES=<slate> BENCH_WINNER=<name> DEFAULT_CHUNK_STRATEGY=<name> python scripts/run_benchmarks.py
```
- B1 chunking (Recall@5/Hit@5/MRR) → B2 retriever (bm25/vector/hybrid) → B3 rerank → B4 latency P50/P95 → B5 LLM-judge (faithfulness/precision/recall/fluency 0-5). Results land in `data/evaluation/results_benchmark_<ts>.json`.
- Winner = max Recall@5 (tie-break MRR → Hit@5), hàm thuần `pick_winner` trong script. Slate đầy đủ = keys của `VARIANTS` (12 strategies; winner đã lock 2026-10-12: `hybrid_section_semantic` R 0.8875/Hit 1.0/MRR 0.9083).
- B4/B5 E2E có thể ghi facts của `default_user` vào memory Chroma và traces `fresh benchmark question...`. Chỉ chạy trên bản sao DB/index hoặc môi trường benchmark cô lập; ghi lại đường dẫn và kiểm tra nó không trỏ vào dữ liệu dev/prod trước khi chạy. Dọn dữ liệu benchmark sau run, không xóa mù dữ liệu dùng chung. Chỉ commit results JSON bằng `git add -f` khi tác vụ yêu cầu lưu kết quả (`data/*` đang gitignored).
- B6-B8 (triage/labs/solvers, pure functions, 0 LLM calls) are opt-in via `--services`; gold sets live in `data/evaluation/`. B5 judge follows `LLM_PROVIDER` (`openai` default; `nvidia` uses `NVIDIA_JUDGE_MODEL`); `--skip-judge` for offline runs (`HF_HUB_OFFLINE=1` is set by the script).

## Verification gates

| Phạm vi sửa | Bằng chứng cần có |
|---|---|
| Backend/API/DB | Chạy test trực tiếp cho module sửa bằng `.\.venv\Scripts\python.exe -m pytest tests/test_<module>.py -q`, rồi `.\.venv\Scripts\python.exe -m pytest -q` khi môi trường và phạm vi cho phép. Với API, dùng fixture `client`; không kiểm thử trên DB dev. |
| Frontend/UI | Trong `frontend/` chạy `npx tsc --noEmit` và `npm run build`. Nếu sửa tương tác hoặc `data-testid`, cập nhật spec liên quan và chạy `npm run test:e2e -- <spec>` với backend test/seed cô lập; `demo_recorder` vẫn manual-only. |
| RAG, chunking, prompt | Chạy test liên quan, rồi benchmark trên golden set phù hợp. B1-B3 có thể đánh giá riêng; B4/B5 cần kiểm tra cô lập dữ liệu và provider/chi phí trước khi chạy. Ghi metric, cấu hình strategy/provider, dataset và file report để so sánh. |
| Triage, labs, solvers | Chạy test unit của module và benchmark `--services` khi có gold set liên quan. Kiểm tra cả đường lỗi và hành vi an toàn, không chỉ điểm trung bình. |

Test E2E cần backend đang chạy và dữ liệu seed; Playwright config tự khởi chạy frontend, không tự khởi chạy backend. Nếu thiếu API key, dịch vụ, seed hay môi trường cô lập, ghi **chưa xác minh đầy đủ** cùng lệnh/điều kiện để chạy lại. Một kết quả test cũ trong README/AGENTS.md không thay thế kết quả của phiên hiện tại.

## Session handoff

Đọc `WORKFLOW_STATE.md` khi bắt đầu nếu công việc tiếp nối; xác nhận yêu cầu mới nhất trước khi thực hiện. Trước khi dừng, ghi ngắn gọn mục tiêu, trạng thái, quyết định, bằng chứng kiểm thử, file thay đổi, việc chưa xong và bước tiếp theo vào handoff của phiên. File này là scratch và không commit: nếu cần bàn giao qua máy/clone khác, đưa tóm tắt vào PR hoặc tài liệu được phép lưu, tránh phụ thuộc duy nhất vào file local.

## Conventions

- Refactor with aliases: keep old public names (functions, classes, routes, schemas) importable as thin wrappers. Deletion over addition otherwise.
- Conventional commits: `feat:`, `fix:`, `chore:`, `refactor:` (one-line subject).
- `WORKFLOW_STATE.md` là scratch local — không commit; dùng handoff trong PR khi cần lưu xuyên clone.
- No new deps for what stdlib/httpx/already-installed covers. `requirements.txt` mirrors `pyproject.toml` base.

## Gotchas

- Shell is Windows PowerShell 5.1: no `&&` (use `; if ($?) { ... }`), no `head`/`tail` (use `Select-Object -First/-Last`), quote paths with spaces.
- Imports use dual `try: from src.… except: from …` fallbacks — preserve them; keep `PYTHONPATH` including `src`.
- `scripts/reembed_local.py`, `reembed_nvidia.py`, `seed_admin.py` are thin shims — edit the real targets (`reembed.py`, `seed_demo_users.py --force`).
- e2e (`frontend/e2e/`) needs a live backend + seeds; `demo_recorder` is manual-only, not CI.
- Secrets: no real keys on disk or in prompts. `.env` files stay gitignored; README shows only an `sk-proj-...` placeholder.
- PowerShell phá quoting `python -c` nhiều dòng: viết script vào `C:\Users\DELL\AppData\Local\Temp\opencode\*.py` rồi chạy (pre-approved cho thư mục temp này), không inline.
- Frontend: cấm emoji làm icon (dùng `lucide-react` có sẵn); giữ nguyên mọi `data-testid`; đổi copy UI phải update `frontend/e2e/*.spec.ts` trong cùng diff.
- Backend suite hiện ~444 tests (~10 phút). Lỗi mạng thoáng qua tới NVIDIA API (ConnectError/10054) đã có retry xoay keys trong `src/shared/nvidia.py` — gặp thì chạy lại là được, không sửa code.
