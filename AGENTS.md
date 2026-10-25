# AGENTS.md — DiaTrack

## Mục tiêu và quy trình

DiaTrack là bản demo trợ lý theo dõi đường huyết: FastAPI, Next.js, RAG tài liệu y tế, tracker và tóm tắt cho bác sĩ. Đọc `README.md` và tài liệu liên quan trong `docs/` trước khi sửa. Repo hiện không có file trạng thái phiên ở gốc.

1. Xác định một phạm vi nhỏ, đầu vào/đầu ra, trường hợp lỗi và tiêu chí hoàn thành. Với chức năng sức khỏe, ghi rõ đường an toàn, nguồn dẫn, ranh giới không chẩn đoán/điều trị; giữ Plan của SOAP dành cho bác sĩ như sản phẩm hiện mô tả.
2. Kiểm tra `git status`, không ghi đè thay đổi có sẵn. Đọc code, test và hợp đồng liên quan thay vì suy ra từ README rằng mọi tính năng đã hoạt động.
3. Chạy test gần phần sửa và bước xác minh bên dưới. Báo lệnh, kết quả, trường hợp không chạy được. Nếu thiếu dịch vụ ngoài, API key hoặc seed, ghi **chưa xác minh đầy đủ**.
4. Bàn giao mục tiêu, file đổi, bằng chứng, rủi ro, việc còn mở và bước tiếp theo. Với việc nhiều phiên hoặc nhiều máy, lưu tóm tắt trong PR hoặc tài liệu được yêu cầu; đừng phụ thuộc vào trí nhớ cuộc trò chuyện.

## Bản đồ repo

- `src/api/main.py`: FastAPI composition root; `src/diabetes/`, `vitals/`, `triage/`, `labs/`, `monitors/`, `auth/`, `chat/`, `admin/`: domain và router; `src/shared/`: config, RAG và tiện ích chung. Đối chiếu `docs/api_contract.md`, `data_contract.md`, `architecture_diagram.md`, `evaluation.md` theo loại việc.
- `frontend/app/`, `components/`, `lib/`: Next.js UI và API client; `frontend/e2e/`: Playwright; `tests/`: pytest; `tests/conftest.py`: fixture và helper auth.
- `scripts/`: crawl, seed, benchmark. Các lệnh crawl/seed có thể ghi dữ liệu; đọc script và đường dẫn trước khi chạy. `data/evaluation/`: gold set và kết quả benchmark.

## Khởi động trên Windows PowerShell

```powershell
$env:PYTHONIOENCODING='utf-8'
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m uvicorn src.api.main:app --reload --port 8000
# Terminal khác, từ repo root:
Set-Location frontend
npm ci
npm run dev
```

Backend ở http://localhost:8000, frontend ở http://localhost:3000. Bỏ bước tạo venv nếu đã có; dùng Python của `.venv` cho mọi lệnh. Chỉ cài `.[local]` khi cần embeddings/reranker local. `.env` và `frontend/.env.local` theo README và `frontend/.env.local.example`; không commit hoặc in key thật.

## Xác minh

| Phạm vi | Bằng chứng cần có |
|---|---|
| Backend/API/DB | Chạy `.\.venv\Scripts\python.exe -m pytest tests/test_<module>.py -q` cho module sửa; chạy thêm `.\.venv\Scripts\python.exe -m pytest -q` khi môi trường cho phép. Test API nên dùng fixture `client` và auth helper trong `tests/conftest.py`. |
| Frontend | Trong `frontend/`: `npx tsc --noEmit`, `npm run build`; chạy `npm run test:e2e -- <spec>` nếu sửa luồng UI. Playwright tự khởi chạy frontend nhưng cần backend và seed phù hợp; không coi video demo là test. |
| RAG/chunking | Chạy test liên quan và benchmark theo `docs/benchmark.md` / `scripts/run_benchmarks.py`; ghi dataset, provider, cấu hình, metric và report. `--skip-judge` bỏ bước judge, không chứng minh chất lượng câu trả lời. |
| Triage/labs/solvers | Chạy test module, benchmark `--services` nếu thay logic được gold set bao phủ; kiểm tra cả tình huống nguy hiểm và từ chối ngoài phạm vi. |

`tests/conftest.py` chuyển các đường DB glucose/auth/vitals trong config sang file tạm. Không mặc định rằng mọi DB, Chroma index hoặc module đã import sẵn đều được cô lập; kiểm tra đường ghi thật trước test/eval. Benchmark có thể gọi API và ghi vào dữ liệu dev: dùng bản sao DB/index hoặc môi trường riêng, xác nhận đường dẫn trước khi chạy, dọn đúng dữ liệu thử nghiệm sau đó. Không chạy benchmark live chỉ để sửa tài liệu.

## Quy ước hoàn thành

Giữ hợp đồng API và các nguồn dẫn y tế; sửa UI thì cập nhật test/selector liên quan. Một tác vụ chỉ đạt khi hành vi theo tiêu chí và các phép kiểm tra phù hợp đã chạy thành công. Ghi rõ mọi test bị bỏ qua và lý do; kết quả test cũ trong README không phải bằng chứng của phiên hiện tại.
