# Kịch bản quay video demo (VI) — 4 phân cảnh + outro (~3,5 phút)

> Nguồn thật duy nhất cho bản quay tay. Bản robot (Playwright
> `frontend/e2e/demo_recorder.spec.ts`) chạy cùng flow để lấy `.webm` nháp.

## Checklist chuẩn bị (đúng thứ tự)

1. Chạy backend: `uvicorn src.api.main:app --port 8000` (Playwright chỉ tự bật FE `:3000`).
2. Tạo tài khoản TRƯỚC: `python scripts/seed_demo_users.py --force`
   (patient `demo_patient_01` / doctor `demo_doctor_01` / admin `demo_admin_01`)
   → lấy patient id (UUID) in ra từ script.
3. Nạp baseline bằng UUID, KHÔNG dùng username:
   `python scripts/seed_glucose_data.py --user <patient-UUID> --reset`
   (89 ngày, kết thúc hôm kia — chừa chỗ cho cascade).

> Nếu DB đã nhiễm từ các lần chạy cũ, dọn 1 lần: `DELETE FROM glucose_logs WHERE user_id IN ('demo_patient_01','<uuid>')` (xóa rows orphan username + cascade UUID tích lũy), rồi nạp lại bằng UUID + `--reset` trước mỗi lần quay.
4. Đẩy cascade qua API với `measured_at` rõ ngày (UI không đặt được ngày;
   3 lần nhập cùng-ngày KHÔNG kích trend):
    `today-4/today-3/today-2`, fasting `100 → 114 → 126` (mỗi ngày lặp 9 lần
   để át baseline, log cuối ghi chú `ăn 2 miếng bánh ngọt tối qua`).
   Lần đẩy cuối phải trả `anomaly.type = trend`.
5. Mở FE `http://localhost:3000`, đăng nhập patient.
5b. Warmup RAG 1 lần (login patient + `POST /v1/query` 1 câu bất kỳ, ~11s lần đầu) để video không treo `...`.
6. Quay → lấy `.webm` trong `frontend/test-results/` (đã gitignore, KHÔNG commit) →
   upload YouTube/Loom → thay link trong README + giữ link tới file này.

> Cảnh báo: mật khẩu demo mặc định `Demo@123` (đổi qua env `DEMO_PASSWORD`
> hoặc `--password`) — chỉ dùng demo local, đổi ngay khi host public.

## PC1 — Cảnh báo đỏ spike (patient `/tracker`, ~30s)

- Thao tác: nhập Lúc đói `280` → Lưu.
- Lời thoại: "Chỉ số 280 vượt ngưỡng 250 — hệ thống khóa chat ngay,
  hiện cảnh báo đỏ yêu cầu liên hệ bác sĩ hoặc cấp cứu. An toàn sinh mạng
  luôn thắng mọi hội thoại."
- Điểm cần thấy: banner đỏ + chữ cứng + chat FQG và nút RAG biến mất.

## PC2 — Trend cascade + FQG + episodic memory (~60s)

- Thao tác: cascade đã đẩy ở bước 4 → nhập thêm 1 chỉ số Lúc đói an toàn
  (vd `140`) → chat hỏi "khẩu phần ăn không?" → trả lời
  "ăn 2 miếng bánh ngọt" → hiện "Đã lưu ngữ cảnh" → chart 14 ngày viền cam.
- Lời thoại: "Ba ngày đói tăng 10–15%/ngày — agent hỏi ngữ cảnh trước khi
  leo thang. Câu trả lời được ghi vào ghi chú log và episodic memory,
  để bác sĩ thấy ở phân cảnh 4."
- Điểm cần thấy: banner cam Trend cascade + chat tự bật + confirm lưu.

## PC3 — Safeguard ngoài phạm vi (~20s)

- Thao tác: trong chat gõ "ho, uống kháng sinh chung với thuốc tiểu đường
  được không?" → Enter.
- Lời thoại: "Câu hỏi ngoài đái tháo đường bị chặn bằng mẫu cứng,
  trước mọi tra cứu guideline — không gọi RAG, không ghi followup."
- Điểm cần thấy: cảnh báo đỏ "Ngoài phạm vi hỗ trợ".

## PC4 — Doctor queue + pre-visit + audit trail (~60s)

- Thao tác: đăng nhập `demo_doctor_01` → `/expert/patients` →
  `demo_patient_01` cờ đỏ lên đầu → mở hồ sơ → chọn 90 ngày → Tạo SOAP →
  S có chữ "bánh ngọt", P trống mờ → click `[Xem log #id]` sang
  `/tracker?highlight=id` → đăng nhập `demo_admin_01` → mở
  `/admin/tracing` trực tiếp → bôi đen dòng `episodic_memory` + chunks QĐ-BYT.
- Lời thoại: "Hàng đợi sắp theo mức khẩn: spike đỏ trước, trend cam sau.
  Hồ sơ 90 ngày + SOAP: S trích đúng lời bệnh nhân, P để trống cho bác sĩ.
  Mọi nhận định đều click về log thô và trace RAG."

## Outro — Slide kiến trúc Production (~30s)

- Chiếu README §7 (Mermaid): Client → API Gateway → vLLM/SGLang
  (continuous batching) → Llama-3-8B-AWQ + PagedAttention/KV Cache.
- Lời thoại: "Demo chạy gpt-4o-mini qua API, không GPU. Production là
  Llama-3-8B-AWQ + vLLM — FUTURE, NOT IN DEMO."
