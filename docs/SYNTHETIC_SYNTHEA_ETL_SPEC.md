# Đặc tả ETL Synthea → DiaTrack (doc-only, không implement trong cycle này)

> Trạng thái: **tài liệu kiến trúc**. Repo KHÔNG chứa Java/Gradle/parser/fixture/CSV
> Synthea. Demo/scale hiện tại dùng Faker (`scripts/seed_synthetic_data.py`).
> Tài liệu này trả lời phỏng vấn "tại sao không chạy Synthea thật".

## 1. Trade-off: Synthea thật vs Faker custom

| Tiêu chí | Synthea (MIT, FHIR) + module Diabetes | Faker custom (hiện tại) |
|---|---|---|
| Cài đặt | Java 17 + Gradle, tải ~GB, build chậm | `pip install faker`, nhẹ, CI 0đ |
| Output | Bundle FHIR (Patient/Observation/MedicationRequest) | Thẳng schema `glucose_logs` + 2 sidecar JSON |
| Điều khiển spike | Khó ép đúng 3 archetype benchmark (well/dawn/high-risk) + trend cascade +10–15%/ngày x3 | Công thức Base+Noise+Spikes per-archetype, deterministic `--seed` |
| Mapping | Phải convert FHIR → schema nội bộ, test mapping LOINC | Không mapping, literal hợp lệ sẵn (`fasting/pre_meal/post_meal_2h/bedtime/random`) |
| Thời gian seed | Phút + ETL | 10.800 logs / ~1.3s (`--days 90`, 30 BN) |

Kết luận: giữ Synthea ở mức đặc tả; demo middle-scale (1000+ BN, 30+ BS,
multi-hospital geo-routing) đi bằng Faker mở rộng (`--n-patients/--n-doctors/--hospitals`).

## 2. Pipeline ETL (khi cần chạy thật)

```
Synthea CLI (module diabetes, ~50 BN)
  → patients.csv / observations.csv / medications.csv
  → seed_synthea.py (LỌC + MAP, không commit trong cycle này)
  → glucose_logs + doctors_patients.json + synthetic_roster.json
```

- **Lọc observations:** chỉ giữ LOINC glucose máu (`2339-0`, `2345-7`,
  `1558-6` fasting, `4548-4` HbA1c%). HbA1c% dùng cho Assessment
  (mục tiêu <7% theo BYT QĐ5481 + ADA 2024), không phải điểm time-series.
- **Nội suy time-series (time-bucket):** Synthea cho điểm rời rạc → gom theo
  ngày, mỗi ngày lấy tối đa 1 giá trị/khung (sáng=fasting 06:00,
  trưa=post_meal_2h 13:00, chiều=pre_meal 18:00, tối=bedtime 22:00);
  khung thiếu → linear interpolation giữa 2 ngày kề (tối đa 2 ngày trống,
  quá thì để trống, `detect_trend` coi như đứt chuỗi). Kết quả: 4 logs/ngày,
  `end_offset=3` chừa 3 ngày live demo.
- **ID scheme:** `synthea_<resource-id[:8]>` (prefix khác `syn_patient_` để
  không lẫn benchmark Faker); không tạo auth accounts (queue DISTINCT
  fail-open vẫn hiện).
- **Gán slot bác sĩ:** round-robin PCP trên roster hiện có; slot giữ mock
  `slot_generator` (hashlib busy), không persist appointments (propose-only).

## 3. Medications → previsit_notes (không schema mới)

`MedicationRequest` (metformin/insulin/...) KHÔNG có bảng thuốc riêng.
ETL tóm tắt thành 1 dòng đưa vào `previsit_notes` + dual-write vào `notes`
của log gần nhất kèm trailing-tag `" [AI Triage]"`, ví dụ:

> `Đang dùng metformin 500mg sáng + insulin glargine tối [AI Triage]`

SOAP S tự fill từ `notes`; trang previsit hiện chip `[Nguồn: AI Triage]`
(detect segment-exact trên split `" | "`, không substring). Quy tắc tag:
body truncate ~470 chars trước rồi mới gắn tag ở cuối (tránh truncate-tail
nuốt prefix); FQG followup (`source=None`) không tag nên notes cũ không bao
giờ bị gắn sai.

## 4. Scale rationale (Faker middle-scale)

- Flags: `--n-patients/--n-doctors/--hospitals` (default 30/5/1 byte-identical,
  tests cũ làm gate) + `--days/--skip-b4` (scale demo `--days 30 --skip-b4`
  để 1000 BN ≈ 120k rows trong budget).
- Roster thêm `hospitals` + doctor `hospital_id/lat-lon`; patient sidecar thêm
  `lat/lon` (1/3 gần viện để greedy <5km luôn non-empty).
- `solvers.py`: ưu tiên coords thật (haversine, math stdlib) khi cả 2 phía có
  lat/lon, fallback hash khi thiếu (tương thích ngược — default-30 distances
  không đổi, có assert).
- Queue `GET /v1/doctor/patients`: `limit` (default 100, cap 200) + `offset` +
  `ORDER BY user_id` ổn định; response additive `{patients, total, limit, offset}`.
