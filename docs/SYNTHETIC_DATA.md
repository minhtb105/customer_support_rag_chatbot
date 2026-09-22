# Dữ liệu tổng hợp (Synthetic Data) — Demo đa bệnh nhân / đa bác sĩ

> Demo-only. Không phải dữ liệu bệnh nhân thật. Không dùng cho chẩn đoán.

## Vì sao custom thay vì Synthea?

[Synthea](https://github.com/synthetichealth/synthea) (MIT, chuẩn FHIR, có module
Diabetes) là công cụ chuẩn để sinh quần thể bệnh nhân giả. Nhưng cho demo này,
custom script thắng vì 3 lý do:

| Tiêu chí | Synthea | Custom (`seed_synthetic_data.py`) |
|---|---|---|
| Chạy thử | Cần Java + build Gradle, output FHIR JSON phải convert về schema `glucose_logs` | 1 lệnh Python, ghi thẳng SQLite (<10s) |
| Spike benchmark | Module Diabetes sinh phân bố chung, khó ép đúng 3 archetypes (`well` 5% / `dawn` 40% sáng / `high-risk` 30%) | Can thiệp trực tiếp từng archetype, số nằm đúng band `detect_trend` 10–15% |
| Gắn demo hiện có | Không biết `anomaly_ids`, `end_offset=3`, dual-write notes+episodic | Sinh ra để khớp: literal hợp lệ (không 422), tuần spike có sẵn fact "bánh ngọt" cho SOAP S |

Kết luận: giữ Synthea trong tầm ngắm khi cần quần thể FHIR lớn; demo hiện tại
dùng custom cho nhanh và khớp logic product.

## 3 Archetypes

| Archetype | Fasting 06:00 | Post-meal 13:00 | Spike | Quên thuốc |
|---|---|---|---|---|
| `well_controlled` (10 bn) | 95–110 | 130–160 | 5% (260–300) | 2% |
| `dawn_phenomenon` (10 bn) | 40% sáng >180 (185–230), còn lại 95–115 | 130–160 (bình thường) | tập trung 06:00 | 10% |
| `high_risk_comorbid` (10 bn) | 100–160 (dao động mạnh) | 140–220 | 30% (80% cao 260–300, 20% hạ 55–68), kèm THA | 25% |

Chung: 4 logs/ngày (06:00 `fasting` / 13:00 `post_meal_2h` / 18:00 `pre_meal` /
22:00 `bedtime`), 90 ngày, seed tới T-3 (`--end-offset 3`, chừa 3 ngày live
demo). 5 bác sĩ: Nội tiết ×2, Biến chứng ĐTĐ ×1, Dinh dưỡng ×2; mỗi bác sĩ
quản 6 bệnh nhân (PCP round-robin).

## Lệnh seed / demo

```bash
python scripts/seed_synthetic_data.py --dry-run          # đếm ~10800, không ghi
python scripts/seed_synthetic_data.py --reset --seed 42  # nạp 30 bn + 2 JSON
```

Sidecars: `metadata/doctors_patients.json` (ID → tên đẹp, tuổi, giới tính,
comorbidity, PCP, archetype) + `metadata/synthetic_roster.json` (5 bác sĩ +
giờ làm việc). Helper đọc sidecar (fail-open): `src/features/synthetic_roster.py`
(`get_patient_display_name` / `get_patient_pcp` / `query_doctor_availability`).

Demo: login admin/doctor → `/expert/patients` thấy 30 bệnh nhân tên tiếng Việt,
sort critical → trend → watch → safe → click vào xem sparkline 90d + SOAP.
Chat check-in tuần spike (B4): có `OPENAI_API_KEY` thì GPT-4o-mini diễn đạt,
không có thì template deterministic — CI vẫn xanh.
