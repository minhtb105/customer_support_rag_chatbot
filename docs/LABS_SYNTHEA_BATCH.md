"""Lab-report Q&A (Nura) — hieu phieu xet nghiem dai thao duong.

Doc-only: nghien cuu cach Synthea sinh phieu XN dai thao duong theo dot
(batch longitudinal) va anh xa sang Faker seed hien co (khong chay Java).

> Demo-only. Khong phai du lieu benh nhan that. Khong dung cho chan doan.
> Khung: "Synthea-batch-inspired Faker modulation" — hoc phuong phap
> Synthea, implement bang Python/Faker + SQLite thuan.

## 1. Synthea sinh phieu XN dai thao duong theo dot nhu the nao

- **Encounters dinh ky**: moi dot kham (wellness/chronic-care/follow-up)
  sinh 1 encounter; observations chi sinh khi co encounter (khong phai
  daily nhu glucose fingerstick). Do la ly do phieu XN thua hon
  time-series do duong huyet hang ngay.
- **Observations LOINC**: moi chi so trong phieu la 1 observation
  (DATE, PATIENT, ENCOUNTER, CODE-LOINC, VALUE, UNITS). Panel dai thao
  duong gom HbA1c (4548-4), glucose (1558-6), lipid (TG 2571-8 / TC
  2093-3 / LDL 18262-6 / HDL 18263-4), than (creatinine 2160-0 / eGFR
  33914-3), gan (AST 1920-8 / ALT 1742-6).
- **Longitudinal**: lap lai encounters theo lich (vd 3 thang/lan) tao
  trend HbA1c/eGFR theo thoi gian; conditions/meds co START/STOP nen
  phieu sau phan anh dieu tri truoc.
- **Validation kieu JAMIA Table-2**: tune xac suat de phan bo quan the
  khop thuc te, khong ep tung ca the (giong `SYNTHETIC_SYNTHEA_METHOD.md`).

## 2. Anh xa sang Faker seed hien co (`scripts/seed_lab_data.py`)

| Synthea | Ta (`src/labs/` + seed) |
|---|---|
| Encounter dinh ky | 2-4 reports/BN/90 ngay tai moc 30/60/90 (+45 khi can dot thu 4), jitter +-2 ngay |
| Observation LOINC | 10 observations/report (`lab_observations`: report_id, loinc, value, unit, ref_low/high, flag) |
| Longitudinal trend | `sampled_at` cach nhau giup trend HbA1c/FPG/eGFR doc duoc qua `get_history` |
| Conditions/meds START/STOP | Khong schema moi — nam trong notes/sidecar nhu cycle truoc (defer care-engine) |
| Determinism | `random.Random(seed)` + `--reset` scoped `syn_patient_%` + `--dry-run` |

Tables (`src/labs/store.py`, reuse `vitals/base_tracker` pattern):
`lab_reports(report_id, user_id, sampled_at, facility, visit_no)` +
`lab_observations(report_id, loinc, name, value, unit, ref_low, ref_high, flag)`.
DB rieng `metadata/labs.db` (`LAB_DB_PATH` trong `shared/config.py`) —
khong nhoi vao `glucose_logs.db`, khong cham episodic/glucose/B4.

## 3. Chuan tham chieu VN + FLAG QD3312

- Chuan: **BYT QD5481/2020** (thay the QD3319/2017) + muc tieu HbA1c < 7%
  (BYT QD5481 + ADA 2024). Ref ranges trong `src/labs/lab_thresholds.py`
  theo nguong lab pho bien, muc tieu ĐTĐ theo QD5481.
- **FLAG QD3312**: corpus `data/raw/pdfs/byt/` chi co **QD3319 + QD5481 +
  QD3192 (THA)** — KHONG co file QD3312 (grep 3312 = 0 hit ngoai
  WORKFLOW_STATE plan-text). Kha nang user nham QD3312 = QD3319.
  Implementor KHONG che so QD3312; can user xac nhan so QD dung truoc
  khi khoa nguong B2/B3. Panic examples chi dung gia tri IN-PANEL
  (vd HbA1c > 12%, glucose doi > 300 mg/dL, eGFR < 15) — khong Kali/Na+.
