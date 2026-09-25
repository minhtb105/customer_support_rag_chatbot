"""Lab panel ref ranges + panic values (Nura B1).

Panel VN (BYT QD5481/2020 + QD3319/2017 superseded — xem docs/LABS_SYNTHEA_BATCH.md):
HbA1c, glucose doi (FPG), lipid (TG/TC/LDL/HDL), than (creatinine/eGFR), gan (AST/ALT).

Ghi chu D1: corpus chi co QD3319/QD5481/QD3192-THA — KHONG co QD3312
(user viet QD3312, kha nang nham QD3319; khong che so QD3312).
Muc tieu ĐTĐ: HbA1c < 7% (BYT QD5481 + ADA 2024).

Panic examples CHI dung gia tri IN-PANEL (khong Kali/Na+ ngoai panel).
"""
from __future__ import annotations

PANEL: dict = {
    "hba1c": {"loinc": "4548-4", "name": "HbA1c", "unit": "%",
              "ref_low": 4.0, "ref_high": 5.6,
              "panic_low": None, "panic_high": 12.0},
    "fpg": {"loinc": "1558-6", "name": "Glucose doi", "unit": "mg/dL",
            "ref_low": 70.0, "ref_high": 100.0,
            "panic_low": 50.0, "panic_high": 300.0},
    "tg": {"loinc": "2571-8", "name": "Triglyceride", "unit": "mg/dL",
           "ref_low": 0.0, "ref_high": 150.0,
           "panic_low": None, "panic_high": 1000.0},
    "tc": {"loinc": "2093-3", "name": "Cholesterol toan phan", "unit": "mg/dL",
           "ref_low": 0.0, "ref_high": 200.0,
           "panic_low": None, "panic_high": None},
    "ldl": {"loinc": "18262-6", "name": "LDL-C", "unit": "mg/dL",
            "ref_low": 0.0, "ref_high": 100.0,
            "panic_low": None, "panic_high": 250.0},
    "hdl": {"loinc": "18263-4", "name": "HDL-C", "unit": "mg/dL",
            "ref_low": 40.0, "ref_high": 100.0,
            "panic_low": None, "panic_high": None},
    "creatinine": {"loinc": "2160-0", "name": "Creatinine", "unit": "mg/dL",
                   "ref_low": 0.6, "ref_high": 1.2,
                   "panic_low": None, "panic_high": 5.0},
    "egfr": {"loinc": "33914-3", "name": "eGFR", "unit": "mL/min/1.73m2",
             "ref_low": 90.0, "ref_high": 120.0,
             "panic_low": 15.0, "panic_high": None},
    "ast": {"loinc": "1920-8", "name": "AST", "unit": "U/L",
            "ref_low": 10.0, "ref_high": 40.0,
            "panic_low": None, "panic_high": 1000.0},
    "alt": {"loinc": "1742-6", "name": "ALT", "unit": "U/L",
            "ref_low": 7.0, "ref_high": 56.0,
            "panic_low": None, "panic_high": 1000.0},
}

LOINC_TO_KEY = {v["loinc"]: k for k, v in PANEL.items()}


def get_panel() -> dict:
    return PANEL


def flag_value(key_or_loinc: str, value: float) -> str:
    """low | normal | high theo ref range."""
    key = LOINC_TO_KEY.get(key_or_loinc, key_or_loinc)
    spec = PANEL[key]
    if value < spec["ref_low"]:
        return "low"
    if spec["ref_high"] is not None and value > spec["ref_high"]:
        return "high"
    return "normal"


def is_panic(key_or_loinc: str, value: float) -> bool:
    """True khi vuot nguong panic (fail-safe: miss khong chap nhan)."""
    key = LOINC_TO_KEY.get(key_or_loinc, key_or_loinc)
    spec = PANEL[key]
    if spec["panic_low"] is not None and value < spec["panic_low"]:
        return True
    if spec["panic_high"] is not None and value > spec["panic_high"]:
        return True
    return False
