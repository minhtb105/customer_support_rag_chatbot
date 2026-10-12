"""
Giải mojibake font legacy trong PDF Bảng thành phần VN (FAO mirror).

Quan sát: 1 ký tự lỗi <-> 1 ký tự Việt, nhất quán toàn văn bản
(Vd: G¹o nÕp c¸i = Gạo nếp cái; B¸nh phë = Bánh phở; Ng« b¾p t−¬i = Ngô bắp tươi).
Chiến lược: seed map từ ~40 cặp chắc chắn -> decode -> quét ký tự lạ còn sót
-> bổ sung theo ngữ cảnh, lặp đến khi sạch.
"""
from __future__ import annotations

import json
from pathlib import Path

INTERIM = Path(__file__).resolve().parents[2] / "data" / "raw" / "nutrition" / "_interim"

# garbled -> true. Dựng từ tên món staple chắc chắn, verify nhất quán ở bước sau.
SEED = {
    "¹": "ạ", "Õ": "ế", "¸": "á", "Î": "ẻ", "·": "ã", "ø": "ứ",
    "«": "ô", "¾": "ắ", "μ": "à", "®": "đ", "ë": "ở", "È": "ẩ",
    "ª": "ê", "−": "ư", "¬": "ơ",
    # font map lệch codepoint: glyph ú/ờ rơi vào ô ó/ê
    "ó": "ú", "ê": "ờ",
    # vòng 2 (từ ngữ cảnh): §Ëu=Đậu, ¨n=ăn, phÇn=phần, đưîc=được,
    # nh©n=nhân, thÞt=thịt, Th¶i bá=Thải bỏ
    "§": "Đ", "Ë": "ậ", "¨": "ă", "Ç": "ầ", "î": "ợ",
    "©": "â", "Þ": "ị", "¶": "ả",
    # á U+00E1 genuine trong VNI (thái/đá/cá/Tái) — chỉ 3 từ là ỏ: xem WORD_FIXES
    # vòng 3: d−ìng = dưỡng; vòng 4: thùc = thực; vòng 5: sè = số, Cèm = Cốm
    "ì": "ỡ", "ù": "ự", "è": "ố",
    # vòng 6 (từ tên món): Cñ=Củ, BÝ=Bí, ngät=ngọt, chÝn=chín, nghÖ=nghệ,
    # th×a=thìa, bÑ=bẹ, mÌ=mè, tö=tơ, chïa=chùa, läc=lọc
    "ñ": "ủ", "Ý": "í", "ä": "ọ", "×": "ì", "Ñ": "ẹ", "Ì": "è",
    "Ö": "ệ", "ö": "ử", "ï": "ú",
    # vòng 7: Rau giÒn cơm (Amaranth); vòng 8: Bét=Bột, luéc=luộc
    "Ò": "ề", "é": "ộ",
    # vòng 9: S÷a=Sữa, Củ Êu=Củ ấu
    "÷": "ữ", "Ê": "ấ",
    # vòng 10: B¸nh mú = Bánh mì (French bread); vòng 11: Lßng = Lòng
    "ú": "ì", "ß": "ò",
    # vòng 12: VNI-only (Arial giữ nguyên) — Tôm biÓn, Hång/đång, dõa, (cån, bÐo, vít
    "Ó": "ể", "å": "ồ", "õ": "ừ", "Ð": "é", "í": "ớ",
    # vòng 13: M¾m=Mắm, N−íc=Nước, cã=có (ã U+00E3 -> ó; VNI-only)
    "Æ": "ặ", "ã": "ó",
    # vòng 14: Cïi=Cùi, dõa=dừa, Muçm=Muỗm
    "ï": "ù", "õ": "ừ", "ç": "ỗ",
}

# punctuation thật, không phải mojibake
KEEP = {"–", "“", "”", "•", "—", "…"}

# cặp (garbled, true) dùng để VERIFY map: map phải tái tạo đúng tên thật
VERIFY_PAIRS = [
    ("G¹o nÕp c¸i", "Gạo nếp cái"),
    ("G¹o tÎ gi·", "Gạo tẻ giã"),
    ("G¹o løt", "Gạo lứt"),
    ("Ng« b¾p t−¬i", "Ngô bắp tươi"),
    ("Ng« vμng h¹t kh«", "Ngô vàng hạt khô"),
    ("B¸nh bao", "Bánh bao"),
    ("B¸nh ®a nem", "Bánh đa nem"),
    ("B¸nh ®óc", "Bánh đúc"),
    ("B¸nh phë", "Bánh phở"),
    ("B¸nh quÈy", "Bánh quẩy"),
    ("Kª", "Kê"),
    ("(lo¹i th−êng)", "(loại thường)"),
    ("§Ëu cô ve", "Đậu cô ve"),
    ("phÇn ¨n đưîc", "phần ăn được"),
    ("B¸nh bao nh©n thÞt", "Bánh bao nhân thịt"),
    ("Th¶i bá", "Thải bỏ"),
    ("thùc phẩm", "thực phẩm"),
    ("M· sè", "Mã số"),
    ("Cèm", "Cốm"),
    ("Bét g¹o nÕp", "Bột gạo nếp"),
    ("S÷a đậu nành", "Sữa đậu nành"),
    ("Củ Êu", "Củ ấu"),
    ("B¸nh mú", "Bánh mì"),
    ("Lßng gà", "Lòng gà"),
    ("Tôm biÓn", "Tôm biển"),
    ("Hång", "Hồng"),
    ("Kẹo dõa", "Kẹo dừa"),
    ("Bia (cån", "Bia (cồn"),
    ("M¾m t«m ®Æc", "Mắm tôm đặc"),
    ("N−íc m¾m", "Nước mắm"),
    ("S÷a ®Æc cã ®−êng", "Sữa đặc có đường"),
    ("Cïi dõa", "Cùi dừa"),
    ("Muçm", "Muỗm"),
    ("Cñ sắn", "Củ sắn"),
    ("BÝ ngô", "Bí ngô"),
    ("Nho ngät", "Nho ngọt"),
    ("Xoài chÝn", "Xoài chín"),
    ("Khoai lang nghÖ", "Khoai lang nghệ"),
    ("Dầu mÌ", "Dầu mè"),
    ("Ngô bao tö", "Ngô bao tử"),
]


_TABLE = str.maketrans(SEED)


def is_vni_font(fontname: str) -> bool:
    """Font VNI (VnArial/VnTime...) chứa mojibake; Arial/Symbol giữ nguyên."""
    return "vn" in (fontname or "").lower()


def decode_span(text: str, fontname: str, table: dict = SEED) -> str:
    if is_vni_font(fontname):
        return decode_words(decode(text, table))
    return text


# Cá biệt từng từ (sau char-decode): ký tự VNI đa nghĩa đã verify theo font.
# Chỉ match nguyên token (tránh ăn 'bán', 'thái', 'cá'...).
_WORD_FIXES = [
    ("bá", "bỏ"),  # Thải bỏ
    ("Báng", "Bỏng"),  # Bỏng ngô
    ("thá", "thỏ"),  # Thịt thỏ
    ("rèt", "rốt"),  # Cà rốt (è còn lại -> ố: số/muối/ốc)
]


def decode_words(text: str) -> str:
    import re as _re
    for a, b in _WORD_FIXES:
        text = _re.sub(r"(?<![\wÀ-ỹ])" + _re.escape(a) + r"(?![\wÀ-ỹ])", b, text)
    return text


WORD_FIXES = _WORD_FIXES  # alias tương thích


def decode(text: str, table: dict = SEED) -> str:
    # ponytail: str.translate, 1 pass C; custom table -> maketrans tại chỗ
    return text.translate(_TABLE if table is SEED else str.maketrans(table))


def main() -> None:
    pages = json.loads((INTERIM / "vn_fct_pages.json").read_text(encoding="utf-8"))
    # 1. verify seed nhất quán (char + word fixes)
    bad = []
    for g, t in VERIFY_PAIRS:
        got = decode_words(decode(g, SEED))
        if got != t:
            bad.append((g, t, got))
    if bad:
        print("SEED MÂU THUẪN:")
        for g, t, got in bad:
            print(f"  {g!r} expect {t!r} got {got!r}")
        return
    print(f"seed OK ({len(SEED)} ký tự, {len(VERIFY_PAIRS)} cặp verify)")
    # 2. quét ký tự lạ còn sót sau decode (trừ punctuation thật)
    import re
    strange: dict[str, list[str]] = {}
    for idx, text in pages.items():
        dec = decode(text, SEED)
        for m in set(re.findall(r"[^\x00-\x7F\u00C0-\u1EF9\s]", dec)):
            # ký tự Latin-1 cao còn sót (không phải Việt hợp lệ) + ngữ cảnh
            if ord(m) < 0x80 or "\u00C0" <= m <= "\u1EF9" or m in KEEP:
                continue
            ctx = re.search(r".{0,25}" + re.escape(m) + r".{0,25}", dec)
            strange.setdefault(m, []).append(f"p{idx}: ...{(ctx.group(0) if ctx else '')}...")
    print(f"còn {len(strange)} ký tự lạ:")
    for ch, ctxs in sorted(strange.items()):
        print(f"  U+{ord(ch):04X} {ch!r} x{len(ctxs)} vd: {ctxs[0][:90]}")
        if len(ctxs) > 1:
            print(f"      vd2: {ctxs[1][:90]}")


if __name__ == "__main__":
    main()
