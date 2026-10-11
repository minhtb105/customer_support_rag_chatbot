"""
Ingest foods JSON (data/raw/nutrition/foods/*.json) vào Chroma `structure`.

- 1 món = 1 chunk tự chứa: tên VI+EN + số + khuyến nghị (BM25 bắt từ khóa Việt).
- metadata: source_id=food_id, dataset=nutrition, section_path=[category, tên],
  file_name=<file>.json, chunking_strategy=structure — đúng schema indexer.
- idempotent theo chunk_hash trong metadata_store (file_key nutrition_foods).
Chạy:  python scripts/nutrition/ingest_food_json.py
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]
FOODS_DIR = ROOT / "data" / "raw" / "nutrition" / "foods"

try:
    from src.shared.indexer import (
        get_or_create_vectorstore, compute_chunk_hash, _resolve_index_db_dir)
    from src.shared.chunk_strategies import serialize_metatdata
    from src.shared.models.chunk import Chunk, ChunkMetadata
    from src.shared.metadata_store import (
        init_db, upsert_file_and_chunks, get_chunk_hashes_for_file,
        find_vector_ids_for_chunk_hashes, delete_chunks_by_hashes)
    from src.shared.config import META_DB_PATH, EMBEDDING_MODEL
except ImportError:
    from shared.indexer import (  # type: ignore
        get_or_create_vectorstore, compute_chunk_hash, _resolve_index_db_dir)
    from shared.chunk_strategies import serialize_metatdata  # type: ignore
    from shared.models.chunk import Chunk, ChunkMetadata  # type: ignore
    from shared.metadata_store import (  # type: ignore
        init_db, upsert_file_and_chunks, get_chunk_hashes_for_file,
        find_vector_ids_for_chunk_hashes, delete_chunks_by_hashes)
    from shared.config import META_DB_PATH, EMBEDDING_MODEL  # type: ignore

FILE_KEY = "nutrition_foods::structure"
GI_VN = {"Low": "thấp", "Medium": "trung bình", "High": "cao", "Unknown": "chưa có"}


def gi_cat(gi) -> str:
    if gi is None:
        return "Unknown"
    if gi <= 55:
        return "Low"
    return "Medium" if gi <= 69 else "High"


def render(food: dict) -> str:
    n, s, g, r = (food["nutrition_per_100g"], food["serving"],
                  food["glycemic"], food["diabetes_recommendation"])
    names = food["names"]["vi"] + f" ({food['names']['en']})"
    aka = food["names"].get("aliases_vi") or []
    lines = [
        f"{names} — nhóm {food['category']}.",
    ]
    if aka:
        lines.append(f"Tên gọi khác: {', '.join(aka)}.")
    lines.extend([
        f"Khẩu phần chuẩn: {s['size_g']:g} g ({s['description']}).",
        f"Dinh dưỡng mỗi 100 g: {n['energy_kcal']:g} kcal; "
        f"carbs {n['total_carbohydrate_g']:g} g (đường {n['sugars_g']:g} g, "
        f"xơ {n['dietary_fiber_g']:g} g); đạm {n['protein_g']:g} g; "
        f"béo {n['total_fat_g']:g} g.",
    ])
    if g["gi"] is None:
        lines.append("Chưa có chỉ số GI cho món này (không đường bột hoặc chưa đo).")
    else:
        gl_std_txt = f"{r['gl_std']}" if r["gl_std"] is not None else "không tính"
        lines.append(
            f"GI {g['gi']:g} ({GI_VN[gi_cat(g['gi'])]}, nguồn {g['gi_tier']}). "
            f"GL theo khẩu phần chuẩn: {gl_std_txt}.")
    lines.append(
        f"Khuyến nghị tiểu đường: nhẹ/kiểm soát tốt {r['mild_g']:g} g "
        f"(~{r['choices_mild']:g} carb-choice"
        f"{', GL ' + str(r['gl_mild']) if r['gl_mild'] is not None else ''}); "
        f"nặng/kiểm soát kém {r['severe_g']:g} g "
        f"(~{r['choices_severe']:g} carb-choice"
        f"{', GL ' + str(r['gl_severe']) if r['gl_severe'] is not None else ''}).")
    lines.append(f"Lưu ý: {r['notes']}")
    return "\n".join(lines)


def main() -> None:
    import sys as _sys
    rebuild = "--rebuild" in _sys.argv
    init_db(META_DB_PATH)
    vector_db = get_or_create_vectorstore(_resolve_index_db_dir("structure"))
    if rebuild:
        # Xóa chunk foods cũ (text đổi -> hash đổi) rồi nạp lại từ đầu
        old = get_chunk_hashes_for_file(FILE_KEY) or []
        vids = find_vector_ids_for_chunk_hashes(old)
        if vids:
            try:
                vector_db._collection.delete(ids=vids)
            except Exception as e:  # noqa: BLE001
                log.warning("delete cũ: %s", str(e)[:150])
        delete_chunks_by_hashes(old)
        log.info("rebuild: đã xóa %d chunks cũ", len(old))
    files = sorted(FOODS_DIR.glob("*.json"))
    if not files:
        log.warning("Chưa có foods JSON (chạy build_foods.py trước)")
        return
    old_hashes = set(get_chunk_hashes_for_file(FILE_KEY) or [])
    texts, metas, ids, rows = [], [], [], []
    for fp in files:
        food = json.loads(fp.read_text(encoding="utf-8"))
        chunk = Chunk(source_id=food["food_id"], chunk_id=f"{food['food_id']}_0",
                      text=render(food),
                      metadata=ChunkMetadata(
                          section_path=[food["category"], food["names"]["vi"]],
                          chunk_index=0))
        h = compute_chunk_hash(chunk)
        if h in old_hashes:
            continue
        vid = f"{fp.name}_{chunk.chunk_id}_{h[:12]}"
        raw_meta = chunk.metadata.model_dump()
        raw_meta.update({"source_id": food["food_id"],
                         "chunking_strategy": "structure",
                         "dataset": "nutrition",
                         "file_name": fp.name,
                         "embedding_model": EMBEDDING_MODEL,
                         "chunk_hash": h})
        texts.append(chunk.text)
        metas.append(serialize_metatdata(raw_meta))
        ids.append(vid)
        rows.append({"chunk_hash": h, "chunk_index": 0, "vector_id": vid,
                     "extra_meta": raw_meta})
    if texts:
        for i in range(0, len(texts), 32):
            sl = slice(i, i + 32)
            vector_db.add_texts(texts=texts[sl], metadatas=metas[sl], ids=ids[sl])
            log.info("... embedded %d/%d", min(i + 32, len(texts)), len(texts))
    fp_hash = hashlib.sha256("".join(
        sorted(fp.name for fp in files)).encode()).hexdigest()
    upsert_file_and_chunks(FILE_KEY, fp_hash, rows)
    log.info("Done. added=%d skipped=%d", len(texts), len(files) - len(texts))


if __name__ == "__main__":
    main()
