"""Child worker: docling convert + chunk MỘT (pdf, strategy), print chunks JSON.
Parent (indexer) spawns 1 child/PDF — crash process-wide của docling chỉ giết
child, parent bắt exit-code != 0 rồi đi pymupdf fallback. Không load model
embedding ở child (semantic/hybrid chunking-time embed vẫn ở parent...).

KHÔNG — thiết kế chốt đơn giản hơn: child chỉ CONVERT docling -> DoclingDocument
JSON ra file; parent load JSON + chunk (embed_fn ở parent như cũ). Mọi strategy
dùng chung 1 convert. Crash convert -> parent pymupdf toàn phần.
"""
import os
import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption


def main():
    pdf_path, out_json = sys.argv[1], sys.argv[2]
    opts = PdfPipelineOptions()
    opts.do_ocr = False
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
    doc = conv.convert(pdf_path).document
    doc.save_as_json(out_json)
    print(f"CONVERT-OK {pdf_path}")


if __name__ == "__main__":
    main()
