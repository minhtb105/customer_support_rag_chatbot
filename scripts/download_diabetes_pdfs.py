"""
Wrapper nhỏ để tải 3 PDF diabetes public — gọi từ crawl_guidelines.diabetes3

Sử dụng:
  python scripts/download_diabetes_pdfs.py
  python scripts/download_diabetes_pdfs.py --force
"""

import argparse
from crawl_guidelines import download_diabetes_3
import json


def main():
    parser = argparse.ArgumentParser(description="Download 3 diabetes PDFs (WHO + ADA)")
    parser.add_argument("--force", action="store_true", help="Tải lại dù đã tồn tại")
    args = parser.parse_args()
    stats = download_diabetes_3(force=args.force)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
