#!/usr/bin/env python3
"""
One-shot migration: remove legacy cost fields from global memory (Chroma) metadata.

Drops per-record metadata keys: cost_reduction_rate, original_cost, rewritten_cost.
Embeddings and documents are unchanged.

Usage (from project root):
  python scripts/strip_global_memory_cost_metadata.py
  python scripts/strip_global_memory_cost_metadata.py --storage-path /path/to/chroma_db
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.utils.path_config import setup_python_path, load_project_env

    setup_python_path()
    load_project_env()

    from src.Query_Rewriter.global_memory import GlobalMemoryManager

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--storage-path",
        default=None,
        help="Chroma persistence directory (default: data/global_memory/chroma_db under project root)",
    )
    args = parser.parse_args()

    gm = GlobalMemoryManager(storage_path=args.storage_path)
    n = gm.strip_legacy_cost_metadata()
    print(f"strip_legacy_cost_metadata: updated {n} record(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
