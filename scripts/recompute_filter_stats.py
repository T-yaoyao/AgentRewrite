#!/usr/bin/env python3
"""
Recompute stats block in filter.json:
- total_queries
- equivalent_count / equivalence_rate
- improved_count / improvement_rate
- equivalence_threshold / speed_up_threshold

Improved counting rule:
1) speed_up > speed_up_threshold
2) original_query and rewritten_query are different after normalization:
   - replace literal "\\n" and real newlines with spaces
   - remove all whitespace
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


def normalize_sql_for_compare(sql_text: Any) -> str:
    if sql_text is None:
        return ""
    s = str(sql_text)
    s = s.replace("\\n", " ").replace("\n", " ")
    s = re.sub(r"\s+", "", s)
    return s


def split_filter_payload(data: List[Any]) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Support both formats:
    1) [ [ori_result, re_result], ...items ]
    2) [ ori_result, re_result, stats?, ...items ]
    """
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError("Unexpected filter.json format")

    if isinstance(data[0], list):
        header = data[0]
        if len(header) < 2 or not isinstance(header[0], dict) or not isinstance(header[1], dict):
            raise ValueError("Malformed legacy header in filter.json")
        ori_result, re_result = header[0], header[1]
        items = [x for x in data[1:] if isinstance(x, dict)]
        return ori_result, re_result, items

    ori_result = data[0] if isinstance(data[0], dict) else {}
    re_result = data[1] if isinstance(data[1], dict) else {}
    start = 3 if len(data) > 2 and isinstance(data[2], dict) and "total_queries" in data[2] else 2
    items = [x for x in data[start:] if isinstance(x, dict)]
    return ori_result, re_result, items


def recompute_stats(items: List[Dict[str, Any]], equivalence_threshold: float, speed_up_threshold: float) -> Dict[str, Any]:
    total_queries = len(items)
    equivalent_count = sum(1 for x in items if x.get("equivalence") is True)

    improved_count = 0
    for x in items:
        sp = x.get("speed_up")
        if not isinstance(sp, (int, float)) or sp <= speed_up_threshold:
            continue
        oq = normalize_sql_for_compare(x.get("original_query", ""))
        rq = normalize_sql_for_compare(x.get("rewritten_query", ""))
        if oq != rq:
            improved_count += 1

    return {
        "total_queries": total_queries,
        "equivalent_count": equivalent_count,
        "equivalence_rate": (equivalent_count / total_queries) if total_queries else 0.0,
        "improved_count": improved_count,
        "improvement_rate": (improved_count / total_queries) if total_queries else 0.0,
        "equivalence_threshold": equivalence_threshold,
        "speed_up_threshold": speed_up_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute stats block for filter.json")
    parser.add_argument("filter_path", help="Path to filter.json")
    parser.add_argument("--speed_up_threshold", type=float, default=0.01, help="Speed-up threshold (default: 0.01)")
    parser.add_argument("--equivalence_threshold", type=float, default=0.01, help="Equivalence threshold metadata (default: 0.01)")
    args = parser.parse_args()

    p = Path(args.filter_path)
    data = json.loads(p.read_text(encoding="utf-8"))

    ori_result, re_result, items = split_filter_payload(data)
    stats = recompute_stats(items, args.equivalence_threshold, args.speed_up_threshold)
    output = [ori_result, re_result, stats] + items

    p.write_text(json.dumps(output, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"Updated: {p}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

