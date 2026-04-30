"""
Compare two PostgreSQL EXPLAIN (FORMAT JSON) plan trees for structural signals.

Used when estimated Total Cost is worse but the plan may still run faster
(e.g. Seq Scan -> Index Scan, Nested Loop -> Hash Join, lower row estimates).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# Node types: scans
_SEQ = "Seq Scan"
_INDEX = ("Index Scan", "Index Only Scan")
_BITMAP = ("Bitmap Index Scan", "Bitmap Heap Scan")

# Join nodes
_NL = "Nested Loop"
_HJ = "Hash Join"
_MJ = "Merge Join"
_ROWS_DRAMATIC_REDUCTION_RATIO = 0.10


def _parse_explain_to_plan_roots(explain_payload: Any) -> List[Dict[str, Any]]:
    """
    Accept JSON str / dict / list as returned by DBMS_RAW_EXPLAIN_JSON_Tool
    and return a list of root Plan dicts.
    """
    if explain_payload is None:
        return []
    if isinstance(explain_payload, str):
        s = explain_payload.strip()
        if not s:
            return []
        try:
            explain_payload = json.loads(s)
        except json.JSONDecodeError:
            return []
    if isinstance(explain_payload, dict) and "error" in explain_payload and "Plan" not in explain_payload:
        return []
    if isinstance(explain_payload, list):
        roots: List[Dict[str, Any]] = []
        for item in explain_payload:
            if not isinstance(item, dict):
                continue
            if "error" in item and "Plan" not in item:
                continue
            if "Plan" in item and isinstance(item["Plan"], dict):
                roots.append(item["Plan"])
        return roots
    if isinstance(explain_payload, dict) and "Plan" in explain_payload and isinstance(
        explain_payload["Plan"], dict
    ):
        return [explain_payload["Plan"]]
    return []


def _accumulate_node_metrics(node: Optional[Dict[str, Any]], acc: Dict[str, Any]) -> None:
    if not node:
        return
    nt = node.get("Node Type") or ""
    if nt:
        acc["by_type"][nt] = acc["by_type"].get(nt, 0) + 1
    if nt == _SEQ:
        acc["seq_scan"] += 1
    if nt in _INDEX:
        acc["index_scan"] += 1
    if nt in _BITMAP:
        acc["bitmap_scan"] += 1
    if nt == _NL:
        acc["nested_loop"] += 1
    if nt == _HJ:
        acc["hash_join"] += 1
    if nt == _MJ:
        acc["merge_join"] += 1
    pr = node.get("Plan Rows")
    if pr is not None:
        try:
            acc["sum_plan_rows"] += float(pr)
        except (TypeError, ValueError):
            pass
    for child in node.get("Plans") or []:
        if isinstance(child, dict):
            _accumulate_node_metrics(child, acc)


def aggregate_plan_metrics(plan_root: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    acc = {
        "by_type": {},
        "seq_scan": 0,
        "index_scan": 0,
        "bitmap_scan": 0,
        "nested_loop": 0,
        "hash_join": 0,
        "merge_join": 0,
        "sum_plan_rows": 0.0,
    }
    _accumulate_node_metrics(plan_root, acc)
    acc["index_like"] = acc["index_scan"] + acc["bitmap_scan"]
    acc["join_hash_merge"] = acc["hash_join"] + acc["merge_join"]
    return acc


def _merge_metrics(roots: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not roots:
        return {
            "by_type": {},
            "seq_scan": 0,
            "index_scan": 0,
            "bitmap_scan": 0,
            "nested_loop": 0,
            "hash_join": 0,
            "merge_join": 0,
            "sum_plan_rows": 0.0,
            "index_like": 0,
            "join_hash_merge": 0,
        }
    merged = {
        "by_type": {},
        "seq_scan": 0,
        "index_scan": 0,
        "bitmap_scan": 0,
        "nested_loop": 0,
        "hash_join": 0,
        "merge_join": 0,
        "sum_plan_rows": 0.0,
    }
    for r in roots:
        m = aggregate_plan_metrics(r)
        for k, v in m["by_type"].items():
            merged["by_type"][k] = merged["by_type"].get(k, 0) + v
        merged["seq_scan"] += m["seq_scan"]
        merged["index_scan"] += m["index_scan"]
        merged["bitmap_scan"] += m["bitmap_scan"]
        merged["nested_loop"] += m["nested_loop"]
        merged["hash_join"] += m["hash_join"]
        merged["merge_join"] += m["merge_join"]
        merged["sum_plan_rows"] += m["sum_plan_rows"]
    merged["index_like"] = merged["index_scan"] + merged["bitmap_scan"]
    merged["join_hash_merge"] = merged["hash_join"] + merged["merge_join"]
    return merged


def _rows_improved(o_rows: float, r_rows: float, min_rows: float = 100.0) -> bool:
    if o_rows < min_rows:
        return False
    if o_rows <= 0:
        return False
    return (r_rows / o_rows) <= 0.95


def _scan_structurally_better(om: Dict[str, Any], rm: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Seq Scan 减少且伴随更多索引/位图类访问，或全表扫明显减少（>=2）。
    """
    reasons: List[str] = []
    d_seq = om["seq_scan"] - rm["seq_scan"]
    d_idx = rm["index_like"] - om["index_like"]
    if d_seq >= 1 and d_idx >= 1:
        reasons.append(
            f"Scan: SeqScan {om['seq_scan']}->{rm['seq_scan']}, index-like {om['index_like']}->{rm['index_like']}"
        )
        return True, reasons
    if d_seq >= 2:
        reasons.append(f"SeqScan reduced by {d_seq}")
        return True, reasons
    return False, []


def _join_structurally_better(om: Dict[str, Any], rm: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Nested Loop 减少且 Hash/Merge Join 总次数不减少（或净增加大表连接能力）。
    """
    reasons: List[str] = []
    d_nl = om["nested_loop"] - rm["nested_loop"]
    o_hm = om["join_hash_merge"]
    r_hm = rm["join_hash_merge"]
    d_hm = r_hm - o_hm
    if d_nl >= 1 and d_hm >= 0 and r_hm > 0:
        reasons.append(
            f"Join: NestedLoop {om['nested_loop']}->{rm['nested_loop']}, "
            f"Hash/Merge {o_hm}->{r_hm}"
        )
        return True, reasons
    if d_nl >= 1 and d_hm >= 1:
        reasons.append(
            f"Join: NestedLoop {om['nested_loop']}->{rm['nested_loop']}, Hash/Merge +{d_hm}"
        )
        return True, reasons
    return False, []


def compare_explain_plan_structures(original_explain: Any, rewritten_explain: Any) -> Dict[str, Any]:
    """
    Compare original vs rewritten EXPLAIN JSON for heuristics that suggest real improvement
    despite higher optimizer Total Cost.

    Returns:
        original_metrics, rewritten_metrics, deltas, structural_improvement (bool),
        preserve_despite_higher_cost (bool), reasons (str list)
    """
    o_roots = _parse_explain_to_plan_roots(original_explain)
    r_roots = _parse_explain_to_plan_roots(rewritten_explain)
    if not o_roots or not r_roots:
        return {
            "ok": False,
            "original_metrics": _merge_metrics(o_roots),
            "rewritten_metrics": _merge_metrics(r_roots),
            "structural_improvement": False,
            "preserve_despite_higher_cost": False,
            "rows_ratio": None,
            "rows_drastically_reduced": False,
            "reasons": [],
        }

    om = _merge_metrics(o_roots)
    rm = _merge_metrics(r_roots)

    scan_ok, scan_rs = _scan_structurally_better(om, rm)
    join_ok, join_rs = _join_structurally_better(om, rm)
    rows_ok = _rows_improved(om["sum_plan_rows"], rm["sum_plan_rows"])
    rows_ratio: Optional[float] = None
    rows_drastically_reduced = False
    rows_r: List[str] = []
    if om["sum_plan_rows"] > 0:
        rows_ratio = rm["sum_plan_rows"] / om["sum_plan_rows"]
        row_pct = rows_ratio * 100.0
        rows_drastically_reduced = rows_ratio <= _ROWS_DRAMATIC_REDUCTION_RATIO
    if rows_ok and rows_ratio is not None:
        rows_r.append(
            f"行数估计求和(各节点 Plan Rows 累加，非 EXPLAIN 的 Total Cost): "
            f"{om['sum_plan_rows']:.1f} -> {rm['sum_plan_rows']:.1f} "
            f"(重写侧约为原的 {row_pct:.4f}%)"
        )
    if rows_drastically_reduced and rows_ratio is not None:
        rows_r.append(
            f"强信号: 行数估计求和大幅下降，重写侧约为原的 {(rows_ratio * 100.0):.4f}% "
            f"(<= {_ROWS_DRAMATIC_REDUCTION_RATIO * 100:.0f}%)"
        )

    reasons = list(scan_rs) + list(join_rs) + rows_r
    structural = scan_ok or join_ok or rows_ok or rows_drastically_reduced
    # Only skip cost rollback when we have a concrete structural signal
    preserve = structural or rows_drastically_reduced

    return {
        "ok": True,
        "original_metrics": om,
        "rewritten_metrics": rm,
        "structural_improvement": structural,
        "preserve_despite_higher_cost": preserve,
        "rows_ratio": rows_ratio,
        "rows_drastically_reduced": rows_drastically_reduced,
        "reasons": reasons,
    }
