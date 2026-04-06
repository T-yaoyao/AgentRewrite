"""
根据 SQL 中出现的表名，从完整 DDL 中裁剪出相关片段，供 LangGraph state 与各 Agent 复用。
并对预置/采集的表级统计做按表过滤。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, List, Set

from src.utils.llm_json_utils import loads_with_repair


def load_schema_file(schema_path: str) -> str:
    if not schema_path:
        return ""
    try:
        return Path(schema_path).read_text(encoding="utf-8")
    except OSError:
        return ""


def extract_table_names_from_sql(sql: str) -> Set[str]:
    """
    从 SQL 中抽取可能引用的表名（小写、无引号规范化）。
    同时保留 schema.table 与裸表名，便于与 DDL 匹配。
    """
    if not sql or not str(sql).strip():
        return set()
    # 去掉单行与块注释，减少误匹配
    s = re.sub(r"--[^\n]*", " ", sql)
    s = re.sub(r"/\*.*?\*/", " ", s, flags=re.DOTALL)
    # 单引号字符串占位，避免 FROM 'x' 干扰
    s = re.sub(r"'(?:''|[^'])*'", " ", s)

    names: Set[str] = set()
    # FROM / JOIN / UPDATE / INTO 后的 [schema.]table
    pat = re.compile(
        r"(?is)\b(?:FROM|JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|FULL\s+JOIN|"
        r"CROSS\s+JOIN|STRAIGHT_JOIN|UPDATE|INTO)\s+"
        r"(?:ONLY\s+)?"
        r"(?:(\w+)\s*\.\s*)?(\w+)"
    )
    skip_next = {
        "select", "where", "group", "order", "having", "limit", "offset",
        "union", "except", "intersect", "on", "using", "lateral", "unnest",
        "values", "case", "when", "set", "dual",
    }
    for m in pat.finditer(s):
        schema, name = m.group(1), m.group(2)
        if not name:
            continue
        nl, sl = name.lower(), (schema or "").lower()
        if nl in skip_next:
            continue
        names.add(nl)
        if sl and sl not in skip_next:
            names.add(f"{sl}.{nl}")

    return names


def _expand_want(tables: Set[str]) -> Set[str]:
    want = {t.lower().strip() for t in tables if t and str(t).strip()}
    for t in list(want):
        if "." in t:
            want.add(t.rsplit(".", 1)[-1])
    return want


def filter_schema_ddl_to_tables(full_schema: str, tables: Set[str]) -> str:
    """仅保留与 tables 相关的 CREATE TABLE / ALTER TABLE 语句。"""
    if not full_schema.strip():
        return ""
    want = _expand_want(tables)
    if not want:
        return full_schema.strip()

    kept: list[str] = []
    for raw in full_schema.split(";"):
        stmt = raw.strip()
        if not stmt:
            continue
        m = re.match(
            r"create\s+table\s+(?:(\w+)\s*\.\s*)?(\w+)\b",
            stmt,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            sch, tname = m.group(1), m.group(2)
            tl = tname.lower()
            ql = f"{sch.lower()}.{tl}" if sch else tl
            if tl in want or ql in want:
                kept.append(stmt)
            continue
        m = re.match(
            r"alter\s+table\s+(?:(\w+)\s*\.\s*)?(\w+)\b",
            stmt,
            re.IGNORECASE | re.DOTALL,
        )
        if m:
            sch, tname = m.group(1), m.group(2)
            tl = tname.lower()
            ql = f"{sch.lower()}.{tl}" if sch else tl
            if tl in want or ql in want:
                kept.append(stmt)
            continue

    if not kept:
        return ""
    return ";\n\n".join(kept) + ";\n"


def build_filtered_schema_content(schema_path: str, sql: str) -> str:
    """
    读取 schema 文件并按 sql 引用表过滤。
    若无法解析出任何表名，或过滤结果为空，则退回完整文件内容，避免误删。
    """
    full = load_schema_file(schema_path)
    if not full.strip():
        return ""
    tables = extract_table_names_from_sql(sql)
    if not tables:
        return full.strip()
    filtered = filter_schema_ddl_to_tables(full, tables).strip()
    if not filtered:
        return full.strip()
    return filtered


def filter_data_statistics_for_sql(data_statistics: Any, sql: str) -> Any:
    """
    将 [[table_name, row_count], ...] 或 JSON 数组字符串按 SQL 引用表过滤。
    - 抽不到表名、解析失败、或过滤后为空：退回原始 data_statistics（完整数据）。
    - 返回类型与输入一致（list 进 list 出，str 进 str 出）。
    """
    if data_statistics is None:
        return data_statistics
    if not sql or not str(sql).strip():
        return data_statistics

    want = extract_table_names_from_sql(sql)
    if not want:
        return data_statistics

    was_str = isinstance(data_statistics, str)
    if was_str:
        try:
            as_list: List = loads_with_repair(data_statistics)
        except (json.JSONDecodeError, TypeError):
            return data_statistics
    elif isinstance(data_statistics, list):
        as_list = data_statistics
    else:
        return data_statistics

    if not as_list or not isinstance(as_list, list):
        return data_statistics

    filtered: List = []
    for row in as_list:
        if not isinstance(row, (list, tuple)) or len(row) < 1:
            continue
        tname = str(row[0]).strip()
        tl = tname.lower()
        base = tl.split(".")[-1]
        if tl in want or base in want:
            filtered.append(list(row) if isinstance(row, tuple) else row)

    if not filtered:
        return data_statistics

    if was_str:
        return json.dumps(filtered, ensure_ascii=False)
    return filtered
