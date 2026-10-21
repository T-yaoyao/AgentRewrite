#!/bin/bash
set -e

# QUITE System Run Script - Simple Configuration

# Source the common environment setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/setup_env.sh"

# Path variables (now using PROJECT_ROOT from setup_env.sh)
INPUT_QUERIES="${PROJECT_ROOT}/dataset/queries/dsb_test.json"
SCHEMA_FILE="${PROJECT_ROOT}/dataset/schemas/dsb_schemas.sql"
OUTPUT_DIR="${PROJECT_ROOT}/output/china_model/pasr/flash/dsb"

# Feature flags
ENABLE_REWRITER="--enable_rewriter"
SAVE_LOGS="--save_rewriter_logs"

# Optional database: bash scripts/run_quite.sh --database dsb
# or bash scripts/run_quite.sh dsb. Omitted values use DB_NAME from .env.
DATABASE_ARGS=()
if [[ $# -gt 0 ]]; then
    case "$1" in
        --database|-d)
            if [[ -z "${2:-}" ]]; then
                echo "缺少数据库名。用法: bash scripts/run_quite.sh [--database 数据库名]"
                exit 1
            fi
            DATABASE_ARGS=(--database "$2")
            ;;
        -*)
            echo "未知参数: $1"
            exit 1
            ;;
        *)
            DATABASE_ARGS=(--database "$1")
            ;;
    esac
fi

# Clear LinUCB statistics and the example memory before every run.
python - << 'PY'
import json
import sys
from pathlib import Path

root = Path(__import__("os").environ["PROJECT_ROOT"])
stats_path = root / "data" / "global_memory" / "rule_bandit_stats.json"
stats_path.parent.mkdir(parents=True, exist_ok=True)
stats_path.write_text("{}\n", encoding="utf-8")
print(f"LinUCB 统计已清空: {stats_path}")

try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
    import chromadb
    from chromadb.config import Settings
except Exception as exc:
    print(f"记忆库清空失败: {exc}")
    sys.exit(1)

name = "sql_optimization_knowledge"
client = chromadb.PersistentClient(
    path=str(root / "data" / "global_memory" / "chroma_db"),
    settings=Settings(anonymized_telemetry=False),
)
try:
    before = client.get_collection(name).count()
except Exception:
    before = 0
try:
    client.delete_collection(name)
except Exception:
    pass
after = client.get_or_create_collection(
    name=name,
    metadata={"description": "SQL optimization knowledge base"},
).count()
print(f"记忆库已清空: {before} -> {after}")
PY

# Execute
python run.py \
    --input_path "${INPUT_QUERIES}" \
    --output_dir "${OUTPUT_DIR}" \
    --schema_file "${SCHEMA_FILE}" \
    ${ENABLE_REWRITER} \
    ${SAVE_LOGS} \
    "${DATABASE_ARGS[@]}"

