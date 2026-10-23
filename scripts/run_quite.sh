#!/bin/bash
set -e

# QUITE / PASR run script.
# Usage:
#   bash scripts/run_quite.sh <dataset>
#   bash scripts/run_quite.sh --dataset <dataset>
#   bash scripts/run_quite.sh <dataset> --database <db_name>
# dataset: calcite | tpch | dsb
# Does not clear LinUCB stats or Chroma memory.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/setup_env.sh"

DATASET=""
DATABASE=""

usage() {
    echo "用法: bash scripts/run_quite.sh <dataset> [--database <db_name>]"
    echo "      bash scripts/run_quite.sh --dataset <dataset> [--database <db_name>]"
    echo "dataset: calcite | tpch | dsb"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|-s)
            [[ -n "${2:-}" ]] || usage
            DATASET="$2"
            shift 2
            ;;
        --database|-d)
            [[ -n "${2:-}" ]] || usage
            DATABASE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "未知参数: $1"
            usage
            ;;
        *)
            if [[ -z "$DATASET" ]]; then
                DATASET="$1"
                shift
            else
                echo "多余位置参数: $1"
                usage
            fi
            ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "请指定数据集。"
    usage
fi

case "$DATASET" in
    calcite|tpch|dsb)
        ;;
    *)
        echo "不支持的数据集: $DATASET（仅支持 calcite | tpch | dsb）"
        exit 1
        ;;
esac

INPUT_QUERIES="${PROJECT_ROOT}/dataset/queries/${DATASET}_test.json"
SCHEMA_FILE="${PROJECT_ROOT}/dataset/schemas/${DATASET}_schemas.sql"
OUTPUT_DIR="${PROJECT_ROOT}/output/china_model/flash/${DATASET}"

if [[ ! -f "$INPUT_QUERIES" ]]; then
    echo "查询文件不存在: $INPUT_QUERIES"
    exit 1
fi
if [[ ! -f "$SCHEMA_FILE" ]]; then
    echo "Schema 文件不存在: $SCHEMA_FILE"
    exit 1
fi

# DB: explicit --database wins; otherwise use dataset name (overrides .env DB_NAME).
if [[ -n "$DATABASE" ]]; then
    DATABASE_ARGS=(--database "$DATABASE")
else
    DATABASE_ARGS=(--database "$DATASET")
fi

ENABLE_REWRITER="--enable_rewriter"
SAVE_LOGS="--save_rewriter_logs"

echo "数据集: ${DATASET}"
echo "查询:   ${INPUT_QUERIES}"
echo "Schema: ${SCHEMA_FILE}"
echo "输出:   ${OUTPUT_DIR}"
echo "数据库: ${DATABASE_ARGS[*]}"

python run.py \
    --input_path "${INPUT_QUERIES}" \
    --output_dir "${OUTPUT_DIR}" \
    --schema_file "${SCHEMA_FILE}" \
    ${ENABLE_REWRITER} \
    ${SAVE_LOGS} \
    "${DATABASE_ARGS[@]}"
