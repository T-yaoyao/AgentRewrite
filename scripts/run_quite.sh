#!/bin/bash

# QUITE System Run Script - Simple Configuration

# Source the common environment setup
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/setup_env.sh"

# Path variables (now using PROJECT_ROOT from setup_env.sh)
INPUT_QUERIES="${PROJECT_ROOT}/dataset/queries/dsb_test.json"
SCHEMA_FILE="${PROJECT_ROOT}/dataset/schemas/dsb_schemas.sql"
# Must be a directory; run.py writes rewritten_queries.json inside it.
OUTPUT_DIR="${PROJECT_ROOT}/output/without/no_planAnalyer_dsb"

# Feature flags
ENABLE_REWRITER="--enable_rewriter"
SAVE_LOGS="--save_rewriter_logs"

# Execute
python run.py \
    --input_path "${INPUT_QUERIES}" \
    --output_dir "${OUTPUT_DIR}" \
    --schema_file "${SCHEMA_FILE}" \
    ${ENABLE_REWRITER} \
    ${SAVE_LOGS}

