#!/bin/bash

# Script to run the EXPLAIN costs extraction
# Usage: ./run_explain_costs.sh [--input INPUT_FILE] [--output OUTPUT_FILE] [--database DB_NAME]
# Example: ./run_explain_costs.sh --input experiments_results/tpch/QUITE_tpch_63queries.json
# Example: ./run_explain_costs.sh --input experiments_results/dsb/QUITE_dsb_156queries.json --database dsb

echo "Running EXPLAIN costs extraction..."

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if required packages are installed
python3 -c "import psycopg2" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "psycopg2 is not installed. Installing..."
    pip3 install psycopg2-binary
fi

# Run the extraction script with all arguments passed through
python3 explain_costs_extractor.py "$@"

echo "Extraction completed. Check the output CSV file for results."
