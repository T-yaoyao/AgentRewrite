#!/bin/bash

# Script to run the EXPLAIN costs extraction
# Make sure to modify the database configuration in the Python script first

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

# Run the extraction script
python3 explain_costs_extractor.py

echo "Extraction completed. Check query_costs_comparison.csv for results."
