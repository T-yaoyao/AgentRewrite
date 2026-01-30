#!/usr/bin/env python3
"""
Extract query costs using EXPLAIN (COSTS ON) for original and rewritten queries.
Outputs a CSV file with columns: id, original_query_costs, rewritten_query_costs
"""

import json
import csv
import psycopg2
import re
from typing import Dict, Tuple, Optional

def extract_cost_from_explain(explain_output: str) -> Optional[float]:
    """
    Extract cost from EXPLAIN output.
    Looks for patterns like "cost=0.00..100.00" and extracts the total cost.
    """
    # Find all cost patterns in the output
    cost_pattern = r'cost=(\d+(?:\.\d+)?)\.\.(\d+(?:\.\d+)?)'
    matches = re.findall(cost_pattern, explain_output)

    if not matches:
        return None

    # Get the maximum total cost (the rightmost number in cost=start..end)
    max_cost = 0.0
    for start_cost, end_cost in matches:
        max_cost = max(max_cost, float(end_cost))

    return max_cost if max_cost > 0 else None

def get_query_cost(conn, sql: str) -> Optional[float]:
    """
    Execute EXPLAIN (COSTS ON) for a query and extract the cost.
    """
    try:
        explain_sql = f"EXPLAIN (COSTS ON) {sql}"
        with conn.cursor() as cursor:
            cursor.execute(explain_sql)
            result = cursor.fetchall()

        # Combine all explain lines into a single string
        explain_output = '\n'.join(row[0] for row in result)
        cost = extract_cost_from_explain(explain_output)

        return cost
    except Exception as e:
        print(f"Error getting cost for query: {str(e)}")
        return None

def process_queries(json_file: str, output_csv: str, db_config: Dict[str, str]):
    """
    Process all queries in the JSON file and extract costs.
    """
    # Read JSON file
    print(f"Reading JSON file: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Connect to database
    print("Connecting to database...")
    conn = psycopg2.connect(**db_config)

    results = []

    # Skip the first element which contains statistics
    query_data = data[1:]

    print(f"Processing {len(query_data)} queries...")

    for i, item in enumerate(query_data):
        query_id = item['id']
        original_query = item['original_query']
        rewritten_query = item['rewritten_query']

        print(f"Processing query {query_id} ({i+1}/{len(query_data)})")

        # Get costs for both queries
        original_cost = get_query_cost(conn, original_query)
        rewritten_cost = get_query_cost(conn, rewritten_query)

        # Calculate cost reduction rate
        cost_reduction_rate = None
        if original_cost is not None and rewritten_cost is not None and original_cost > 0:
            cost_reduction_rate = (original_cost - rewritten_cost) / original_cost

        results.append({
            'id': query_id,
            'original_query_costs': original_cost,
            'rewritten_query_costs': rewritten_cost,
            'costs_reduce_rate': cost_reduction_rate
        })

        print(f"  Original cost: {original_cost}")
        print(f"  Rewritten cost: {rewritten_cost}")

    # Close database connection
    conn.close()

    # Write to CSV
    print(f"Writing results to CSV: {output_csv}")
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['id', 'original_query_costs', 'rewritten_query_costs', 'costs_reduce_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print("Done!")

def main():
    # Database configuration - modify these according to your setup
    db_config = {
        'host': 'localhost',
        'database': 'tpch',  # Replace with your database name
        'user': 'postgres',  # Replace with your username
        'password': 'password',  # Replace with your password
        'port': '5432'
    }

    json_file = 'experiments_results/tpch/QUITE_hint_tpch_63queries.json'
    output_csv = 'experiments_results/tpch/QUITE_hint_tpch_63queries_query_costs_comparison.csv'

    process_queries(json_file, output_csv, db_config)

if __name__ == '__main__':
    main()
