#!/usr/bin/env python3
"""
Analyze the query cost comparison results
"""

import csv
import statistics
from typing import List, Tuple

def load_costs(csv_file: str) -> List[Tuple[str, float, float, float]]:
    """Load costs from CSV file"""
    costs = []
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row['id']
            original_cost = float(row['original_query_costs']) if row['original_query_costs'] and row['original_query_costs'] != 'None' else None
            rewritten_cost = float(row['rewritten_query_costs']) if row['rewritten_query_costs'] and row['rewritten_query_costs'] != 'None' else None
            reduction_rate = float(row['costs_reduce_rate']) if row.get('costs_reduce_rate') and row['costs_reduce_rate'] and row['costs_reduce_rate'] != 'None' else None
            costs.append((query_id, original_cost, rewritten_cost, reduction_rate))
    return costs

def analyze_costs(costs: List[Tuple[str, float, float, float]]):
    """Analyze the cost data"""
    print("=== Query Cost Analysis ===")
    print(f"Total queries analyzed: {len(costs)}")

    # Calculate improvements
    improvements = []
    original_better = 0
    rewritten_better = 0
    equal_cost = 0

    for query_id, original, rewritten, reduction_rate in costs:
        if original is None or rewritten is None:
            continue

        if rewritten < original:
            rewritten_better += 1
            improvement = (original - rewritten) / original * 100
            improvements.append(improvement)
        elif rewritten > original:
            original_better += 1
        else:
            equal_cost += 1

    print(f"\nCost comparison:")
    print(f"  Rewritten queries with lower cost: {rewritten_better}")
    print(f"  Original queries with lower cost: {original_better}")
    print(f"  Queries with equal cost: {equal_cost}")

    if improvements:
        print("\nImprovement statistics (when rewritten is better):")
        print(f"  Average improvement: {statistics.mean(improvements):.2f}%")
        print(f"  Median improvement: {statistics.median(improvements):.2f}%")
        print(f"  Max improvement: {max(improvements):.2f}%")
        print(f"  Min improvement: {min(improvements):.2f}%")

    # Show all improvements (cost decreased)
    improvement_queries = sorted(
        [(query_id, original, rewritten, reduction_rate * 100)
         for query_id, original, rewritten, reduction_rate in costs
         if reduction_rate is not None and reduction_rate > 0],
        key=lambda x: x[3], reverse=True  # Sort by improvement percentage descending
    )

    print(f"\n=== All Queries with Cost Reduction (Improvement) === [{len(improvement_queries)} queries]")
    if improvement_queries:
        for query_id, original, rewritten, improvement in improvement_queries:
            print(f"Query {query_id}: {original:.2f} -> {rewritten:.2f} ({improvement:.2f}% improvement)")
    else:
        print("No queries with cost improvement found.")

    # Show all cost increases (degradation)
    worse_queries = sorted(
        [(query_id, original, rewritten, reduction_rate * 100)
         for query_id, original, rewritten, reduction_rate in costs
         if reduction_rate is not None and reduction_rate < 0],
        key=lambda x: x[3]  # Sort by degradation percentage ascending (most negative first)
    )

    print(f"\n=== All Queries with Cost Increase (Worse) === [{len(worse_queries)} queries]")
    if worse_queries:
        for query_id, original, rewritten, degradation in worse_queries:
            print(f"Query {query_id}: {original:.2f} -> {rewritten:.2f} ({abs(degradation):.2f}% worse)")
    else:
        print("No queries with cost increase found.")

    # Show queries with equal cost
    equal_queries = [(query_id, original, rewritten)
                     for query_id, original, rewritten, reduction_rate in costs
                     if reduction_rate is not None and reduction_rate == 0]

    print(f"\n=== Queries with Equal Cost === [{len(equal_queries)} queries]")
    if equal_queries:
        for query_id, original, rewritten in equal_queries:
            print(f"Query {query_id}: {original:.2f} -> {rewritten:.2f} (equal cost)")
    else:
        print("No queries with equal cost found.")

def main():
    csv_file = 'experiments_results/tpch/query_costs_comparison.csv'
    costs = load_costs(csv_file)
    analyze_costs(costs)

if __name__ == '__main__':
    main()
