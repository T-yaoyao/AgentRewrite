#!/usr/bin/env python3
"""
Analyze the query cost comparison results
"""

import csv
import statistics
from typing import List, Tuple


def sort_by_query_id(query_id: str) -> Tuple[int, str]:
    """
    辅助函数：用于按 query_id 排序
    如果 query_id 是纯数字，按数字大小排序；否则按字符串排序
    """
    try:
        return (0, int(query_id))  # 数字排在前面，按数字大小
    except ValueError:
        return (1, query_id)  # 非数字排在后面，按字符串排序

def load_costs(csv_file: str) -> List[Tuple[str, float, float, float]]:
    """Load costs from CSV file"""
    costs = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_id = row.get('id', '')
            
            # 尝试多种可能的列名
            original_cost = None
            rewritten_cost = None
            reduction_rate = None
            
            # 尝试匹配列名
            for key in row.keys():
                key_lower = key.lower()
                if 'original' in key_lower and 'cost' in key_lower:
                    original_cost = row[key] if row[key] and row[key] != 'None' and row[key].strip() else None
                elif ('rewrite' in key_lower or 'rewritten' in key_lower) and 'cost' in key_lower:
                    rewritten_cost = row[key] if row[key] and row[key] != 'None' and row[key].strip() else None
                elif 'reduction' in key_lower or 'reduce' in key_lower:
                    reduction_rate = row[key] if row[key] and row[key] != 'None' and row[key].strip() else None
            
            # 转换数据类型
            try:
                original_cost = float(original_cost) if original_cost else None
            except (ValueError, TypeError):
                original_cost = None
                
            try:
                rewritten_cost = float(rewritten_cost) if rewritten_cost else None
            except (ValueError, TypeError):
                rewritten_cost = None
                
            try:
                reduction_rate = float(reduction_rate) if reduction_rate else None
            except (ValueError, TypeError):
                reduction_rate = None
            
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
    # Calculate improvement percentage if reduction_rate is not available
    improvement_queries = []
    for query_id, original, rewritten, reduction_rate in costs:
        if original is None or rewritten is None:
            continue
        if rewritten < original:
            if reduction_rate is not None:
                # Check if reduction_rate is in decimal form (0-1) or percentage form (0-100)
                # If reduction_rate > 1, it's already a percentage; otherwise it's a decimal
                if reduction_rate > 1:
                    # Already a percentage (e.g., 76.63 means 76.63%)
                    improvement_pct = reduction_rate
                else:
                    # Decimal form (e.g., 0.7663 means 76.63%), convert to percentage
                    improvement_pct = reduction_rate * 100
            else:
                # Calculate improvement percentage
                improvement_pct = (original - rewritten) / original * 100
            improvement_queries.append((query_id, original, rewritten, improvement_pct))
    
    # 按 improvement 从大到小排序
    improvement_queries = sorted(improvement_queries, key=lambda x: x[3], reverse=True)

    print(f"\n=== All Queries with Cost Reduction (Improvement) === [{len(improvement_queries)} queries]")
    if improvement_queries:
        for query_id, original, rewritten, improvement in improvement_queries:
            print(f"Query {query_id}: {original:.2f} -> {rewritten:.2f} ({improvement:.2f}% improvement)")
    else:
        print("No queries with cost improvement found.")

    # Show all cost increases (degradation)
    worse_queries = []
    for query_id, original, rewritten, reduction_rate in costs:
        if original is None or rewritten is None:
            continue
        if rewritten > original:
            if reduction_rate is not None:
                # Check if reduction_rate is in decimal form (0-1) or percentage form (0-100)
                # If reduction_rate > 1 or < -1, it's already a percentage; otherwise it's a decimal
                if abs(reduction_rate) > 1:
                    # Already a percentage (e.g., -146.48 means -146.48%)
                    degradation_pct = abs(reduction_rate)
                else:
                    # Decimal form (e.g., -0.14648 means -14.648%), convert to percentage
                    degradation_pct = abs(reduction_rate) * 100
            else:
                # Calculate degradation percentage
                degradation_pct = (rewritten - original) / original * 100
            worse_queries.append((query_id, original, rewritten, degradation_pct))
    
    # 按 degradation 从小到大排序
    worse_queries = sorted(worse_queries, key=lambda x: x[3])

    print(f"\n=== All Queries with Cost Increase (Worse) === [{len(worse_queries)} queries]")
    if worse_queries:
        for query_id, original, rewritten, degradation in worse_queries:
            print(f"Query {query_id}: {original:.2f} -> {rewritten:.2f} ({abs(degradation):.2f}% worse)")
    else:
        print("No queries with cost increase found.")

    # Show queries with equal cost
    equal_queries = [(query_id, original, rewritten)
                     for query_id, original, rewritten, reduction_rate in costs
                     if original is not None and rewritten is not None and original == rewritten]
    
    # 按 query_id 排序
    equal_queries = sorted(equal_queries, key=lambda x: sort_by_query_id(x[0]))

    print(f"\n=== Queries with Equal Cost === [{len(equal_queries)} queries]")
    if equal_queries:
        for query_id, original, rewritten in equal_queries:
            print(f"Query {query_id}: {original:.2f} -> {rewritten:.2f} (equal cost)")
    else:
        print("No queries with equal cost found.")

def main():
    import sys
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = 'output/tpch_test/rewritten_queries.csv'
    
    print(f"Analyzing costs from: {csv_file}")
    costs = load_costs(csv_file)
    analyze_costs(costs)

if __name__ == '__main__':
    main()
