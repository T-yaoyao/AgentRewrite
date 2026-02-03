#!/usr/bin/env python3
"""
计算CSV文件中time_cost的平均值和llm_costs的总和
"""

import csv
import sys
from pathlib import Path

def calculate_stats(csv_file_path):
    """
    计算CSV文件的统计信息
    
    Args:
        csv_file_path: CSV文件路径
        
    Returns:
        dict: 包含统计信息的字典
    """
    time_costs = []
    llm_costs_sum = 0.0
    total_rows = 0
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                total_rows += 1
                
                # 提取time_cost
                time_cost_str = row.get('time_cost', '').strip()
                if time_cost_str:
                    try:
                        time_cost = float(time_cost_str)
                        time_costs.append(time_cost)
                    except ValueError:
                        print(f"⚠️ 警告: 无法解析time_cost值 '{time_cost_str}' (行 {total_rows + 1})")
                
                # 累加llm_costs
                llm_cost_str = row.get('llm_costs', '').strip()
                if llm_cost_str:
                    try:
                        llm_cost = float(llm_cost_str)
                        llm_costs_sum += llm_cost
                    except ValueError:
                        print(f"⚠️ 警告: 无法解析llm_costs值 '{llm_cost_str}' (行 {total_rows + 1})")
    
    except FileNotFoundError:
        print(f"❌ 错误: 文件 {csv_file_path} 不存在")
        return None
    except Exception as e:
        print(f"❌ 读取文件时出错: {e}")
        return None
    
    if not time_costs:
        print("⚠️ 警告: 没有找到有效的time_cost数据")
        return None
    
    # 计算平均值
    time_cost_avg = sum(time_costs) / len(time_costs)
    
    return {
        'total_rows': total_rows,
        'valid_time_cost_rows': len(time_costs),
        'time_cost_avg': time_cost_avg,
        'llm_costs_sum': llm_costs_sum,
        'time_cost_min': min(time_costs),
        'time_cost_max': max(time_costs)
    }


def print_stats(stats, csv_file_path):
    """
    打印统计信息
    
    Args:
        stats: 统计信息字典
        csv_file_path: CSV文件路径
    """
    if stats is None:
        return
    
    print("=" * 60)
    print(f"统计信息: {Path(csv_file_path).name}")
    print("=" * 60)
    print(f"总记录数: {stats['total_rows']}")
    print(f"有效time_cost记录数: {stats['valid_time_cost_rows']}")
    print()
    print("time_cost 统计:")
    print("-" * 60)
    print(f"平均值 (Mean):     {stats['time_cost_avg']:>15.6f} 秒")
    print(f"最小值 (Min):      {stats['time_cost_min']:>15.6f} 秒")
    print(f"最大值 (Max):      {stats['time_cost_max']:>15.6f} 秒")
    print()
    print("llm_costs 统计:")
    print("-" * 60)
    print(f"总和 (Sum):        {stats['llm_costs_sum']:>15.6f}")
    print("=" * 60)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        # 默认使用output/tpch_test/rewritten_queries.csv
        csv_file_path = "output/tpch_test/rewritten_queries.csv"
    else:
        csv_file_path = sys.argv[1]
    
    # 计算统计信息
    stats = calculate_stats(csv_file_path)
    
    # 打印结果
    print_stats(stats, csv_file_path)


if __name__ == "__main__":
    main()


