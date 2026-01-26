#!/usr/bin/env python3
"""
计算CSV文件中执行时间的统计信息
包括：平均值、中位数、75th百分位数、95th百分位数
单位从毫秒转换为秒
"""

import csv
import sys
import numpy as np
from pathlib import Path


def calculate_statistics(csv_file_path):
    """
    读取CSV文件并计算统计信息
    
    Args:
        csv_file_path: CSV文件路径
        
    Returns:
        dict: 包含统计信息的字典
    """
    execution_times_ms = []
    
    # 读取CSV文件
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            # 检测分隔符（可能是逗号或制表符）
            sample = f.read(1024)
            f.seek(0)
            delimiter = '\t' if '\t' in sample else ','
            
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                # 跳过空行
                if not row or not any(row.values()):
                    continue
                    
                # 获取执行时间（可能是execution_time_ms字段）
                time_value = None
                for key in row.keys():
                    if 'time' in key.lower() or 'execution' in key.lower():
                        time_value = row[key]
                        break
                
                if time_value:
                    try:
                        time_ms = float(time_value.strip())
                        if time_ms > 0:  # 忽略无效值
                            execution_times_ms.append(time_ms)
                    except ValueError:
                        continue
    
    except FileNotFoundError:
        print(f"错误: 文件 {csv_file_path} 不存在")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
    
    if not execution_times_ms:
        print("错误: 没有找到有效的执行时间数据")
        return None
    
    # 转换为numpy数组以便计算
    times_array = np.array(execution_times_ms)
    
    # 计算统计信息（单位：毫秒）
    mean_ms = np.mean(times_array)
    median_ms = np.median(times_array)
    percentile_75_ms = np.percentile(times_array, 75)
    percentile_95_ms = np.percentile(times_array, 95)
    
    # 转换为秒
    mean_s = mean_ms / 1000.0
    median_s = median_ms / 1000.0
    percentile_75_s = percentile_75_ms / 1000.0
    percentile_95_s = percentile_95_ms / 1000.0
    
    return {
        'count': len(execution_times_ms),
        'mean_ms': mean_ms,
        'median_ms': median_ms,
        'percentile_75_ms': percentile_75_ms,
        'percentile_95_ms': percentile_95_ms,
        'mean_s': mean_s,
        'median_s': median_s,
        'percentile_75_s': percentile_75_s,
        'percentile_95_s': percentile_95_s,
        'min_ms': np.min(times_array),
        'max_ms': np.max(times_array),
        'min_s': np.min(times_array) / 1000.0,
        'max_s': np.max(times_array) / 1000.0
    }


def print_statistics(stats, csv_file_path):
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
    print(f"数据点数量: {stats['count']}")
    print()
    print("执行时间统计 (单位: 秒):")
    print("-" * 60)
    print(f"平均值 (Mean):        {stats['mean_s']:>15.6f} s")
    print(f"中位数 (Median):      {stats['median_s']:>15.6f} s")
    print(f"75th百分位数:        {stats['percentile_75_s']:>15.6f} s")
    print(f"95th百分位数:        {stats['percentile_95_s']:>15.6f} s")
    print(f"最小值 (Min):        {stats['min_s']:>15.6f} s")
    print(f"最大值 (Max):        {stats['max_s']:>15.6f} s")
    print()
    print("执行时间统计 (单位: 毫秒):")
    print("-" * 60)
    print(f"平均值 (Mean):        {stats['mean_ms']:>15.6f} ms")
    print(f"中位数 (Median):      {stats['median_ms']:>15.6f} ms")
    print(f"75th百分位数:        {stats['percentile_75_ms']:>15.6f} ms")
    print(f"95th百分位数:        {stats['percentile_95_ms']:>15.6f} ms")
    print(f"最小值 (Min):        {stats['min_ms']:>15.6f} ms")
    print(f"最大值 (Max):        {stats['max_ms']:>15.6f} ms")
    print("=" * 60)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python calculate_statistics.py <csv_file_path>")
        print("示例: python calculate_statistics.py dataset/queries/original_tpch_queries.csv")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    
    # 计算统计信息
    stats = calculate_statistics(csv_file_path)
    
    # 打印结果
    print_statistics(stats, csv_file_path)
    
    # 可选：保存结果到文件
    if stats:
        output_file = Path(csv_file_path).with_suffix('.statistics.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"统计信息: {Path(csv_file_path).name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"数据点数量: {stats['count']}\n\n")
            f.write("执行时间统计 (单位: 秒):\n")
            f.write("-" * 60 + "\n")
            f.write(f"平均值 (Mean):        {stats['mean_s']:>15.6f} s\n")
            f.write(f"中位数 (Median):      {stats['median_s']:>15.6f} s\n")
            f.write(f"75th百分位数:        {stats['percentile_75_s']:>15.6f} s\n")
            f.write(f"95th百分位数:        {stats['percentile_95_s']:>15.6f} s\n")
            f.write(f"最小值 (Min):        {stats['min_s']:>15.6f} s\n")
            f.write(f"最大值 (Max):        {stats['max_s']:>15.6f} s\n")
            f.write("\n")
            f.write("执行时间统计 (单位: 毫秒):\n")
            f.write("-" * 60 + "\n")
            f.write(f"平均值 (Mean):        {stats['mean_ms']:>15.6f} ms\n")
            f.write(f"中位数 (Median):      {stats['median_ms']:>15.6f} ms\n")
            f.write(f"75th百分位数:        {stats['percentile_75_ms']:>15.6f} ms\n")
            f.write(f"95th百分位数:        {stats['percentile_95_ms']:>15.6f} ms\n")
            f.write(f"最小值 (Min):        {stats['min_ms']:>15.6f} ms\n")
            f.write(f"最大值 (Max):        {stats['max_ms']:>15.6f} ms\n")
            f.write("=" * 60 + "\n")
        print(f"\n统计结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

