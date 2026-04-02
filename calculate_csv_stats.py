#!/usr/bin/env python3
"""
计算 rewritten_queries.json 中 time_cost 的平均值
"""

import json
import sys
from pathlib import Path


def calculate_avg_time_cost(json_file_path):
    """
    计算 JSON 文件中 time_cost 的平均值
    
    Args:
        json_file_path: JSON 文件路径
        
    Returns:
        float | None: 平均 time_cost
    """
    time_costs = []
    
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在: {json_file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ 错误: JSON 解析失败: {e}")
        return None
    except Exception as e:  # pragma: no cover
        print(f"❌ 错误: 读取文件失败: {e}")
        return None

    if not isinstance(data, list):
        print("❌ 错误: JSON 顶层必须是数组")
        return None

    for idx, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            continue
        value = item.get("time_cost")
        if value is None:
            continue
        try:
            time_costs.append(float(value))
        except (TypeError, ValueError):
            print(f"⚠️ 警告: 第 {idx} 条记录的 time_cost 无法解析: {value}")

    if not time_costs:
        print("⚠️ 警告: 没有找到有效的 time_cost 数据")
        return None

    return sum(time_costs) / len(time_costs)


def main():
    """主函数"""
    if len(sys.argv) < 2:
        json_file_path = "output/tpch_claude4.5/rewritten_queries.json"
    else:
        json_file_path = sys.argv[1]

    avg = calculate_avg_time_cost(json_file_path)
    if avg is None:
        sys.exit(1)

    print(f"文件: {Path(json_file_path)}")
    print(f"平均 time_cost: {avg:.6f} 秒")


if __name__ == "__main__":
    main()
