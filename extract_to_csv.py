#!/usr/bin/env python3
"""
从rewritten_queries.json提取数据到CSV文件
"""

import json
import csv
from pathlib import Path

# 输入和输出文件路径
input_file = Path("output/tpch_test/rewritten_queries.json")
output_file = Path("output/tpch_test/rewritten_queries.csv")

# 读取JSON文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 准备CSV数据
csv_rows = []

for item in data:
    # 提取基本信息
    # 将原始SQL也转换为单行
    original_sql = item.get('original_query', '')
    original_sql = ' '.join(original_sql.split()) if original_sql else ''
    
    row = {
        'id': item.get('id', ''),
        'original_query': original_sql,
        'original_costs': '',
        'rewrite_costs': '',
        'costs_reduction_rate': '',
        'time_cost': item.get('time_cost', ''),
        'llm_costs': item.get('llm_costs', ''),
        'rewritten_query': ''
    }
    
    # 提取rewritten_query（从tpch数组中）
    rewritten_query_data = item.get('rewritten_query', {})
    if isinstance(rewritten_query_data, dict) and 'tpch' in rewritten_query_data:
        tpch_list = rewritten_query_data['tpch']
        if tpch_list and len(tpch_list) > 0:
            tpch_item = tpch_list[0]
            # 将多行SQL转换为单行（替换换行符为空格，并清理多余空格）
            rewritten_sql = tpch_item.get('rewritten_query', '')
            # 替换换行符为空格，并清理多余空格
            rewritten_sql = ' '.join(rewritten_sql.split())
            row['rewritten_query'] = rewritten_sql
            row['original_costs'] = tpch_item.get('original_costs', '')
            row['rewrite_costs'] = tpch_item.get('rewrite_costs', '')
            row['costs_reduction_rate'] = tpch_item.get('costs_reduction_rate', '')
    
    csv_rows.append(row)

# 写入CSV文件
fieldnames = ['id', 'original_query', 'rewritten_query', 'original_costs', 
              'rewrite_costs', 'costs_reduction_rate', 'time_cost', 'llm_costs']

with open(output_file, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"✅ 成功提取 {len(csv_rows)} 条记录到 {output_file}")

