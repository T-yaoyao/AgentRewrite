# 数据分析脚本使用指南

本文档介绍用于分析 SQL 重写结果的工具脚本。

## 目录

- [脚本概览](#脚本概览)
- [详细使用说明](#详细使用说明)
  - [extract_rewrites.py](#extract_rewritespy)
  - [extract_to_csv.py](#extract_to_csvpy)
  - [explain_costs_extractor.py](#explain_costs_extractorpy)
  - [execute_sql_from_csv.py](#execute_sql_from_csvpy)
  - [calculate_csv_stats.py](#calculate_csv_statspy)
  - [analyze_costs.py](#analyze_costspy)
  - [run_explain_costs.sh](#run_explain_costssh)
- [典型工作流程](#典型工作流程)

---

## 脚本概览

| 脚本 | 用途 | 输入 | 输出 |
|------|------|------|------|
| `extract_rewrites.py` | 从 JSON 中提取未重写和已重写的 SQL | JSON 文件 | 2 个 CSV 文件 |
| `extract_to_csv.py` | 从 `rewritten_queries.json` 提取完整数据 | JSON 文件 | CSV 文件 |
| `explain_costs_extractor.py` | 执行 EXPLAIN 获取查询成本 | JSON 文件 | CSV 文件（成本对比） |
| `execute_sql_from_csv.py` | 执行 SQL 并测量执行时间 | CSV/JSON 文件 | CSV 文件（执行时间） |
| `calculate_csv_stats.py` | 计算 CSV 统计信息 | CSV 文件 | 控制台输出 |
| `analyze_costs.py` | 分析成本比较结果 | CSV 文件 | 控制台输出 |
| `run_explain_costs.sh` | 运行 EXPLAIN 成本提取的便捷脚本 | 命令行参数 | 调用 Python 脚本 |

---

## 详细使用说明

### extract_rewrites.py

**用途**：从 `rewritten_queries.json` 中提取两类 SQL：
1. **未重写的 SQL**：原始 SQL 与重写 SQL 完全一致，且未应用任何规则
2. **已重写的 SQL**：应用了规则的重写 SQL

**支持格式**：
- `rewritten_queries.json` 格式（旧结构）
- `QUITE_tpch_63queries.json` 格式（实验结果结构）

**用法**：
```bash
# 基本用法（使用默认路径）
python3 extract_rewrites.py

# 指定输入和输出目录
python3 extract_rewrites.py \
  --input experiments_results/tpch/QUITE_tpch_63queries.json \
  --output-dir experiments_results/tpch

# 自定义输出文件名
python3 extract_rewrites.py \
  --input output/tpch_test/rewritten_queries.json \
  --output-dir output/tpch_test \
  --original-filename original_unrewritten.csv \
   
```

**输出文件**：
- `original_unrewritten.csv`：包含 `id` 和 `original_query` 列
- `rewrite_sql.csv`：包含 `id` 和 `rewritten_query` 列

**参数说明**：
- `--input, -i`：输入 JSON 文件路径（默认：`output/tpch_test/rewritten_queries.json`）
- `--output-dir, -o`：输出目录（默认：`output/tpch_test`）
- `--original-filename`：未重写 SQL 输出文件名（默认：`original_unrewritten.csv`）
- `--rewrite-filename`：重写 SQL 输出文件名（默认：`rewrite_sql.csv`）

---

### extract_to_csv.py

**用途**：从 `rewritten_queries.json` 提取完整数据到 CSV 文件，包括：
- `id`、`original_query`、`rewritten_query`
- `original_costs`、`rewrite_costs`、`costs_reduction_rate`
- `time_cost`、`llm_costs`

**用法**：
```bash
# 直接运行（使用硬编码的默认路径）
python3 extract_to_csv.py
```

**注意**：此脚本的输入输出路径是硬编码的：
- 输入：`output/tpch_test/rewritten_queries.json`
- 输出：`output/tpch_test/rewritten_queries.csv`

如需修改路径，请直接编辑脚本中的 `input_file` 和 `output_file` 变量。

**输出格式**：
- 所有 SQL 查询会被转换为单行（去除换行符）
- 包含所有相关字段的完整数据

---

### explain_costs_extractor.py

**用途**：从 JSON 文件中提取 SQL 查询，对每条 SQL 执行 `EXPLAIN (FORMAT JSON)` 获取成本，并计算成本降低率。

**支持格式**：
1. `rewritten_queries.json` 格式：从 `rewritten_query.tpch[0].rewritten_query` 提取
2. `QUITE_tpch_63queries.json` 格式：从 `original_query` 和 `rewritten_query` 字段提取

**用法**：
```bash
# 基本用法（输出到同目录）
python3 explain_costs_extractor.py \
  --input experiments_results/tpch/QUITE_tpch_63queries.json

# 指定输出文件
python3 explain_costs_extractor.py \
  --input experiments_results/tpch/QUITE_tpch_63queries.json \
  --output experiments_results/tpch/my_costs.csv

# 处理 rewritten_queries.json
python3 explain_costs_extractor.py \
  --input output/tpch_test/rewritten_queries.json
```

**输出文件**：
- 默认：`<输入文件目录>/query_costs_comparison.csv`
- 包含列：`id`、`original_costs`、`rewrite_costs`、`costs_reduction_rate`

**数据库配置**：
脚本使用环境变量配置数据库连接：
- `DB_HOST`（默认：`localhost`）
- `DB_PORT`（默认：`5432`）
- `DB_NAME`（默认：`tpch`）
- `DB_USER`（默认：`postgres`）
- `DB_PASSWORD`（默认：`123456`）

**参数说明**：
- `--input, -i`：输入 JSON 文件路径（必需）
- `--output, -o`：输出 CSV 文件路径（可选，默认：输入文件同目录下的 `query_costs_comparison.csv`）

---

### execute_sql_from_csv.py

**用途**：执行 CSV 或 JSON 文件中的 SQL 查询，测量执行时间。

**功能**：
- 每条 SQL 执行 5 次
- 去掉执行时间中的最小值和最大值
- 对剩余 3 次取平均值作为最终结果

**支持格式**：
1. **CSV 文件**：自动检测 `original_query` 或 `rewritten_query` 列
2. **JSON 文件**：期望结构 `[{"id": "...", "query": "select ..."}, ...]`

**用法**：
```bash
# 执行 CSV 文件中的 SQL
python3 execute_sql_from_csv.py output/tpch_test/rewrite_sql.csv

# 执行 JSON 文件中的 SQL
python3 execute_sql_from_csv.py dataset/queries/tpch_queries.json
```

**输出文件**：
- CSV 输入：`<原文件名>_exec_times.csv`
- JSON 输入：`<原文件名>_exec_times.csv`
- 包含列：`id`、`execution_time_s`

**数据库配置**：
使用环境变量（与 `explain_costs_extractor.py` 相同）

**注意事项**：
- 脚本会自动识别 CSV 表头中的 `original_query` 或 `rewritten_query` 列
- 对于没有结果集的语句（如 INSERT/UPDATE），会自动处理

---

### calculate_csv_stats.py

**用途**：计算 CSV 文件中 `time_cost` 和 `llm_costs` 列的统计信息。

**功能**：
- `time_cost`：平均值、最小值、最大值
- `llm_costs`：总和

**用法**：
```bash
# 使用默认文件
python3 calculate_csv_stats.py

# 指定 CSV 文件
python3 calculate_csv_stats.py output/tpch_test/rewritten_queries.csv
```

**默认文件**：`output/tpch_test/rewritten_queries.csv`

**输出示例**：
```
============================================================
统计信息: rewritten_queries.csv
============================================================
总记录数: 63
有效time_cost记录数: 63

time_cost 统计:
------------------------------------------------------------
平均值 (Mean):           25.123456 秒
最小值 (Min):              1.234567 秒
最大值 (Max):            300.000000 秒

llm_costs 统计:
------------------------------------------------------------
总和 (Sum):                1.234567
============================================================
```

---

### analyze_costs.py

**用途**：分析查询成本比较结果，统计改进、变差和相等的查询。

**功能**：
- 统计成本降低、增加和相等的查询数量
- 计算改进率统计（平均值、中位数、最大值、最小值）
- 按 `query_id` 排序输出所有类别的查询

**用法**：
```bash
# 使用默认文件
python3 analyze_costs.py

# 指定 CSV 文件
python3 analyze_costs.py experiments_results/tpch/query_costs_comparison.csv
```

**默认文件**：`output/tpch_test/rewritten_queries.csv`

**输出内容**：
1. **总体统计**：
   - 总查询数
   - 重写后成本更低的查询数
   - 原始查询成本更低的查询数
   - 成本相等的查询数

2. **改进统计**（当重写更好时）：
   - 平均改进率
   - 中位数改进率
   - 最大改进率
   - 最小改进率

3. **详细列表**（按 `query_id` 排序）：
   - 所有成本降低的查询
   - 所有成本增加的查询
   - 所有成本相等的查询

**支持的 CSV 列名**：
脚本会自动匹配以下列名：
- 原始成本：包含 `original` 和 `cost` 的列
- 重写成本：包含 `rewrite`/`rewritten` 和 `cost` 的列
- 降低率：包含 `reduction` 的列

---

### run_explain_costs.sh

**用途**：运行 `explain_costs_extractor.py` 的便捷 Shell 脚本。

**功能**：
- 检查 Python 3 是否安装
- 检查并安装 `psycopg2-binary`（如需要）
- 将所有命令行参数传递给 Python 脚本

**用法**：
```bash
# 基本用法
./run_explain_costs.sh --input experiments_results/tpch/QUITE_tpch_63queries.json

# 指定输出文件
./run_explain_costs.sh \
  --input experiments_results/tpch/QUITE_tpch_63queries.json \
  --output experiments_results/tpch/my_costs.csv
```

**参数**：所有参数会直接传递给 `explain_costs_extractor.py`

**注意**：首次运行前需要添加执行权限：
```bash
chmod +x run_explain_costs.sh
```

---

## 典型工作流程

### 工作流程 1：分析重写结果

```bash
# 1. 从 JSON 提取完整数据到 CSV
python3 extract_to_csv.py

# 2. 计算统计信息
python3 calculate_csv_stats.py output/tpch_test/rewritten_queries.csv
```

### 工作流程 2：获取 EXPLAIN 成本并分析

```bash
# 1. 从 JSON 提取 SQL 并执行 EXPLAIN 获取成本
python3 explain_costs_extractor.py \
  --input experiments_results/tpch/QUITE_tpch_63queries.json

# 或者使用 Shell 脚本
./run_explain_costs.sh \
  --input experiments_results/tpch/QUITE_tpch_63queries.json

# 2. 分析成本比较结果
python3 analyze_costs.py experiments_results/tpch/query_costs_comparison.csv
```

### 工作流程 3：测量实际执行时间

```bash
# 1. 提取已重写的 SQL
python3 extract_rewrites.py \
  --input experiments_results/tpch/QUITE_tpch_63queries.json \
  --output-dir experiments_results/tpch

# 2. 执行 SQL 并测量执行时间
python3 execute_sql_from_csv.py experiments_results/tpch/rewrite_sql.csv

# 3. 查看执行时间结果
cat experiments_results/tpch/rewrite_sql_exec_times.csv
```

### 工作流程 4：完整分析流程

```bash
# 1. 提取未重写和已重写的 SQL
python3 extract_rewrites.py \
  --input output/tpch_test/rewritten_queries.json

# 2. 获取 EXPLAIN 成本
python3 explain_costs_extractor.py \
  --input output/tpch_test/rewritten_queries.json

# 3. 分析成本
python3 analyze_costs.py output/tpch_test/query_costs_comparison.csv

# 4. 测量实际执行时间（可选）
python3 execute_sql_from_csv.py output/tpch_test/rewrite_sql.csv

# 5. 计算统计信息
python3 calculate_csv_stats.py output/tpch_test/rewritten_queries.csv
```

---

## 注意事项

1. **数据库连接**：
   - `explain_costs_extractor.py` 和 `execute_sql_from_csv.py` 需要数据库连接
   - 确保数据库服务正在运行，且环境变量配置正确

2. **文件格式**：
   - 大部分脚本支持两种 JSON 格式（旧结构和实验结果结构）
   - CSV 文件应使用 UTF-8 编码

3. **执行时间**：
   - `execute_sql_from_csv.py` 会对每条 SQL 执行 5 次，可能需要较长时间
   - 建议在测试环境中运行，避免影响生产环境

4. **错误处理**：
   - 所有脚本都包含基本的错误处理
   - 如果遇到问题，请检查输入文件格式和数据库连接

---

## 依赖项

- Python 3.6+
- `psycopg2` 或 `psycopg2-binary`（用于数据库连接）
- 标准库：`csv`、`json`、`argparse`、`pathlib`、`statistics`

安装依赖：
```bash
pip3 install psycopg2-binary
```

