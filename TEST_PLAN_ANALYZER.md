# 智能计划分析器测试指南

## 概述

`test_plan_analyzer.py` 是一个独立的测试工具，用于测试智能计划分析器（Intelligent Plan Analyzer）的功能。

## 功能特性

- ✅ 自动识别执行计划中的性能瓶颈
- ✅ 量化分析：成本估算、代价占比统计
- ✅ 生成针对性的优化建议
- ✅ 支持单语句和多语句 SQL
- ✅ 支持交互式模式
- ✅ 详细的格式化输出

## 使用方法

### 1. 测试单个 SQL 查询

```bash
python test_plan_analyzer.py --sql "SELECT * FROM orders WHERE o_orderdate >= '1997-01-01'"
```

### 2. 从文件读取 SQL

```bash
python test_plan_analyzer.py --file dataset/queries/test_sql.sql
```

### 3. 交互式模式

```bash
python test_plan_analyzer.py --interactive
```

在交互式模式下，你可以：
- 输入 SQL 查询（多行，空行结束）
- 输入 `exit` 或 `quit` 退出
- 使用 `Ctrl+C` 退出

### 4. 详细输出模式

添加 `--verbose` 或 `-v` 参数可以显示：
- 完整的文本格式报告
- 完整的 JSON 输出

```bash
python test_plan_analyzer.py --sql "SELECT ..." --verbose
```

## 输出格式

测试工具会输出以下信息：

### 1. 性能瓶颈列表
- 节点类型（如 Seq Scan、Hash Join 等）
- 对象表名
- 代价占比
- 自身消耗和累积消耗
- 是否为流水线阻塞节点
- 上下文信息

### 2. 成本分析
- 总代价
- 最大树代价
- 瓶颈数量和总代价
- 瓶颈代价占比

### 3. 优化建议
- 基于瓶颈类型自动生成的优化建议
- 针对扫描、连接、聚合等不同类型的建议

### 4. 详细报告（verbose 模式）
- 完整的文本格式分析报告
- 完整的 JSON 输出

## 示例输出

```
================================================================================
  🔍 智能计划分析器测试
================================================================================

  📝 SQL 查询:
  SELECT o_orderpriority, count(*) as order_count FROM orders WHERE o_orderdate >= date '1997-01-01'...

  ✅ DBMS 连接初始化成功
  ✅ 数据库连接成功

  🔄 正在分析执行计划...
  ✅ 分析完成

  📊 发现 2 个性能瓶颈:

  [1] Seq Scan
      • 对象: orders
      • 代价占比: 45.23%
      • 自身消耗: 1,234.56
      • 累积消耗: 2,345.67
      • 上下文: Filter: (o_orderdate >= '1997-01-01'::date)

  [2] Hash Join
      • 对象: N/A
      • 代价占比: 32.15%
      • 自身消耗: 876.54
      • 累积消耗: 1,234.56

  💰 成本分析:
      • 总代价: 2,345.67
      • 最大树代价: 2,345.67
      • 瓶颈数量: 2
      • 瓶颈总代价: 2,111.10
      • 瓶颈代价占比: 90.01%

  💡 优化建议 (2 条):

      [1] 发现 1 个全表扫描（Seq Scan）瓶颈，建议：1) 添加适当的索引；2) 优化 WHERE 条件以利用索引；3) 考虑使用覆盖索引。
      [2] 发现 1 个哈希连接瓶颈，建议检查连接表的大小和内存配置。
```

## 前置条件

1. **数据库配置**
   - 确保 `.env` 文件中配置了正确的数据库连接信息：
     ```
     DB_HOST=localhost
     DB_PORT=5432
     DB_NAME=your_database
     DB_USER=your_username
     DB_PASSWORD=your_password
     ```

2. **数据库连接**
   - 确保数据库服务正在运行
   - 确保数据库连接信息正确
   - 确保有执行 EXPLAIN 的权限

3. **依赖项**
   - Python 3.7+
   - psycopg2
   - 项目依赖已安装

## 故障排除

### 1. 数据库连接失败

```
❌ 数据库连接失败: connection refused
```

**解决方案：**
- 检查数据库服务是否运行
- 检查 `.env` 文件中的连接信息
- 检查网络连接和防火墙设置

### 2. PlanAnalyzer 导入失败

```
Warning: Failed to import PlanAnalyzer
Falling back to basic EXPLAIN...
```

**解决方案：**
- 确保 `src/Rewrite_Middleware/plan_analyzer.py` 文件存在
- 检查 Python 路径配置

### 3. SQL 语法错误

```
❌ 错误: syntax error at or near "..."
```

**解决方案：**
- 检查 SQL 语法是否正确
- 确保 SQL 语句完整
- 检查数据库表是否存在

## 测试示例 SQL

### TPC-H 查询示例

```sql
-- Q1: 价格统计查询
SELECT l_returnflag, l_linestatus, 
       sum(l_quantity) as sum_qty,
       sum(l_extendedprice) as sum_base_price
FROM lineitem
WHERE l_shipdate <= date '1998-12-01' - interval '90 day'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;

-- Q4: 订单优先级统计
SELECT o_orderpriority, count(*) as order_count
FROM orders
WHERE o_orderdate >= date '1997-01-01'
  AND o_orderdate < date '1997-01-01' + interval '3' month
  AND exists (
    SELECT * FROM lineitem
    WHERE l_orderkey = o_orderkey
      AND l_commitdate < l_receiptdate
  )
GROUP BY o_orderpriority
ORDER BY o_orderpriority;
```

## 与系统集成

测试工具的输出格式与系统实际使用的格式完全一致，可以直接用于：

1. **验证分析器功能**：确保分析器正常工作
2. **调试优化问题**：查看详细的瓶颈分析
3. **性能评估**：评估 SQL 查询的性能特征
4. **规则选择验证**：验证瓶颈识别是否正确

## 相关文件

- `src/Rewrite_Middleware/middleware.py` - 智能计划分析器实现
- `src/Rewrite_Middleware/plan_analyzer.py` - PlanAnalyzer 核心逻辑
- `test_module.py` - 完整的系统测试套件

