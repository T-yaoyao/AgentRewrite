# QueryExecutionTimeAnalyzer 使用说明

## 功能描述

`QueryExecutionTimeAnalyzer.java` 是一个Java程序，用于：
1. 读取 `experiments_results/tpch/QUITE_tpch_63queries_rewritten_only.json` 文件中的所有SQL查询
2. 对每条查询执行4次 `EXPLAIN ANALYZE`
3. 忽略第1次冷启动，取第2-4次的平均执行时间作为最终结果（用于消除缓存影响和获得更稳定的结果）
4. 将结果输出到CSV文件：`experiments_results/tpch/query_execution_times.csv`

## 前置要求

1. **Java JDK 8 或更高版本**
   ```bash
   java -version
   javac -version
   ```

2. **PostgreSQL数据库**
   - 确保数据库已启动并可访问
   - 确保数据库中存在相应的表和数据

3. **依赖库**（脚本会自动下载）
   - Gson 2.10.1（用于JSON解析）
   - PostgreSQL JDBC Driver 42.7.1（用于数据库连接）

## 使用方法

### 方法1：使用自动脚本（推荐）

```bash
# 设置数据库连接环境变量（可选，有默认值）
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=tpch
export DB_USER=postgres
export DB_PASSWORD=your_password

# 运行脚本
./compile_and_run.sh
```

### 方法2：手动编译和运行

```bash
# 1. 下载依赖库
wget https://repo1.maven.org/maven2/com/google/code/gson/gson/2.10.1/gson-2.10.1.jar
wget https://jdbc.postgresql.org/download/postgresql-42.7.1.jar

# 2. 编译
javac -cp ".:gson-2.10.1.jar:postgresql-42.7.1.jar" QueryExecutionTimeAnalyzer.java

# 3. 运行
java -cp ".:gson-2.10.1.jar:postgresql-42.7.1.jar" QueryExecutionTimeAnalyzer
```

./compile_and_run.sh

## 环境变量配置

程序会从环境变量读取数据库配置，如果未设置则使用默认值：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| DB_HOST | localhost | 数据库主机地址 |
| DB_PORT | 5432 | 数据库端口 |
| DB_NAME | tpch | 数据库名称 |
| DB_USER | postgres | 数据库用户名 |
| DB_PASSWORD | (空) | 数据库密码 |

## 输出文件

程序会在 `experiments_results/tpch/query_execution_times.csv` 生成结果文件。

CSV格式：
```csv
id,execution_time_ms
1,26.649828
2,27.632657
3,27.038805
...
```

- `id`: 查询ID（字符串）
- `execution_time_ms`: 第2-4次执行的平均时间（毫秒，保留6位小数）

如果某条查询执行失败，`execution_time_ms` 会显示为 `-1`。

## 执行流程

1. 读取JSON文件，解析所有查询
2. 连接PostgreSQL数据库
3. 对每条查询：
   - 执行第1次 `EXPLAIN ANALYZE`（冷启动，忽略结果）
   - 执行第2次 `EXPLAIN ANALYZE`（记录结果）
   - 执行第3次 `EXPLAIN ANALYZE`（记录结果）
   - 执行第4次 `EXPLAIN ANALYZE`（记录结果）
   - 计算第2-4次的平均执行时间
   - 每次执行间隔100毫秒
4. 将所有结果写入CSV文件

## 注意事项

1. **执行时间**：63条查询，每条执行4次，总共需要执行252次查询，可能需要较长时间
2. **数据库负载**：程序会在数据库上执行大量查询，请确保数据库有足够的资源
3. **网络连接**：确保Java程序可以访问PostgreSQL数据库
4. **权限**：确保数据库用户有执行 `EXPLAIN ANALYZE` 的权限

## 故障排除

### 问题1：编译错误 - 找不到Gson类
**解决**：确保Gson库已下载并在classpath中
```bash
wget https://repo1.maven.org/maven2/com/google/code/gson/gson/2.10.1/gson-2.10.1.jar
```

### 问题2：编译错误 - 找不到PostgreSQL驱动
**解决**：确保PostgreSQL JDBC驱动已下载并在classpath中
```bash
wget https://jdbc.postgresql.org/download/postgresql-42.7.1.jar
```

### 问题3：运行时错误 - 无法连接数据库
**解决**：
- 检查数据库是否运行：`pg_isready`
- 检查环境变量是否正确设置
- 检查防火墙设置
- 检查数据库用户权限

### 问题4：执行时间解析失败
**解决**：
- 确保PostgreSQL版本支持 `EXPLAIN ANALYZE ... FORMAT JSON`
- 检查SQL查询语法是否正确
- 查看控制台错误信息

## 示例输出

```
读取JSON文件: experiments_results/tpch/QUITE_tpch_63queries_rewritten_only.json
找到 63 条查询
连接数据库: jdbc:postgresql://localhost:5432/tpch
数据库连接成功
处理查询 ID: 1
  第 1 次执行（冷启动，忽略结果）...
  执行时间: 26.870062 ms (冷启动，已忽略)
  第 2 次执行...
  执行时间: 26.649828 ms
  第 3 次执行...
  执行时间: 26.650123 ms
  第 4 次执行...
  执行时间: 26.649900 ms
  平均执行时间（第2-4次）: 26.649950 ms
查询 ID 1 完成，平均执行时间: 26.649950 ms
...
处理完成！
成功: 63 条
失败: 0 条
结果已保存到: experiments_results/tpch/query_execution_times.csv
```

