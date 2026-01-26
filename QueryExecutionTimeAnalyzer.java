import java.io.*;
import java.sql.*;
import java.util.*;
import java.lang.reflect.Type;
import com.google.gson.*;
import com.google.gson.reflect.TypeToken;

/**
 * QueryExecutionTimeAnalyzer
 * 
 * 读取JSON文件中的SQL查询，对每条查询执行4次EXPLAIN ANALYZE，
 * 忽略第1次冷启动，取第2-4次的平均执行时间作为最终结果，输出到CSV文件。
 */
public class QueryExecutionTimeAnalyzer {
    
    private static final String JSON_FILE = "dataset/queries/calcite_queries.json";
    private static final String CSV_OUTPUT = "dataset/queries/original_calcite_queries.csv";
    
    // 数据库连接配置（从环境变量或配置文件读取）
    private String dbHost;
    private String dbPort;
    private String dbName;
    private String dbUser;
    private String dbPassword;
    
    public QueryExecutionTimeAnalyzer() {
        // 从环境变量读取数据库配置
        dbHost = System.getenv("DB_HOST");
        if (dbHost == null) dbHost = "localhost";
        
        dbPort = System.getenv("DB_PORT");
        if (dbPort == null) dbPort = "5432";
        
        dbName = System.getenv("DB_NAME");
        if (dbName == null) dbName = "calcite";
        
        dbUser = System.getenv("DB_USER");
        if (dbUser == null) dbUser = "postgres";
        
        dbPassword = System.getenv("DB_PASSWORD");
        if (dbPassword == null) dbPassword = "";
    }
    
    /**
     * 读取JSON文件并解析查询列表
     */
    private List<QueryItem> readQueriesFromJson(String jsonFile) throws IOException {
        Gson gson = new Gson();
        FileReader reader = new FileReader(jsonFile);
        
        Type listType = new TypeToken<List<QueryItem>>(){}.getType();
        List<QueryItem> queries = gson.fromJson(reader, listType);
        
        reader.close();
        return queries;
    }
    
    /**
     * 执行EXPLAIN ANALYZE并提取执行时间
     * @param conn 数据库连接
     * @param sql SQL查询
     * @return 执行时间（毫秒），如果解析失败返回-1，如果超时返回300000（300秒）
     */
    private double executeExplainAnalyze(Connection conn, String sql) throws SQLException {
        String explainSql = "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql;
        
        try (Statement stmt = conn.createStatement()) {
            // 设置查询超时时间为300秒
            stmt.setQueryTimeout(300);
            
            try (ResultSet rs = stmt.executeQuery(explainSql)) {
                if (rs.next()) {
                    // PostgreSQL的EXPLAIN ANALYZE返回JSON格式
                    String jsonResult = rs.getString(1);
                    
                    // 解析JSON获取执行时间
                    return parseExecutionTime(jsonResult);
                }
            }
        } catch (SQLTimeoutException e) {
            // 查询超时，记录为300秒（300000毫秒）
            System.err.println("查询执行超时（超过300秒），记录为300秒");
            return 300000.0;
        } catch (Exception e) {
            System.err.println("执行EXPLAIN ANALYZE时出错: " + e.getMessage());
            e.printStackTrace();
            return -1;
        }
        
        return -1;
    }
    
    /**
     * 从EXPLAIN ANALYZE的JSON结果中解析执行时间
     * PostgreSQL的EXPLAIN ANALYZE FORMAT JSON返回格式：
     * [{"Plan": {"Execution Time": 26.649, ...}}]
     */
    private double parseExecutionTime(String jsonResult) {
        try {
            Gson gson = new Gson();
            JsonArray jsonArray = gson.fromJson(jsonResult, JsonArray.class);
            
            if (jsonArray.size() > 0) {
                JsonObject root = jsonArray.get(0).getAsJsonObject();
                
                // PostgreSQL的EXPLAIN ANALYZE返回格式中，Execution Time在Plan对象内
                if (root.has("Plan")) {
                    JsonObject planObj = root.getAsJsonObject("Plan");
                    if (planObj.has("Execution Time")) {
                        return planObj.get("Execution Time").getAsDouble();
                    }
                }
                
                // 如果直接在根对象中（某些情况下）
                if (root.has("Execution Time")) {
                    return root.get("Execution Time").getAsDouble();
                }
            }
        } catch (Exception e) {
            System.err.println("解析执行时间时出错: " + e.getMessage());
            System.err.println("JSON内容: " + jsonResult);
            e.printStackTrace();
        }
        
        return -1;
    }
    
    /**
     * 对单条查询执行4次，忽略第1次冷启动，返回第2-4次的平均执行时间
     */
    private double getExecutionTime(Connection conn, String sql, String queryId) {
        System.out.println("处理查询 ID: " + queryId);
        
        // 执行4次，忽略第1次冷启动，计算第2-4次的平均时间
        List<Double> executionTimes = new ArrayList<>();
        
        for (int i = 1; i <= 4; i++) {
            try {
                if (i == 1) {
                    System.out.println("  第 " + i + " 次执行（冷启动，忽略结果）...");
                } else {
                    System.out.println("  第 " + i + " 次执行...");
                }
                
                double time = executeExplainAnalyze(conn, sql);
                
                if (time > 0) {
                    if (i == 1) {
                        System.out.println("  执行时间: " + time + " ms (冷启动，已忽略)");
                    } else {
                        executionTimes.add(time);
                        System.out.println("  执行时间: " + time + " ms");
                    }
                } else {
                    System.err.println("  警告: 无法获取执行时间");
                }
                
                // 短暂延迟，避免数据库压力过大
                Thread.sleep(100);
                
            } catch (Exception e) {
                System.err.println("  执行出错: " + e.getMessage());
                e.printStackTrace();
            }
        }
        
        // 计算第2-4次的平均时间
        if (executionTimes.size() == 3) {
            double sum = 0.0;
            for (Double time : executionTimes) {
                sum += time;
            }
            double average = sum / executionTimes.size();
            System.out.println("  平均执行时间（第2-4次）: " + String.format("%.6f", average) + " ms");
            return average;
        } else if (executionTimes.size() > 0) {
            // 如果只有部分成功，计算可用结果的平均值
            double sum = 0.0;
            for (Double time : executionTimes) {
                sum += time;
            }
            double average = sum / executionTimes.size();
            System.out.println("  平均执行时间（部分结果）: " + String.format("%.6f", average) + " ms");
            return average;
        } else {
            System.err.println("  错误: 无法获取任何有效的执行时间");
            return -1;
        }
    }
    
    /**
     * 处理所有查询并生成CSV文件
     */
    public void processQueries() {
        Connection conn = null;
        
        try {
            // 读取查询列表
            System.out.println("读取JSON文件: " + JSON_FILE);
            List<QueryItem> queries = readQueriesFromJson(JSON_FILE);
            System.out.println("找到 " + queries.size() + " 条查询");
            
            // 连接数据库
            String jdbcUrl = String.format("jdbc:postgresql://%s:%s/%s", dbHost, dbPort, dbName);
            System.out.println("连接数据库: " + jdbcUrl);
            conn = DriverManager.getConnection(jdbcUrl, dbUser, dbPassword);
            System.out.println("数据库连接成功");
            
            // 准备CSV输出
            FileWriter csvWriter = new FileWriter(CSV_OUTPUT);
            csvWriter.write("id,execution_time_ms\n");
            
            // 处理每条查询
            int successCount = 0;
            int failCount = 0;
            
            for (QueryItem query : queries) {
                String sqlQuery = query.getSqlQuery();
                if (sqlQuery == null) {
                    System.err.println("查询 ID " + query.id + " 没有有效的SQL查询（rewritten_query和query字段都为空）");
                    csvWriter.write(String.format("%s,-1\n", query.id));
                    failCount++;
                    continue;
                }
                double executionTime = getExecutionTime(conn, sqlQuery, query.id);
                
                if (executionTime > 0) {
                    csvWriter.write(String.format("%s,%.6f\n", query.id, executionTime));
                    successCount++;
                    System.out.println("查询 ID " + query.id + " 完成，平均执行时间: " + String.format("%.6f", executionTime) + " ms");
                } else {
                    csvWriter.write(String.format("%s,-1\n", query.id));
                    failCount++;
                    System.err.println("查询 ID " + query.id + " 执行失败");
                }
                
                csvWriter.flush();
            }
            
            csvWriter.close();
            
            System.out.println("\n处理完成！");
            System.out.println("成功: " + successCount + " 条");
            System.out.println("失败: " + failCount + " 条");
            System.out.println("结果已保存到: " + CSV_OUTPUT);
            
        } catch (Exception e) {
            System.err.println("处理过程中出错: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (conn != null) {
                try {
                    conn.close();
                } catch (SQLException e) {
                    e.printStackTrace();
                }
            }
        }
    }
    
    /**
     * 主函数
     */
    public static void main(String[] args) {
        QueryExecutionTimeAnalyzer analyzer = new QueryExecutionTimeAnalyzer();
        analyzer.processQueries();
    }
    
    /**
     * JSON查询项的数据类
     * 兼容两种JSON格式：
     * 1. rewritten_query 字段（QUITE_tpch_63queries_rewritten_only.json）
     * 2. query 字段（tpch_queries.json）
     */
    static class QueryItem {
        String id;
        String rewritten_query;
        String query;  // 兼容原始查询格式
        
        /**
         * 获取SQL查询，优先使用rewritten_query，如果没有则使用query
         */
        String getSqlQuery() {
            if (rewritten_query != null && !rewritten_query.trim().isEmpty()) {
                return rewritten_query;
            } else if (query != null && !query.trim().isEmpty()) {
                return query;
            } else {
                return null;
            }
        }
    }
}

