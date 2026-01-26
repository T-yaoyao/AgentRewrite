#!/bin/bash

# 编译和运行QueryExecutionTimeAnalyzer的脚本

# 设置颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== QueryExecutionTimeAnalyzer 编译和运行脚本 ===${NC}"

# 检查Java是否安装
if ! command -v javac &> /dev/null; then
    echo -e "${RED}错误: 未找到 javac 命令，请先安装 Java JDK${NC}"
    exit 1
fi

if ! command -v java &> /dev/null; then
    echo -e "${RED}错误: 未找到 java 命令，请先安装 Java JDK${NC}"
    exit 1
fi

echo -e "${YELLOW}Java 版本:${NC}"
java -version

# 检查Gson库是否存在
GSON_JAR="gson-2.10.1.jar"
if [ ! -f "$GSON_JAR" ]; then
    echo -e "${YELLOW}未找到 Gson 库，正在下载...${NC}"
    wget -q https://repo1.maven.org/maven2/com/google/code/gson/gson/2.10.1/gson-2.10.1.jar -O $GSON_JAR
    if [ $? -ne 0 ]; then
        echo -e "${RED}下载 Gson 库失败，请手动下载并放在当前目录${NC}"
        exit 1
    fi
    echo -e "${GREEN}Gson 库下载成功${NC}"
fi

# 检查PostgreSQL JDBC驱动是否存在
POSTGRESQL_JAR="postgresql-42.7.1.jar"
if [ ! -f "$POSTGRESQL_JAR" ]; then
    echo -e "${YELLOW}未找到 PostgreSQL JDBC 驱动，正在下载...${NC}"
    wget -q https://jdbc.postgresql.org/download/postgresql-42.7.1.jar -O $POSTGRESQL_JAR
    if [ $? -ne 0 ]; then
        echo -e "${RED}下载 PostgreSQL JDBC 驱动失败，请手动下载并放在当前目录${NC}"
        exit 1
    fi
    echo -e "${GREEN}PostgreSQL JDBC 驱动下载成功${NC}"
fi

# 编译Java文件
echo -e "${YELLOW}正在编译 Java 文件...${NC}"
javac -cp ".:$GSON_JAR:$POSTGRESQL_JAR" QueryExecutionTimeAnalyzer.java

if [ $? -ne 0 ]; then
    echo -e "${RED}编译失败${NC}"
    exit 1
fi

echo -e "${GREEN}编译成功${NC}"

# 检查环境变量
echo -e "${YELLOW}检查数据库配置...${NC}"
if [ -z "$DB_HOST" ]; then
    echo -e "${YELLOW}警告: DB_HOST 未设置，将使用默认值 localhost${NC}"
fi
if [ -z "$DB_PORT" ]; then
    echo -e "${YELLOW}警告: DB_PORT 未设置，将使用默认值 5432${NC}"
fi
if [ -z "$DB_NAME" ]; then
    echo -e "${YELLOW}警告: DB_NAME 未设置，将使用默认值 tpch${NC}"
fi
if [ -z "$DB_USER" ]; then
    echo -e "${YELLOW}警告: DB_USER 未设置，将使用默认值 postgres${NC}"
fi
if [ -z "$DB_PASSWORD" ]; then
    echo -e "${YELLOW}警告: DB_PASSWORD 未设置，将使用空密码${NC}"
fi

# 运行程序
echo -e "${GREEN}开始运行程序...${NC}"
echo ""

java -cp ".:$GSON_JAR:$POSTGRESQL_JAR" QueryExecutionTimeAnalyzer

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}程序执行完成！${NC}"
    echo -e "${GREEN}结果文件: experiments_results/tpch/query_execution_times.csv${NC}"
else
    echo ""
    echo -e "${RED}程序执行失败${NC}"
    exit 1
fi




