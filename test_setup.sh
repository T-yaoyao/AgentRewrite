#!/bin/bash

# 测试脚本：验证环境和配置

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== 环境测试脚本 ===${NC}"

# 1. 检查Java
echo -e "${YELLOW}1. 检查Java环境...${NC}"
if command -v java &> /dev/null && command -v javac &> /dev/null; then
    echo -e "${GREEN}✓ Java已安装${NC}"
    java -version
else
    echo -e "${RED}✗ Java未安装或未在PATH中${NC}"
    exit 1
fi

# 2. 检查JSON文件
echo -e "${YELLOW}2. 检查JSON文件...${NC}"
JSON_FILE="experiments_results/tpch/QUITE_tpch_63queries_rewritten_only.json"
if [ -f "$JSON_FILE" ]; then
    echo -e "${GREEN}✓ JSON文件存在${NC}"
    QUERY_COUNT=$(python3 -c "import json; data=json.load(open('$JSON_FILE')); print(len(data))" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  包含 $QUERY_COUNT 条查询${NC}"
    else
        echo -e "${RED}  ✗ JSON文件格式可能有问题${NC}"
    fi
else
    echo -e "${RED}✗ JSON文件不存在: $JSON_FILE${NC}"
    exit 1
fi

# 3. 检查数据库连接（如果PostgreSQL客户端可用）
echo -e "${YELLOW}3. 检查数据库配置...${NC}"
if command -v psql &> /dev/null; then
    DB_HOST=${DB_HOST:-localhost}
    DB_PORT=${DB_PORT:-5432}
    DB_NAME=${DB_NAME:-tpch}
    DB_USER=${DB_USER:-postgres}
    
    echo "  尝试连接: $DB_USER@$DB_HOST:$DB_PORT/$DB_NAME"
    PGPASSWORD=${DB_PASSWORD} psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1;" > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 数据库连接成功${NC}"
    else
        echo -e "${YELLOW}⚠ 数据库连接失败（可能需要在运行时配置）${NC}"
    fi
else
    echo -e "${YELLOW}⚠ psql未安装，跳过数据库连接测试${NC}"
fi

# 4. 检查依赖库
echo -e "${YELLOW}4. 检查依赖库...${NC}"
if [ -f "gson-2.10.1.jar" ]; then
    echo -e "${GREEN}✓ Gson库存在${NC}"
else
    echo -e "${YELLOW}⚠ Gson库不存在（将在编译时自动下载）${NC}"
fi

if [ -f "postgresql-42.7.1.jar" ]; then
    echo -e "${GREEN}✓ PostgreSQL JDBC驱动存在${NC}"
else
    echo -e "${YELLOW}⚠ PostgreSQL JDBC驱动不存在（将在编译时自动下载）${NC}"
fi

echo ""
echo -e "${GREEN}=== 测试完成 ===${NC}"
echo ""
echo "如果所有检查都通过，可以运行: ./compile_and_run.sh"




