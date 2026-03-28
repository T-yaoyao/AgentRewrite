import sys
import os
import json
import re
from pathlib import Path

# Setup project paths - 需要在导入其他模块之前设置
# 获取当前文件的目录
current_file = Path(__file__).resolve()
# 获取项目根目录（从 src/utils 向上两级）
project_root = current_file.parent.parent.parent
# 将项目根目录添加到 Python 路径
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 现在可以导入项目模块
from src.utils.path_config import setup_python_path
setup_python_path()

from src.Rewrite_Middleware.middleware import DBMS
from src.utils.data_distribution import DATABASE_STATISTICS


def get_index_info(dbms_instance):
    """
    PostgreSQL public schema indexes via pg_indexes.
    Returns: [{"table": str, "index_name": str, "definition": str}, ...]
    """
    if dbms_instance is None:
        return []
    out = []
    try:
        dbms_instance.connect()
        dbms_instance.cursor.execute(
            """
            SELECT tablename, indexname, indexdef
            FROM pg_indexes
            WHERE schemaname = 'public'
            ORDER BY tablename, indexname
            """
        )
        for row in dbms_instance.cursor.fetchall():
            r = row if isinstance(row, (list, tuple)) else (
                row.get("tablename"),
                row.get("indexname"),
                row.get("indexdef"),
            )
            out.append({"table": r[0], "index_name": r[1], "definition": r[2] or ""})
    except Exception as e:
        print(f"⚠️ 获取索引信息失败: {e}")
        out = []
    return out


def format_index_info_for_prompt(index_list):
    """Human-readable index block for agent prompts."""
    if not index_list:
        return "当前库未查询到索引信息。"
    lines = []
    cur = None
    for item in index_list:
        t = item.get("table", "")
        if t != cur:
            cur = t
            lines.append(f"表名: {t}")
        lines.append(f"  索引名: {item.get('index_name', '')}")
        d = item.get("definition", "")
        if d:
            lines.append(f"  定义: {d}")
    return "\n".join(lines)


def collect_database_statistics(dbms_instance=None):
    """
    Collect statistics from the database
    
    Args:
        dbms_instance: DBMS instance, if None, a new instance will be created

    Returns:
        tuple: (db_name, statistics_list)
    """
    if dbms_instance is None:
        dbms_instance = DBMS()
    
    print("="*60)
    print("🗄️  Starting Database Statistics Collection")
    print("="*60)
    print(f"🔗 Database: {dbms_instance.db_name}")
    
    dbms_instance.connect()
    result = []
    
    try:
        print("📊 Getting list of tables in public schema...")
        table_names_sql = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name
        """
        dbms_instance.cursor.execute(table_names_sql)
        tables = dbms_instance.cursor.fetchall()
        
        print(f"✅ Found {len(tables)} tables. Starting table scan...")
        print("-" * 60)
        
        for i, table_row in enumerate(tables):
            table_name = table_row['table_name'] if isinstance(table_row, dict) else table_row[0]
            print(f"[{i+1:2d}/{len(tables)}] Scanning '{table_name}'...", end=" ")
            
            try:
                count_sql = f"SELECT COUNT(*) FROM {table_name}"
                dbms_instance.cursor.execute(count_sql)
                count_result = dbms_instance.cursor.fetchone()
                row_count = count_result['count'] if isinstance(count_result, dict) else count_result[0]
                
                result.append([table_name, str(row_count)])
                print(f"✅ {row_count:,} rows")
                
            except Exception as e:
                print(f"❌ Error: {e}")
                result.append([table_name, "0"])
        
        print("-" * 60)
        print(f"✅ Statistics collection completed for {len(result)} tables")

        # Show summary information
        total_rows = sum(int(count) for _, count in result)
        print(f"📊 Total rows across all tables: {total_rows:,}")
        
        return dbms_instance.db_name, result
        
    except Exception as e:
        print(f"❌ Error during collection: {e}")
        return dbms_instance.db_name, []
    finally:
        dbms_instance.close()


def update_data_distribution_file(db_name, statistics):
    """
    Update data_distribution.py file

    Args:
        db_name: Database name
        statistics: List of statistics data
    """
    try:        
        print(f"📁 Updating DATABASE_STATISTICS for: {db_name}")

        # Step 1: Update the dictionary
        DATABASE_STATISTICS[db_name] = statistics
        print(f"✅ Updated in-memory DATABASE_STATISTICS for '{db_name}'")

        # Step 2: Get the path to the data_distribution.py file
        distribution_file = Path(__file__).parent / "data_distribution.py"

        # Step 3: Read the existing file content
        with open(distribution_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Step 4: Generate new dictionary content
        dict_content = "{\n"
        for key, value in DATABASE_STATISTICS.items():
            # Format statistics data
            formatted_stats = json.dumps(value, ensure_ascii=False)
            dict_content += f'    "{key}": {formatted_stats},\n    \n'
        dict_content = dict_content.rstrip(',\n    \n') + '\n}'

        # Step 5: Replace DATABASE_STATISTICS dictionary content
        pattern = r'DATABASE_STATISTICS = \{[^}]*\}'
        replacement = f'DATABASE_STATISTICS = {dict_content}'
        
        new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)

        # Step 6: Write back to the file
        with open(distribution_file, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ Successfully updated {distribution_file}")
        print(f"📋 Database '{db_name}' now has {len(statistics)} tables")
        return True
        
    except Exception as e:
        print(f"❌ Error updating distribution file: {e}")
        return False


def get_data_statistics(db_name: str = None):
    """
    Main function: Collect database statistics and update distribution file

    Args:
        db_name: Optional database name. If None, uses DB_NAME from environment or DBMS instance.

    Returns:
        str: JSON formatted statistics
    """
    # Create DBMS instance
    dbms = DBMS()
    
    # If db_name is provided, override the db_name in DBMS instance
    if db_name:
        dbms.db_name = db_name
        print(f"📌 Using specified database: {db_name}")
    else:
        print(f"📌 Using database from environment: {dbms.db_name}")
    
    # Collect statistics from the database
    db_name, statistics = collect_database_statistics(dbms)
    
    if not statistics:
        print("⚠️  No statistics collected")
        return "[]"
    
    # Update the data_distribution.py file with collected statistics
    success = update_data_distribution_file(db_name, statistics)
    
    if success:
        print("="*60)
        print("🎉 Statistics Collection and Update Complete!")
        print("="*60)
        print(f"Database: {db_name}")
        print(f"📊 Tables: {len(statistics)}")
        print(f"📄 Updated format:")
        print(f"    \"{db_name}\": {json.dumps(statistics)}")
        print("="*60)

    # Return JSON formatted statistics
    return json.dumps(statistics)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Collect database statistics and update data_distribution.py',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 更新当前环境变量指定的数据库
  python src/utils/get_data_statistics.py
  
  # 更新指定数据库
  python src/utils/get_data_statistics.py --db calcite
  python src/utils/get_data_statistics.py --db dsb
  python src/utils/get_data_statistics.py --db tpch
  
  # 更新多个数据库
  python src/utils/get_data_statistics.py --db calcite --db dsb
        """
    )
    
    parser.add_argument(
        '--db',
        '--database',
        dest='db_names',
        action='append',
        help='Database name(s) to update. Can be specified multiple times. If not specified, uses DB_NAME from environment.'
    )
    
    args = parser.parse_args()
    
    if args.db_names:
        # Update multiple databases
        for db_name in args.db_names:
            print("\n" + "="*80)
            print(f"🔄 Updating statistics for database: {db_name}")
            print("="*80)
            try:
                data_statistics = get_data_statistics(db_name)
                print(f"\n✅ Successfully updated {db_name}")
            except Exception as e:
                print(f"\n❌ Failed to update {db_name}: {e}")
                import traceback
                traceback.print_exc()
    else:
        # Update current database from environment
        data_statistics = get_data_statistics()
        print(f"\n🎯 Final JSON Output:")
        print(data_statistics)


