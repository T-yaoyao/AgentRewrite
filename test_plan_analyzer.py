#!/usr/bin/env python3
"""
独立测试智能计划分析器 (Intelligent Plan Analyzer)

用法:
    # 从命令行参数测试
    python test_plan_analyzer.py --sql "SELECT * FROM orders WHERE o_orderdate >= '1997-01-01'"
    
    # 从文件读取 SQL
    python test_plan_analyzer.py --file dataset/queries/test_sql.sql
    
    # 交互式模式
    python test_plan_analyzer.py --interactive
    
    # 显示详细输出
    python test_plan_analyzer.py --sql "SELECT ..." --verbose
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path

# Setup project paths
from src.utils.path_config import PROJECT_ROOT, setup_python_path, load_project_env
setup_python_path()
load_project_env()

from src.Rewrite_Middleware.middleware import DBMS, DBMS_EXPLAIN_Tool


def print_section(title: str, char: str = "="):
    """打印分节标题"""
    print("\n" + char * 80)
    print(f"  {title}")
    print(char * 80)


def print_bottlenecks(bottlenecks: list):
    """格式化打印瓶颈信息"""
    if not bottlenecks:
        print("  ✅ 未发现显著性能瓶颈")
        return
    
    print(f"\n  📊 发现 {len(bottlenecks)} 个性能瓶颈:\n")
    for i, bn in enumerate(bottlenecks, 1):
        print(f"  [{i}] {bn.get('node_type', 'Unknown')}")
        print(f"      • 对象: {bn.get('entity', 'N/A')}")
        print(f"      • 代价占比: {bn.get('cost_percentage', 0):.2f}%")
        print(f"      • 自身消耗: {bn.get('self_cost', 0):,.2f}")
        print(f"      • 累积消耗: {bn.get('total_cost', 0):,.2f}")
        if bn.get('is_blocker', False):
            print(f"      • 🛑 流水线阻塞节点")
        if bn.get('context'):
            print(f"      • 上下文: {bn.get('context', '')}")
        print()


def print_cost_analysis(cost_analysis: dict):
    """格式化打印成本分析"""
    print("\n  💰 成本分析:")
    print(f"      • 总代价: {cost_analysis.get('total_cost', 0):,.2f}")
    print(f"      • 最大树代价: {cost_analysis.get('max_tree_cost', 0):,.2f}")
    if cost_analysis.get('has_limit'):
        print(f"      • 包含 LIMIT 节点")
        if cost_analysis.get('limit_child_cost'):
            print(f"      • LIMIT 子节点代价: {cost_analysis.get('limit_child_cost'):,.2f}")
    print(f"      • 瓶颈数量: {cost_analysis.get('bottleneck_count', 0)}")
    print(f"      • 瓶颈总代价: {cost_analysis.get('total_bottleneck_cost', 0):,.2f}")
    print(f"      • 瓶颈代价占比: {cost_analysis.get('bottleneck_cost_percentage', 0):.2f}%")


def print_recommendations(recommendations: list):
    """格式化打印优化建议"""
    if not recommendations:
        print("\n  💡 暂无优化建议")
        return
    
    print(f"\n  💡 优化建议 ({len(recommendations)} 条):\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"      [{i}] {rec}")


def print_text_report(text_report: str):
    """打印文本格式的报告"""
    if text_report:
        print("\n  📄 详细文本报告:")
        print("  " + "-" * 76)
        for line in text_report.split('\n'):
            print(f"  {line}")
        print("  " + "-" * 76)


def format_output(result_str: str, verbose: bool = False):
    """格式化输出分析结果"""
    # Since DBMS_EXPLAIN_Tool now returns text directly, just print it
    print("📊 执行计划分析结果:")
    print("=" * 60)
    print(result_str)
    print("=" * 60)


def format_single_analysis(analysis: dict, verbose: bool = False):
    """格式化单个分析结果"""
    # 检查是否有错误
    if "error" in analysis:
        print(f"  ❌ 错误: {analysis['error']}")
        if "raw_plan" in analysis:
            print(f"\n  原始执行计划 (前500字符):\n  {analysis['raw_plan'][:500]}")
        return
    
    # 打印瓶颈
    bottlenecks = analysis.get('bottlenecks', [])
    print_bottlenecks(bottlenecks)
    
    # 打印成本分析
    cost_analysis = analysis.get('cost_analysis', {})
    print_cost_analysis(cost_analysis)
    
    # 打印建议
    recommendations = analysis.get('recommendations', [])
    print_recommendations(recommendations)
    
    # 详细模式下打印文本报告
    if verbose:
        text_report = analysis.get('text_report', '')
        print_text_report(text_report)
    
    # 打印 JSON 格式（详细模式）
    if verbose:
        print_section("完整 JSON 输出", "-")
        print(json.dumps(analysis, ensure_ascii=False, indent=2))


async def test_sql(sql: str, verbose: bool = False):
    """测试单个 SQL 查询"""
    print_section("🔍 智能计划分析器测试")
    
    print(f"\n  📝 SQL 查询:")
    print(f"  {sql[:200]}{'...' if len(sql) > 200 else ''}\n")
    
    # 初始化 DBMS
    try:
        dbms = DBMS()
        print("  ✅ DBMS 连接初始化成功")
    except Exception as e:
        print(f"  ❌ DBMS 初始化失败: {e}")
        return
    
    # 测试连接
    try:
        dbms.connect()
        print("  ✅ 数据库连接成功")
        dbms.close()
    except Exception as e:
        print(f"  ❌ 数据库连接失败: {e}")
        print(f"     请检查 .env 文件中的数据库配置")
        return
    
    # 执行分析
    print("\n  🔄 正在分析执行计划...")
    try:
        result_json = await DBMS_EXPLAIN_Tool(dbms, sql)
        print("  ✅ 分析完成\n")
        
        # 格式化输出
        format_output(result_json, verbose)
        
    except Exception as e:
        print(f"  ❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()


async def test_from_file(file_path: str, verbose: bool = False):
    """从文件读取 SQL 并测试"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"❌ 文件不存在: {file_path}")
        return
    
    print_section(f"📂 从文件读取: {file_path.name}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            sql = f.read().strip()
        
        if not sql:
            print("  ❌ 文件为空")
            return
        
        await test_sql(sql, verbose)
        
    except Exception as e:
        print(f"  ❌ 读取文件失败: {e}")


async def interactive_mode(verbose: bool = False):
    """交互式模式"""
    print_section("💬 交互式测试模式")
    print("\n  输入 SQL 查询（输入空行或 'exit' 退出）\n")
    
    # 初始化 DBMS
    try:
        dbms = DBMS()
        dbms.connect()
        dbms.close()
        print("  ✅ 数据库连接就绪\n")
    except Exception as e:
        print(f"  ❌ 数据库连接失败: {e}")
        return
    
    while True:
        try:
            print("  " + "-" * 76)
            print("  SQL> ", end="", flush=True)
            
            sql_lines = []
            while True:
                try:
                    line = input()
                    if not line.strip():
                        break
                    if line.strip().lower() in ['exit', 'quit', 'q']:
                        print("\n  👋 退出交互模式")
                        return
                    sql_lines.append(line)
                except EOFError:
                    print("\n  👋 退出交互模式")
                    return
            
            sql = '\n'.join(sql_lines).strip()
            
            if not sql:
                continue
            
            await test_sql(sql, verbose)
            print()
            
        except KeyboardInterrupt:
            print("\n\n  👋 退出交互模式")
            break
        except Exception as e:
            print(f"\n  ❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='测试智能计划分析器 (Intelligent Plan Analyzer)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 测试单个 SQL
  python test_plan_analyzer.py --sql "SELECT * FROM orders WHERE o_orderdate >= '1997-01-01'"
  
  # 从文件读取
  python test_plan_analyzer.py --file dataset/queries/test_sql.sql
  
  # 交互式模式
  python test_plan_analyzer.py --interactive
  
  # 详细输出
  python test_plan_analyzer.py --sql "SELECT ..." --verbose
        """
    )
    
    parser.add_argument(
        '--sql',
        type=str,
        help='要分析的 SQL 查询语句'
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='包含 SQL 查询的文件路径'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='进入交互式模式'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='显示详细输出（包括文本报告和完整 JSON）'
    )
    
    args = parser.parse_args()
    
    # 检查参数
    if not any([args.sql, args.file, args.interactive]):
        parser.print_help()
        print("\n❌ 请提供 --sql、--file 或 --interactive 参数之一")
        sys.exit(1)
    
    # 执行测试
    if args.interactive:
        asyncio.run(interactive_mode(args.verbose))
    elif args.file:
        asyncio.run(test_from_file(args.file, args.verbose))
    elif args.sql:
        asyncio.run(test_sql(args.sql, args.verbose))


if __name__ == "__main__":
    main()

