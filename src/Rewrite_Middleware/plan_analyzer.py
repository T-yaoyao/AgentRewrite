import json
import os
from typing import Optional

from src.utils.llm_json_utils import loads_with_repair

class PlanAnalyzer:
    def __init__(self, explain_json_str: Optional[str] = None, sql_query: Optional[str] = None, 
                 analyze: bool = False, db_config: Optional[dict] = None, db_name: Optional[str] = None):
        """
        初始化 PlanAnalyzer
        
        Args:
            explain_json_str: 执行计划的 JSON 字符串（可选）
            sql_query: SQL 查询语句（可选，如果提供则从数据库获取执行计划）
            analyze: 是否执行 ANALYZE（实际执行查询），仅在提供 sql_query 时有效
            db_config: 数据库连接配置字典，包含 host, port, database, user, password
            db_name: 数据库名称（用于获取统计信息），如果未提供则从环境变量或 db_config 获取
        """
        self.raw_json = explain_json_str
        self.plan_root = None
        self.max_tree_cost = 0.0  # 全局最大代价
        self.root_total_cost = 0.0 # 根节点代价
        self.has_limit = False  # 是否包含 LIMIT 节点
        self.limit_child_cost = 0.0  # LIMIT 子节点的最大代价（用于计算百分比基准）
        
        # 获取数据库名称用于统计信息查询
        if db_name:
            self.db_name = db_name
        elif db_config and 'database' in db_config:
            self.db_name = db_config['database']
        else:
            self.db_name = os.getenv('DB_NAME', None)
        
        # 加载统计信息字典
        self._load_statistics()
        
        # 如果提供了 SQL 查询，从数据库获取执行计划
        if sql_query:
            explain_json_str = self._get_plan_from_db(sql_query, analyze, db_config)
            if not explain_json_str:
                raise ValueError("无法从数据库获取执行计划")
        
        if not explain_json_str:
            raise ValueError("必须提供 explain_json_str 或 sql_query 之一")
        
        try:
            # 1. JSON 字符串清洗 (保持原本的健壮逻辑)
            cleaned_str = explain_json_str.strip()
            lines = cleaned_str.split('\n')
            cleaned_lines = [line.rstrip('+').rstrip() for line in lines if line.rstrip('+').strip()]
            cleaned_str = '\n'.join(cleaned_lines)
            
            # 2. 解析 JSON
            parsed = loads_with_repair(cleaned_str)
            if isinstance(parsed, list) and len(parsed) > 0 and 'Plan' in parsed[0]:
                self.plan_root = parsed[0]['Plan']
            elif isinstance(parsed, dict) and 'Plan' in parsed:
                self.plan_root = parsed['Plan']
            else:
                self.plan_root = parsed

            # 3. 初始化扫描
            if self.plan_root:
                self.root_total_cost = self.plan_root.get('Total Cost', 0.0)
                # 检测是否存在 LIMIT 节点
                self.has_limit = (self.plan_root.get('Node Type', '') == 'Limit')
                self._scan_global_stats(self.plan_root)
                
                # 如果存在 LIMIT 节点，计算其子节点的最大代价作为基准
                if self.has_limit:
                    children = self.plan_root.get('Plans', [])
                    if children:
                        # 找到 LIMIT 子节点中的最大代价
                        for child in children:
                            child_cost = child.get('Total Cost', 0.0)
                            if child_cost > self.limit_child_cost:
                                self.limit_child_cost = child_cost
                        # 如果子节点代价为0，使用根节点代价作为后备
                        if self.limit_child_cost == 0.0:
                            self.limit_child_cost = self.root_total_cost
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析失败: {e}")
        except Exception as e:
            print(f"❌ 初始化发生未知错误: {e}")
    
    def _get_plan_from_db(self, sql_query: str, analyze: bool = False, 
                          db_config: Optional[dict] = None) -> Optional[str]:
        """
        从数据库获取执行计划的 JSON 字符串
        
        Args:
            sql_query: SQL 查询语句
            analyze: 是否执行 ANALYZE
            db_config: 数据库连接配置
            
        Returns:
            执行计划的 JSON 字符串，失败返回 None
        """
        try:
            # 优先尝试从本地导入（AgentRewrite项目）
            # 尝试多种导入方式以确保兼容性
            PostgresConnector = None
            import os
            import sys
            
            try:
                # 方式1: 从 src.utils 导入（推荐方式）
                # 获取项目根目录（src 的父目录）
                current_file_dir = os.path.dirname(os.path.abspath(__file__))
                # 从 src/Rewrite_Middleware 向上两级到项目根目录
                project_root = os.path.dirname(os.path.dirname(current_file_dir))
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)
                from src.utils.postgres_connector import PostgresConnector
            except ImportError:
                try:
                    # 方式2: 相对导入（同目录，如果文件还在同目录）
                    from .postgres_connector import PostgresConnector
                except ImportError:
                    try:
                        # 方式3: 绝对导入（同目录，向后兼容）
                        current_dir = os.path.dirname(os.path.abspath(__file__))
                        if current_dir not in sys.path:
                            sys.path.insert(0, current_dir)
                        from postgres_connector import PostgresConnector
                    except ImportError:
                        raise ImportError("无法导入 PostgresConnector，请确保 src/utils/postgres_connector.py 文件存在")
            
            if PostgresConnector is None:
                raise ImportError("无法导入 PostgresConnector，请确保已安装相关依赖")
            
            # 获取数据库配置（优先使用传入的配置，其次使用环境变量，最后使用默认值）
            if db_config:
                config = db_config
            else:
                config = {
                    'host': os.getenv('PG_HOST', 'localhost'),
                    'port': int(os.getenv('PG_PORT', '5432')),
                    'database': os.getenv('PG_DATABASE', 'tpch'),
                    'user': os.getenv('PG_USER', 'postgres'),
                    'password': os.getenv('PG_PASSWORD', '123456'),
                }
            
            # 创建数据库连接器
            connector = PostgresConnector(**config)
            
            try:
                # 获取执行计划
                plan_result = connector.execute_explain(sql_query, analyze=analyze, format="JSON")
                
                if 'error' in plan_result:
                    print(f"❌ 获取执行计划失败: {plan_result['error']}")
                    return None
                
                # 提取执行计划的 JSON 字符串
                # execute_explain 返回的格式包含 'raw' 字段，其中包含原始 JSON
                raw_plan = plan_result.get('raw')
                if raw_plan:
                    # 将字典转换为 JSON 字符串
                    return json.dumps([raw_plan], ensure_ascii=False, indent=2)
                else:
                    # 如果没有 raw 字段，尝试从 plan 字段构建
                    plan = plan_result.get('plan', {})
                    if plan:
                        return json.dumps([{'Plan': plan}], ensure_ascii=False, indent=2)
                    else:
                        print("❌ 执行计划结果中未找到有效数据")
                        return None
                        
            finally:
                connector.close()
                
        except ImportError as e:
            print(f"❌ 无法导入 PostgresConnector: {e}")
            print("   请确保 src/utils/postgres_connector.py 文件存在")
            import traceback
            traceback.print_exc()
            return None
        except Exception as e:
            print(f"❌ 从数据库获取执行计划时发生错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _scan_global_stats(self, node):
        """扫描全树，查找最大代价"""
        node_type = node.get('Node Type', '')
        node_cost = node.get('Total Cost', 0.0)
        
        # 如果是 LIMIT 节点，跳过其代价（因为 LIMIT 节点的代价可能不准确）
        # 但如果是 LIMIT 的子节点，需要记录其代价
        if node_type != 'Limit':
            if node_cost > self.max_tree_cost:
                self.max_tree_cost = node_cost

        for child in node.get('Plans', []):
            self._scan_global_stats(child)

    def _extract_table_name(self, node):
        """递归提取节点中的表名"""
        # 直接获取表名
        table_name = node.get('Alias') or node.get('Relation Name')
        if table_name:
            return table_name
        
        # 如果是连接节点或嵌套循环，递归查找子节点中的表名
        children = node.get('Plans', [])
        if children:
            # 对于连接节点，递归查找第一个子节点（通常是左表/外表）
            # 如果第一个子节点也是连接，继续递归
            for child in children:
                child_table = self._extract_table_name(child)
                if child_table:
                    return child_table
        
        return None
    
    def _load_statistics(self):
        """加载数据库统计信息"""
        self.table_stats = {}  # {table_name: row_count}
        
        if not self.db_name:
            return
        
        try:
            from src.utils.data_distribution import get_statistics_list
            statistics = get_statistics_list(self.db_name)
            if statistics:
                # 将统计信息转换为字典格式，便于快速查找
                for table_info in statistics:
                    if isinstance(table_info, list) and len(table_info) >= 2:
                        table_name = table_info[0]
                        row_count = table_info[1]
                        self.table_stats[table_name] = row_count
        except Exception as e:
            # 如果加载失败，静默处理，不影响主要功能
            pass
    
    def _get_table_statistics(self, table_name: str) -> Optional[str]:
        """获取表的统计信息（行数）"""
        if not table_name or table_name == 'N/A':
            return None
        return self.table_stats.get(table_name)
    
    def _format_table_with_stats(self, table_name: str, role: str = None) -> str:
        """格式化表名，包含统计信息"""
        stats = self._get_table_statistics(table_name)
        if stats:
            if role:
                return f"{table_name}: {stats} rows ({role})"
            else:
                return f"{table_name}: {stats} rows"
        else:
            if role:
                return f"{table_name} ({role})"
            else:
                return table_name
    
    def _get_join_tables(self, node):
        """获取连接节点的左右表名"""
        children = node.get('Plans', [])
        if len(children) < 2:
            return None, None
        
        # 左表（第一个子节点）
        left_table = self._extract_table_name(children[0])
        # 右表（第二个子节点）
        right_table = self._extract_table_name(children[1])
        
        return left_table, right_table
    
    def _calculate_metrics(self, node):
        """递归计算节点指标，生成 Effective Cost"""
        node_total = node.get('Total Cost', 0.0)
        node_startup = node.get('Startup Cost', 0.0)
        children = node.get('Plans', [])
        node_type = node.get('Node Type', 'Unknown')
        
        # 对于 LIMIT 节点，特殊处理：跳过其自身代价，只分析子节点
        if node_type == 'Limit':
            # LIMIT 节点本身的代价通常不准确，跳过它
            results = []
            for child in children:
                results.extend(self._calculate_metrics(child))
            return results
        
        # 1. 计算 Self Cost
        children_total_sum = sum(child.get('Total Cost', 0.0) for child in children)
        self_cost = max(0.0, node_total - children_total_sum)
        
        # 2. 计算 Run Cost
        run_cost = max(0.0, node_total - node_startup)

        # 3. 核心算法：Effective Cost (有效代价)
        # 使用真实的 Self Cost
        effective_cost = self_cost

        # 4. 上下文与标记
        # 对于连接节点和嵌套循环，提取两个表名
        entity = node.get('Alias', node.get('Relation Name', 'N/A'))
        
        # 对于单表节点，添加统计信息
        if not any(keyword in node_type for keyword in ['Join', 'Nested Loop']):
            if entity != 'N/A':
                stats = self._get_table_statistics(entity)
                if stats:
                    entity = f"{entity}: {stats} rows"
        
        # 检查是否是连接类型的节点
        is_join_type = any(keyword in node_type for keyword in ['Join', 'Nested Loop'])
        
        if is_join_type and len(children) >= 2:
            # 提取左右表名
            left_table, right_table = self._get_join_tables(node)
            if left_table and right_table:
                # 根据连接类型确定内外表/左右表，并添加统计信息
                if 'Nested Loop' in node_type:
                    # Nested Loop: 第一个子节点是 outer（驱动表），第二个是 inner（被驱动表）
                    left_formatted = self._format_table_with_stats(left_table, "outer")
                    right_formatted = self._format_table_with_stats(right_table, "inner")
                    entity = f"{left_formatted} ⋈ {right_formatted}"
                elif 'Hash Join' in node_type:
                    # Hash Join: 通常第一个是构建表，第二个是探测表
                    left_formatted = self._format_table_with_stats(left_table)
                    right_formatted = self._format_table_with_stats(right_table)
                    entity = f"{left_formatted} ⋈ {right_formatted} (Hash)"
                elif 'Merge Join' in node_type:
                    # Merge Join: 两个已排序的表
                    left_formatted = self._format_table_with_stats(left_table)
                    right_formatted = self._format_table_with_stats(right_table)
                    entity = f"{left_formatted} ⋈ {right_formatted} (Merge)"
                else:
                    # 其他连接类型
                    left_formatted = self._format_table_with_stats(left_table)
                    right_formatted = self._format_table_with_stats(right_table)
                    entity = f"{left_formatted} ⋈ {right_formatted}"
            elif left_table or right_table:
                # 只有一个表名可用
                available_table = left_table or right_table
                available_formatted = self._format_table_with_stats(available_table)
                entity = f"{available_formatted} ⋈ (other)"
        
        context_info = []
        # 对于连接节点，优先显示连接相关的信息
        if is_join_type:
            # 连接类型信息（重要）
            if 'Join Type' in node:
                join_type = node['Join Type']
                context_info.append(f"Join Type: {join_type}")
            
            # 连接条件信息
            if 'Join Filter' in node:
                context_info.append(f"Join Filter: {node['Join Filter']}")
            if 'Hash Cond' in node:
                context_info.append(f"Hash Condition: {node['Hash Cond']}")
            if 'Merge Cond' in node:
                context_info.append(f"Merge Condition: {node['Merge Cond']}")
            
            # Inner Unique 标志（对于某些连接类型很重要）
            if 'Inner Unique' in node:
                inner_unique = node['Inner Unique']
                if inner_unique:
                    context_info.append("Inner Unique: true")
            
            # 其他过滤条件
            if 'Filter' in node:
                context_info.append(f"Filter: {node['Filter']}")
        else:
            # 对于非连接节点，显示过滤和排序信息
            if 'Filter' in node:
                context_info.append(f"Filter: {node['Filter']}")
            if 'Hash Cond' in node:
                context_info.append(f"Hash: {node['Hash Cond']}")
            if 'Sort Key' in node:
                context_info.append(f"Sort: {node['Sort Key']}")
        
        # 对于所有节点，添加其他可能的关键信息
        if 'Index Cond' in node:
            context_info.append(f"Index Condition: {node['Index Cond']}")
        if 'Recheck Cond' in node:
            context_info.append(f"Recheck Condition: {node['Recheck Cond']}")
        if 'Group Key' in node:
            context_info.append(f"Group Key: {node['Group Key']}")
        if 'Sort Key' in node and not is_join_type:
            # 如果之前没有添加 Sort Key，现在添加
            if not any('Sort' in info for info in context_info):
                context_info.append(f"Sort: {node['Sort Key']}")
        
        # 如果连接节点没有其他信息，至少显示连接类型
        if is_join_type and not context_info and 'Join Type' in node:
            context_info.append(f"Join Type: {node['Join Type']}")

        # 阻塞检测 (启动代价占比极高)
        is_blocker = False
        if node_total > 0:
            if (node_startup / node_total > 0.7) and node_startup > 1000:
                is_blocker = True

        current_node = {
            "node_type": node_type,
            "entity": entity,
            "self_cost": self_cost,
            "startup_cost": node_startup,
            "total_cost": node_total,
            "effective_cost": effective_cost, # 排序依据
            "context": "; ".join(context_info),
            "is_blocker": is_blocker
        }
        
        results = [current_node]
        for child in children:
            results.extend(self._calculate_metrics(child))
            
        return results

    def get_top_bottlenecks(self, top_n=5, min_percentage=5):
        """
        获取主要性能瓶颈列表，返回结构化数据而不是格式化字符串
        返回格式: [node1, node2, ...] 其中每个node包含代价信息
        """
        if not self.plan_root:
            return []

        # 分母：用于全量模式 (防止 > 100%)
        # 如果存在 LIMIT 节点，使用 max_tree_cost 作为基准（已排除 LIMIT 节点本身）
        # 否则使用 root_total_cost 和 max_tree_cost 的最大值
        if self.has_limit:
            # LIMIT 节点存在时，max_tree_cost 已经包含了所有子节点的最大代价
            denominator_full = max(self.limit_child_cost, self.max_tree_cost) if self.limit_child_cost > 0 else self.max_tree_cost
        else:
            denominator_full = max(self.root_total_cost, self.max_tree_cost)

        if denominator_full == 0:
            return []

        all_nodes = self._calculate_metrics(self.plan_root)

        # --- 排序策略 ---
        sorted_nodes = sorted(all_nodes, key=lambda x: x['self_cost'], reverse=True)

        # --- 筛选：代价占比大于20%的节点，按顺序输出 ---
        filtered_nodes = []
        seen_startups = set()

        for node in sorted_nodes:
            metric_check = node['self_cost']
            if metric_check <= 1.0: continue

            # 计算显示用的百分比：看自身消耗占全量代价的比例
            display_pct = (node['self_cost'] / denominator_full) * 100
            if display_pct < 1.0: continue

            # 筛选条件：代价占比大于20%
            if display_pct >= 20.0:
                # 添加到结果中
                node['display_pct'] = display_pct
                filtered_nodes.append(node)

        return filtered_nodes

    def format_analysis_report(self, top_n=5, min_percentage=5):
        if not self.plan_root: return "无法生成报告：执行计划解析失败"

        # 分母：用于全量模式 (防止 > 100%)
        # 如果存在 LIMIT 节点，使用 max_tree_cost 作为基准（已排除 LIMIT 节点本身）
        # 否则使用 root_total_cost 和 max_tree_cost 的最大值
        if self.has_limit:
            # LIMIT 节点存在时，max_tree_cost 已经包含了所有子节点的最大代价
            denominator_full = max(self.limit_child_cost, self.max_tree_cost) if self.limit_child_cost > 0 else self.max_tree_cost
        else:
            denominator_full = max(self.root_total_cost, self.max_tree_cost)

        if denominator_full == 0: return "执行计划总代价为0"

        all_nodes = self._calculate_metrics(self.plan_root)

        # --- 排序策略 ---
        sorted_nodes = sorted(all_nodes, key=lambda x: x['self_cost'], reverse=True)

        # --- 筛选：代价占比大于20%的节点，按顺序输出 ---
        filtered_nodes = []
        seen_startups = set()
        
        for node in sorted_nodes:
            metric_check = node['self_cost']
            if metric_check <= 1.0: continue

            # 计算显示用的百分比：看自身消耗占全量代价的比例
            display_pct = (node['self_cost'] / denominator_full) * 100
            if display_pct < 1.0: continue

            # 筛选条件：代价占比大于20%
            if display_pct >= 20.0:
                # 添加到结果中
                node['display_pct'] = display_pct
                filtered_nodes.append(node)
        
        
        if not filtered_nodes:
            mode_info = "（包含 LIMIT）" if self.has_limit else ""
            return f"执行计划总代价: {self.root_total_cost:,.2f}{mode_info}\n未发现显著性能瓶颈。"

        # --- 生成报告 ---
        lines = []
        lines.append(f"执行计划总代价: {self.root_total_cost:,.2f}")
        if self.has_limit:
            lines.append(f"模式检测: 包含 LIMIT（使用子节点代价作为基准: {self.limit_child_cost:,.2f}）")
        else:
            lines.append(f"模式检测: 全量执行")

        lines.append(f"\n发现 {len(filtered_nodes)} 个主要性能瓶颈 (代价占比 > 20%):\n")
        
        for i, node in enumerate(filtered_nodes, 1):
            flags = []

            # 这里的 display_pct 已经是动态计算好的了
            pct_str = f"{node['display_pct']:.1f}%"
            
            # 标题行
            lines.append(f"{i}. 【{node['node_type']}】 {' '.join(flags)}")
            lines.append(f"   - 对象表名: {node['entity']}")
            lines.append(f"   - 代价占比: {pct_str} (基准: {denominator_full:,.0f})")
            lines.append(f"   - 自身消耗: {node['self_cost']:,.2f}")
            lines.append(f"   - 累积消耗: {node['total_cost']:,.2f}")
            
            if node['is_blocker']:
                lines.append(f"   - 🛑 瓶颈类型: 流水线阻塞 (Pipeline Blocker)")
            
            if node['context']:
                lines.append(f"   - 关键信息: {node['context']}")
            lines.append("")
        
        return "\n".join(lines)

# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='分析 PostgreSQL 执行计划')
    parser.add_argument('--sql', type=str, help='SQL 查询语句（如果提供，将从数据库获取执行计划）')
    parser.add_argument('--json', type=str, help='执行计划的 JSON 字符串（如果提供，直接使用）')
    parser.add_argument('--analyze', action='store_true', help='是否执行 ANALYZE（实际执行查询）')
    parser.add_argument('--host', type=str, default=None, help='数据库主机地址（默认从环境变量 PG_HOST 获取）')
    parser.add_argument('--port', type=int, default=None, help='数据库端口（默认从环境变量 PG_PORT 获取）')
    parser.add_argument('--database', type=str, default=None, help='数据库名称（默认从环境变量 PG_DATABASE 获取）')
    parser.add_argument('--user', type=str, default=None, help='数据库用户名（默认从环境变量 PG_USER 获取）')
    parser.add_argument('--password', type=str, default=None, help='数据库密码（默认从环境变量 PG_PASSWORD 获取）')
    
    args = parser.parse_args()
    
    # 构建数据库配置
    db_config = None
    if args.sql:
        db_config = {}
        if args.host:
            db_config['host'] = args.host
        if args.port:
            db_config['port'] = args.port
        if args.database:
            db_config['database'] = args.database
        if args.user:
            db_config['user'] = args.user
        if args.password:
            db_config['password'] = args.password
        if not db_config:
            db_config = None  # 使用环境变量或默认值
    
    try:
        if args.sql:
            # 从数据库获取执行计划
            print(f"📊 正在从数据库获取 SQL 的执行计划...")
            print(f"SQL: {args.sql[:100]}..." if len(args.sql) > 100 else f"SQL: {args.sql}")
            print(f"ANALYZE: {'是' if args.analyze else '否'}\n")
            
            analyzer = PlanAnalyzer(sql_query=args.sql, analyze=args.analyze, db_config=db_config)
            print(analyzer.format_analysis_report())
            
        elif args.json:
            # 直接使用 JSON 字符串
            print("📊 正在分析提供的执行计划 JSON...\n")
            analyzer = PlanAnalyzer(explain_json_str=args.json)
            print(analyzer.format_analysis_report())
            
        else:
            # 交互式模式：从标准输入读取 SQL
            print("=" * 60)
            print("PostgreSQL 执行计划分析工具")
            print("=" * 60)
            print("\n请输入 SQL 查询语句（输入空行结束）:\n")
            
            sql_lines = []
            while True:
                try:
                    line = input()
                    if not line.strip():
                        break
                    sql_lines.append(line)
                except EOFError:
                    break
            
            sql_query = '\n'.join(sql_lines).strip()
            
            if not sql_query:
                print("❌ 未输入 SQL 查询语句")
                sys.exit(1)
            
            print(f"\n📊 正在从数据库获取执行计划...")
            print(f"SQL: {sql_query[:100]}..." if len(sql_query) > 100 else f"SQL: {sql_query}")
            print(f"ANALYZE: 否（仅获取估算计划）\n")
            
            analyzer = PlanAnalyzer(sql_query=sql_query, analyze=False, db_config=db_config)
            print("\n" + "=" * 60)
            print("执行计划分析结果:")
            print("=" * 60 + "\n")
            print(analyzer.format_analysis_report())
            
    except KeyboardInterrupt:
        print("\n\n❌ 用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)