"""
PostgreSQL数据库连接工具（简化版）
用于获取执行计划
"""
import json
from typing import Dict, Any, Optional

from src.utils.llm_json_utils import loads_with_repair
import psycopg2
from psycopg2.extras import RealDictCursor


class PostgresConnector:
    """PostgreSQL数据库连接器（简化版）"""
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "tpch",
        user: str = "postgres",
        password: str = "123456",
    ):
        """
        初始化数据库连接器
        
        Args:
            host: 数据库主机地址
            port: 数据库端口
            database: 数据库名称
            user: 用户名
            password: 密码
        """
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self._connection = None
    
    def _get_connection(self):
        """获取数据库连接（懒加载）"""
        if self._connection is None or self._connection.closed:
            try:
                self._connection = psycopg2.connect(
                    host=self.host,
                    port=self.port,
                    database=self.database,
                    user=self.user,
                    password=self.password,
                    connect_timeout=10
                )
            except psycopg2.Error as e:
                raise
        return self._connection
    
    def close(self):
        """关闭数据库连接"""
        if self._connection and not self._connection.closed:
            self._connection.close()
            self._connection = None
    
    def execute_explain(self, sql_query: str, analyze: bool = False, format: str = "JSON") -> Dict[str, Any]:
        """
        执行 EXPLAIN 或 EXPLAIN ANALYZE 并返回结果
        
        Args:
            sql_query: 要分析的SQL查询
            analyze: 是否执行 ANALYZE（实际执行查询）
            format: 返回格式，'JSON' 或 'TEXT'
            
        Returns:
            dict: 包含执行计划信息的字典
        """
        try:
            conn = self._get_connection()
            
            if format.upper() == "JSON":
                cursor = conn.cursor(cursor_factory=RealDictCursor)
                if analyze:
                    # 实际执行查询，获取真实的执行时间与缓冲区使用情况
                    explain_query = f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE, FORMAT JSON) {sql_query}"
                else:
                    # 只获取优化器估算的执行计划和缓冲区信息，不实际执行查询
                    explain_query = f"EXPLAIN (BUFFERS, VERBOSE, FORMAT JSON) {sql_query}"
                cursor.execute(explain_query)
                result = cursor.fetchone()
                cursor.close()
                
                if result is None:
                    return {'error': 'EXPLAIN执行返回None，可能是SQL语法错误'}
                
                # result 的格式取决于cursor类型：
                # - RealDictCursor: 返回 RealDictRow（类似字典），键是 'QUERY PLAN'，值是列表
                # - 普通cursor: 返回元组，第一个元素是结果
                try:
                    # 尝试不同的结果格式
                    if hasattr(result, 'keys') or isinstance(result, dict):
                        # RealDictCursor 返回字典或 RealDictRow
                        # 键通常是 'QUERY PLAN'（大写，带空格）
                        plan_data = result.get('QUERY PLAN') or result.get('query plan') or result.get('Query Plan')
                        # 如果还是None，尝试获取第一个值
                        if plan_data is None and result:
                            keys = list(result.keys()) if hasattr(result, 'keys') else list(result.keys())
                            if keys:
                                plan_data = result[keys[0]]
                    elif isinstance(result, (list, tuple)):
                        # 普通cursor返回元组，第一个元素是结果
                        if len(result) > 0:
                            plan_data = result[0]
                        else:
                            return {'error': 'EXPLAIN执行返回空结果'}
                    else:
                        plan_data = result
                    
                    if plan_data is None:
                        return {'error': '无法从执行计划结果中提取数据'}
                    
                    # 如果 plan_data 是字符串，尝试解析JSON
                    if isinstance(plan_data, str):
                        try:
                            plan_data = loads_with_repair(plan_data)
                        except json.JSONDecodeError:
                            return {'error': f'无法解析JSON格式的执行计划: {plan_data[:200]}'}
                    
                    # 如果 plan_data 是列表，取第一个元素
                    if isinstance(plan_data, list):
                        if len(plan_data) > 0:
                            plan_node = plan_data[0]
                        else:
                            return {'error': '执行计划列表为空'}
                    else:
                        plan_node = plan_data
                    
                    # plan_node 应该是一个字典
                    if not isinstance(plan_node, dict):
                        return {'error': f'执行计划节点格式错误: {type(plan_node)}'}
                    
                    plan = plan_node.get('Plan', {})
                    
                    # 提取总代价：PostgreSQL EXPLAIN JSON格式中，最外层Plan节点的"Total Cost"是查询的总代价
                    total_cost = None
                    
                    # 优先从最外层Plan节点获取Total Cost（这是真正的总代价）
                    if plan and isinstance(plan, dict):
                        total_cost = plan.get('Total Cost')
                    
                    # 如果Plan中没有Total Cost，尝试从plan_node本身获取（某些格式可能不同）
                    if total_cost is None:
                        total_cost = plan_node.get('Total Cost')
                    
                    return {
                        'plan': plan,
                        'execution_time': plan_node.get('Execution Time') if analyze else None,
                        'planning_time': plan_node.get('Planning Time') if analyze else None,
                        'total_cost': total_cost,
                        'startup_cost': plan.get('Startup Cost') if plan else None,
                        'plan_rows': plan.get('Plan Rows') if plan else None,
                        'actual_rows': plan.get('Actual Rows') if analyze and plan else None,
                        'raw': plan_node
                    }
                except (KeyError, IndexError, TypeError) as e:
                    return {'error': f'解析执行计划失败: {type(e).__name__}: {str(e)}'}
                
            else:  # TEXT format
                cursor = conn.cursor()
                if analyze:
                    explain_query = f"EXPLAIN (ANALYZE, BUFFERS, VERBOSE) {sql_query}"
                else:
                    # 使用 COSTS ON（默认开启，但显式指定更清晰）
                    explain_query = f"EXPLAIN (COSTS ON, VERBOSE) {sql_query}"
                cursor.execute(explain_query)
                rows = cursor.fetchall()
                cursor.close()
                
                plan_text = '\n'.join([row[0] for row in rows])
                
                # 尝试从TEXT格式中提取总代价（第一行的cost=xxx..yyy中的yyy）
                total_cost = None
                if plan_text:
                    import re
                    # 匹配第一行的 cost=xxx..yyy 格式
                    cost_match = re.search(r'cost=(\d+\.?\d*)\.\.(\d+\.?\d*)', plan_text, re.IGNORECASE)
                    if cost_match:
                        # 取第二个值（结束代价，即总代价）
                        total_cost = float(cost_match.group(2))
                
                return {
                    'plan_text': plan_text,
                    'format': 'TEXT',
                    'total_cost': total_cost
                }
                
        except psycopg2.Error as e:
            error_msg = str(e) if e else "未知错误"
            return {'error': error_msg, 'sql': sql_query}
        except Exception as e:
            error_msg = str(e) if e else "未知错误"
            error_type = type(e).__name__
            return {'error': error_msg, 'sql': sql_query}

