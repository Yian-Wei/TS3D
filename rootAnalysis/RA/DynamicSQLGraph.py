import heapq
from collections import defaultdict, deque


class DynamicSQLGraph:
    def __init__(self, timeout_threshold=1):  # 默认超时阈值5秒
        self.nodes = {}  # {sql_id: SQLNode}
        self.adjacency = {}  # 邻接表
        self.timeout = timeout_threshold
        self.temporal_edges = []  # 带时间窗口的边

    class SQLNode:
        def __init__(self, sql_id, start_ts, duration, sql_text):
            self.id = sql_id
            self.start = start_ts  # 时间戳(ms)
            self.end = start_ts + duration
            self.duration = duration
            self.text = sql_text
            self.is_timeout = duration > 1000


    def add_sql(self, sql_id, start_ts, duration, sql_text):
        """动态添加SQL节点并构建时序关系"""
        new_node = self.SQLNode(sql_id, start_ts, duration, sql_text)
        self.nodes[sql_id] = new_node
        self.adjacency[sql_id] = []

        # 动态建立时序因果关系
        for existing_id, node in self.nodes.items():
            if existing_id == sql_id:
                continue
            # 建立时序边规则
            if node.end <= new_node.start:  # 前序关系
                self._add_edge(existing_id, sql_id, 'sequential')
            elif node.start <= new_node.end and node.end >= new_node.start:  # 时间重叠
                self._add_edge(existing_id, sql_id, 'parallel')
                self._add_edge(sql_id, existing_id, 'parallel')

        # 动态维护时间窗口边
        self.temporal_edges.append((
            new_node.start,
            new_node.end,
            sql_id
        ))

    def _add_edge(self, src, dest, edge_type):
        """添加带类型的边  表示其后置边"""
        self.adjacency[src].append({
            'target': dest,
            'type': edge_type,
            'weight': self.nodes[dest].duration
        })

    def detect_abnormal_chains(self):
        """异常检测入口函数"""
        results = {
            'timeout_chains': self._find_timeout_chains(),
            # 'deadlock_loops': self._detect_causal_loops()
        }
        return results

    def _find_timeout_chains(self):
        """基于动态规划的超时链路检测"""
        dp = defaultdict(int)  # {node_id: 累计时长}
        prev = {}

        # 按执行时间排序节点
        sorted_nodes = sorted(self.nodes.values(), key=lambda x: x.start)

        for node in sorted_nodes:
            dp[node.id] = node.duration
            for pred in self._get_predecessors(node.id):
                if dp[pred] + node.duration > dp[node.id]:
                    dp[node.id] = dp[pred] + node.duration
                    prev[node.id] = pred

        # 找出累计超时的最长链
        timeout_nodes = [n.id for n in sorted_nodes if n.is_timeout]
        critical_path = []
        max_duration = 0

        if len(timeout_nodes) == 0:
            sorted_duration = sorted(self.nodes.values(), key=lambda x: x.duration)
            timeout_nodes = [sorted_duration[-1].id]

        for node_id in timeout_nodes:
            current_path = []
            current = node_id
            while current in prev:
                current_path.append(current)
                current = prev[current]
            current_path.reverse()

            path_duration = sum(self.nodes[n].duration for n in current_path)
            if path_duration > max_duration:
                max_duration = path_duration
                critical_path = current_path


        return {
            'critical_path': critical_path,
            'total_duration': max_duration,
            'nodes': [self.nodes[n].text for n in critical_path]
        }

    def _get_predecessors(self, node_id):
        """获取给定 SQL 节点的所有前驱节点"""
        predecessors = []
        for src, edges in self.adjacency.items():
            for edge in edges:
                if edge["target"] == node_id:
                    predecessors.append(src)
        return predecessors


