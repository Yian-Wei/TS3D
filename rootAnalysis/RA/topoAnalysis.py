from collections import deque

# method1
# def topological_sort(graph):
#     indegree = {node: 0 for node in graph}
#     for u in graph:
#         for v in graph[u]:
#             indegree[v] += 1
#
#     queue = deque([node for node in indegree if indegree[node] == 0])  # 用 deque
#     result = []
#
#     while queue:
#         node = queue.popleft()  # O(1) 而不是 O(n)
#         result.append(node)
#         for neighbor in graph[node]:
#             indegree[neighbor] -= 1
#             if indegree[neighbor] == 0:
#                 queue.append(neighbor)
#
#     return result


# 检查是否存在执行时间异常
# def check_time_limit(graph, execution_times, time_limit):
#     topo_order = topological_sort(graph)
#     execution_time = {node: 0 for node in graph}  # 记录每个节点的完成时间
#
#     for node in topo_order:
#         # 当前节点的执行时间是其自己的执行时间加上前置节点的最大执行时间
#         max_time_from_dependencies = 0
#         for parent in graph[node]:
#             max_time_from_dependencies = max(max_time_from_dependencies, execution_time[parent])
#         execution_time[node] = max_time_from_dependencies + execution_times[node]
#
#         # 判断是否超时
#         if execution_time[node] > time_limit:
#             # 超过时间限制，返回异常节点或段
#             return f"Node {node} exceeds time limit!"
#
#     return "No time limit violations"
from collections import deque
import heapq

# method2
def topological_sort_optimized(graph):
    # 计算入度并初始化优先队列（按出度降序排列）
    indegree = {node: 0 for node in graph}
    outdegree = {node: len(graph[node]) for node in graph}  # 用于优先级排序
    reverse_graph = {node: [] for node in graph}  # 反向图用于环检测

    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
            reverse_graph[v].append(u)  # 构建反向图

    # 使用优先队列（最大堆），按出度大小排序
    heap = [(-outdegree[node], node) for node in indegree if indegree[node] == 0]
    heapq.heapify(heap)
    result = []

    while heap:
        _, node = heapq.heappop(heap)
        result.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                heapq.heappush(heap, (-outdegree[neighbor], neighbor))

    if len(result) != len(graph):
        # 存在环，使用反向图快速查找环路径
        remaining = set(graph.keys()) - set(result)
        cycle = find_cycle_with_reverse_graph(reverse_graph, remaining)
        return {
            'topological_order': result,
            'cycle': cycle,
            'has_cycle': True
        }
    else:
        return {
            'topological_order': result,
            'cycle': None,
            'has_cycle': False
        }


def find_cycle_with_reverse_graph(reverse_graph, remaining_nodes):
    visited = {}
    stack = []

    def dfs(node):
        if node in visited:
            return visited[node]
        visited[node] = False
        stack.append(node)
        for parent in reverse_graph[node]:
            if parent not in remaining_nodes:
                continue
            if parent in stack:
                idx = stack.index(parent)
                cycle = stack[idx:] + [parent]
                return cycle
            detected_cycle = dfs(parent)
            if detected_cycle:
                return detected_cycle
        stack.pop()
        visited[node] = True
        return None

    for node in remaining_nodes:
        if node not in visited:
            cycle = dfs(node)
            if cycle:
                return cycle
    return []


from collections import defaultdict


def distributed_topological_sort(graph, partitions):
    """
    适用于分布式数据库的拓扑排序算法。
    :param graph: {node: [dependencies]} 任务依赖图
    :param partitions: {node: partition_id} 节点到分区的映射
    :return: 排序结果及环检测信息
    """

    # 初始化入度、出度、反向图
    indegree = defaultdict(int)
    outdegree = {node: len(graph[node]) for node in graph}
    reverse_graph = defaultdict(list)

    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
            reverse_graph[v].append(u)

    # 分区感知的拓扑排序队列
    partition_queues = defaultdict(list)
    for node in graph:
        if indegree[node] == 0:
            partition = partitions.get(node, 0)
            heapq.heappush(partition_queues[partition], (-outdegree[node], node))

    result = []
    processed_nodes = set()

    while partition_queues:
        # 选择负载最小的分区进行处理
        partition = min(partition_queues, key=lambda p: len(partition_queues[p]))
        if not partition_queues[partition]:
            del partition_queues[partition]
            continue

        _, node = heapq.heappop(partition_queues[partition])
        processed_nodes.add(node)
        result.append(node)

        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                neighbor_partition = partitions.get(neighbor, 0)
                heapq.heappush(partition_queues[neighbor_partition], (-outdegree[neighbor], neighbor))

    # 检查是否有未处理的节点
    remaining_nodes = set(graph.keys()) - processed_nodes
    if remaining_nodes:
        cycle = find_cycle_with_reverse_graph(reverse_graph, remaining_nodes)
        return {
            'topological_order': result,
            'cycle': cycle,
            'has_cycle': True
        }
    else:
        return {
            'topological_order': result,
            'cycle': None,
            'has_cycle': False
        }


def check_time_limit(graph, execution_times, time_limit, node_dict):
    # topo_order = distributed_topological_sort(graph, node_dict)
    topo_order = topological_sort_optimized(graph)
    print(topo_order)
    execution_time = {node: 0 for node in graph}
    max_time = 0
    max_node = 1
    print(graph[4])
    for node in topo_order['topological_order']:
         # 当前节点的执行时间是其自己的执行时间加上前置节点的最大执行时间
        max_time_from_dependencies = 0
        for parent in graph[node]:
            max_time_from_dependencies = max(max_time_from_dependencies, execution_time[parent])
        execution_time[node] = max_time_from_dependencies + execution_times[node]
        # 判断是否超时
        # if len(graph[node]) > 0 and execution_time[node] > time_limit:
        #     if execution_time[node] > max_time:
        #         max_time = execution_time[node]
        #         max_node = node

        if len(graph[node]) > 0 and execution_time[node] > max_time:
            max_time = execution_time[node]
            max_node = node
                # 超过时间限制，返回异常节点或段
    print(f"Node {max_node}, parent node {graph[max_node]}")
    # return (f"Node {node} exceeds time limit!")

    return "No time limit violations"

