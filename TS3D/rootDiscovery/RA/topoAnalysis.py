import heapq

# method2
def topological_sort_optimized(graph):

    indegree = {node: 0 for node in graph}
    outdegree = {node: len(graph[node]) for node in graph}  
    reverse_graph = {node: [] for node in graph}  

    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
            reverse_graph[v].append(u)  

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

    indegree = defaultdict(int)
    outdegree = {node: len(graph[node]) for node in graph}
    reverse_graph = defaultdict(list)

    for u in graph:
        for v in graph[u]:
            indegree[v] += 1
            reverse_graph[v].append(u)

    partition_queues = defaultdict(list)
    for node in graph:
        if indegree[node] == 0:
            partition = partitions.get(node, 0)
            heapq.heappush(partition_queues[partition], (-outdegree[node], node))

    result = []
    processed_nodes = set()

    while partition_queues:

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
        max_time_from_dependencies = 0
        for parent in graph[node]:
            max_time_from_dependencies = max(max_time_from_dependencies, execution_time[parent])
        execution_time[node] = max_time_from_dependencies + execution_times[node]
        # if len(graph[node]) > 0 and execution_time[node] > time_limit:
        #     if execution_time[node] > max_time:
        #         max_time = execution_time[node]
        #         max_node = node

        if len(graph[node]) > 0 and execution_time[node] > max_time:
            max_time = execution_time[node]
            max_node = node

    print(f"Node {max_node}, parent node {graph[max_node]}")
    # return (f"Node {node} exceeds time limit!")

    return "No time limit violations"

