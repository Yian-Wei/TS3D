import pandas as pd
import re
from collections import defaultdict
from datetime import datetime, timedelta

from RA.sqlParse import build_dependency_graph, draw_graph
from RA.topoAnalysis import check_time_limit


def pre_process(file_path):

    df = pd.read_csv(file_path, sep='\t', engine='python')
    df = df[df["label"] == 1]
    # 按日志类型分组
    # 按 detail_type 分组，并将每组数据转换为字典列表
    grouped_data = df.groupby("detail_type").apply(lambda x: x.to_dict('records')).to_dict()
    keys  = grouped_data.keys()
    print(keys)
    # # 按 detail_type 分组
    # grouped_data = filtered_df.groupby("detail_type")["sql_command"].apply(list).to_dict()

    # # 输出分组后的数据
    # for detail_type, sql_commands in grouped_data.items():
    #     print(f"Logs for detail_type: {detail_type}")
    #     for sql in sql_commands:
    #         print(sql)
    #     print("\n" + "=" * 50 + "\n")
    #
    # # 打印分组后的字典
    # print("Grouped data as a dictionary:")
    # print(grouped_data)

    return grouped_data, list(keys)


def transfer_data(log_data):

    # 正则表达式用于提取 SQL 语句
    sql_pattern = re.compile(r"'([^']+)'")

    # 存储 SQL 语句及其相关信息
    sql_list = []
    node_dict = {}
    table_to_sql_ids = defaultdict(list)  # 记录每个表格涉及的 SQL 语句 ID

    # 我们用其中的100条做对比实验
    # 解析日志数据
    # for line in log_data.strip().split('\n'):
    i = 0
    for line in log_data:
        if i < 100:
            i+=1
        else: break
        if not line:
            continue
        # parts = line.split('\t')
        sql = sql_pattern.search(line['sql_command']).group(1)  # 提取 SQL 语句
        duration = float(line['exe_time']) * 1000  # 转换为毫秒

        # 提取涉及的表格
        tables = set(re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE))
        tables.update(re.findall(r'JOIN\s+(\w+)', sql, re.IGNORECASE))
        tables.update(re.findall(r'INSERT\s+INTO\s+(\w+)', sql, re.IGNORECASE))
        tables.update(re.findall(r'UPDATE\s+(\w+)', sql, re.IGNORECASE))
        tables.update(re.findall(r'DELETE\s+FROM\s+(\w+)', sql, re.IGNORECASE))
        tables.update(re.findall(r'LOCK\s+TABLES\s+(\w+)', sql, re.IGNORECASE))

        # 判断操作类型
        if re.search(r'SELECT', sql, re.IGNORECASE):
            operation = 'SELECT'
        elif re.search(r'INSERT', sql, re.IGNORECASE):
            operation = 'INSERT'
        elif re.search(r'UPDATE', sql, re.IGNORECASE):
            operation = 'UPDATE'
        elif re.search(r'DELETE', sql, re.IGNORECASE):
            operation = 'DELETE'
        elif re.search(r'LOCK\s+TABLES', sql, re.IGNORECASE):
            operation = 'LOCK'
        else:
            operation = 'UNKNOWN'

        # 分配 SQL ID
        sql_id = len(sql_list) + 1
        sql_list.append({
            'time': line['timestamp'],
            'node': line['ip:port'],
            'sql_id': sql_id,
            'sql': sql,
            'duration': duration,
            'operation': operation,
            'dependencies': []
        })

        node_dict[sql_id] = line['ip:port']

        # 记录表格与 SQL ID 的映射关系
        for table in tables:
            table_to_sql_ids[table].append(sql_id)

    # 生成依赖关系
    for sql_info in sql_list:
        sql = sql_info['sql']
        sql_id = sql_info['sql_id']
        operation = sql_info['operation']

        # 提取当前 SQL 语句涉及的表格
        tables_in_sql = set(re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql.update(re.findall(r'JOIN\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql.update(re.findall(r'INSERT\s+INTO\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql.update(re.findall(r'UPDATE\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql.update(re.findall(r'DELETE\s+FROM\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql.update(re.findall(r'LOCK\s+TABLES\s+(\w+)', sql, re.IGNORECASE))

        # 检查依赖关系
        for table in tables_in_sql:
            # 查找之前操作过该表格的 SQL 语句
            for dep_sql_id in table_to_sql_ids[table]:
                if dep_sql_id < sql_id:  # 只依赖之前的 SQL 语句
                    dep_sql_info = sql_list[dep_sql_id - 1]
                    dep_operation = dep_sql_info['operation']

                    # 判断是否由于锁导致依赖
                    if (operation == 'SELECT' and dep_operation in ['INSERT', 'UPDATE', 'DELETE', 'LOCK']) or \
                            (operation in ['INSERT', 'UPDATE', 'DELETE'] and dep_operation in ['INSERT', 'UPDATE',
                                                                                               'DELETE', 'LOCK']):
                        # 检查 SQL 语句的条件是否重叠
                        if operation == 'SELECT' and dep_operation in ['INSERT', 'UPDATE', 'DELETE']:
                            # 假设 SELECT 语句的条件与写操作的条件重叠
                            sql_info['dependencies'].append(dep_sql_id)
                        elif operation in ['INSERT', 'UPDATE', 'DELETE'] and dep_operation in ['INSERT', 'UPDATE',
                                                                                               'DELETE']:
                            # 假设写操作的条件重叠
                            sql_info['dependencies'].append(dep_sql_id)

        # 去重依赖关系
        sql_info['dependencies'] = list(set(sql_info['dependencies']))

    # 输出结果
    for sql_info in sql_list:
        print(sql_info)
    return sql_list, node_dict

def cal_select(processed_data):
    # 存储合并后的 SELECT 语句
    merged_selects = defaultdict(lambda: {'count': 0, 'timestamps': [], 'durations': []})

    # 定义时间窗口（1 秒）
    time_window = timedelta(seconds=1)

    # 解析数据
    for entry in processed_data:
        sql = entry['sql']
        duration = entry['duration']
        dependencies = entry['dependencies']
        timestamp = datetime.strptime(entry['time'].strip(), "%Y-%m-%d %H:%M:%S.%f")

        # 仅处理 dependencies 为空的 SELECT 语句
        if entry['operation'] == 'SELECT' and not dependencies:
            # 检查是否可以合并
            merged = False
            for key in merged_selects:
                if key == sql:  # SQL 文本相同
                    last_timestamp = merged_selects[key]['timestamps'][-1]
                    if timestamp - last_timestamp <= time_window:  # 时间戳在同一窗口内
                        merged_selects[key]['count'] += 1
                        merged_selects[key]['timestamps'].append(timestamp)
                        merged_selects[key]['durations'].append(duration)
                        merged = True
                        break

            # 如果不能合并，则新增一条记录
            if not merged:
                merged_selects[sql] = {
                    'count': 1,
                    'timestamps': [timestamp],
                    'durations': [duration]
                }

    # 生成合并后的结果
    merged_results = []
    for sql, data in merged_selects.items():
        avg_duration = sum(data['durations']) / len(data['durations'])  # 计算平均执行时间
        merged_results.append({
            'sql': sql,
            'count': data['count'],
            'avg_duration': avg_duration,
            'first_timestamp': data['timestamps'][0].strftime("%Y-%m-%d %H:%M:%S.%f"),
            'last_timestamp': data['timestamps'][-1].strftime("%Y-%m-%d %H:%M:%S.%f")
        })

    # 输出结果
    for result in merged_results:
        print(result)


def topo_data(data):
    # 初始化 execution_times 和 graph
    execution_times = {}
    graph = {}

    # 遍历数据并填充 execution_times 和 graph
    for entry in data:
        sql_id = entry['sql_id']
        duration = entry['duration']
        dependencies = entry['dependencies']

        # 填充 execution_times
        execution_times[sql_id] = duration

        # 填充 graph
        graph[sql_id] = dependencies

    # 输出结果
    print("execution_times =", execution_times)
    print("graph =", graph)
    return execution_times, graph


