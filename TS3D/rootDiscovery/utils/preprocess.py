import pandas as pd
import re
from collections import defaultdict
from datetime import datetime, timedelta


def pre_process(file_path):

    df = pd.read_csv(file_path, sep='\t', engine='python')
    df = df[df["label"] == 1]

    grouped_data = df.groupby("detail_type").apply(lambda x: x.to_dict('records')).to_dict()
    keys  = grouped_data.keys()
    print(keys)

    return grouped_data, list(keys)


def transfer_data(log_data):

    sql_pattern = re.compile(r"'([^']+)'")

    sql_list = []
    node_dict = {}
    table_to_sql_ids = defaultdict(list)  

    i = 0
    for line in log_data:
        if i < 100:
            i+=1
        else: break
        if not line:
            continue
        # parts = line.split('\t')
        sql = sql_pattern.search(line['sql_command']).group(1)  
        duration = float(line['exe_time']) * 1000  

        tables = set(re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE))
        tables.update(re.findall(r'JOIN\s+(\w+)', sql, re.IGNORECASE))
        tables.update(re.findall(r'INSERT\s+INTO\s+(\w+)', sql, re.IGNORECASE))
        tables.update(re.findall(r'UPDATE\s+(\w+)', sql, re.IGNORECASE))
        tables.update(re.findall(r'DELETE\s+FROM\s+(\w+)', sql, re.IGNORECASE))
        tables.update(re.findall(r'LOCK\s+TABLES\s+(\w+)', sql, re.IGNORECASE))

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

        for table in tables:
            table_to_sql_ids[table].append(sql_id)

    for sql_info in sql_list:
        sql = sql_info['sql']
        sql_id = sql_info['sql_id']
        operation = sql_info['operation']

        tables_in_sql = set(re.findall(r'FROM\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql.update(re.findall(r'JOIN\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql.update(re.findall(r'INSERT\s+INTO\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql.update(re.findall(r'UPDATE\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql.update(re.findall(r'DELETE\s+FROM\s+(\w+)', sql, re.IGNORECASE))
        tables_in_sql.update(re.findall(r'LOCK\s+TABLES\s+(\w+)', sql, re.IGNORECASE))

        for table in tables_in_sql:
            for dep_sql_id in table_to_sql_ids[table]:
                if dep_sql_id < sql_id:  
                    dep_sql_info = sql_list[dep_sql_id - 1]
                    dep_operation = dep_sql_info['operation']

                    if (operation == 'SELECT' and dep_operation in ['INSERT', 'UPDATE', 'DELETE', 'LOCK']) or \
                            (operation in ['INSERT', 'UPDATE', 'DELETE'] and dep_operation in ['INSERT', 'UPDATE',
                                                                                               'DELETE', 'LOCK']):

                        if operation == 'SELECT' and dep_operation in ['INSERT', 'UPDATE', 'DELETE']:

                            sql_info['dependencies'].append(dep_sql_id)
                        elif operation in ['INSERT', 'UPDATE', 'DELETE'] and dep_operation in ['INSERT', 'UPDATE',
                                                                                               'DELETE']:

                            sql_info['dependencies'].append(dep_sql_id)

        sql_info['dependencies'] = list(set(sql_info['dependencies']))

    for sql_info in sql_list:
        print(sql_info)
    return sql_list, node_dict

def cal_select(processed_data):

    merged_selects = defaultdict(lambda: {'count': 0, 'timestamps': [], 'durations': []})

    time_window = timedelta(seconds=1)

    for entry in processed_data:
        sql = entry['sql']
        duration = entry['duration']
        dependencies = entry['dependencies']
        timestamp = datetime.strptime(entry['time'].strip(), "%Y-%m-%d %H:%M:%S.%f")

        if entry['operation'] == 'SELECT' and not dependencies:

            merged = False
            for key in merged_selects:
                if key == sql: 
                    last_timestamp = merged_selects[key]['timestamps'][-1]
                    if timestamp - last_timestamp <= time_window: 
                        merged_selects[key]['count'] += 1
                        merged_selects[key]['timestamps'].append(timestamp)
                        merged_selects[key]['durations'].append(duration)
                        merged = True
                        break

            if not merged:
                merged_selects[sql] = {
                    'count': 1,
                    'timestamps': [timestamp],
                    'durations': [duration]
                }

    merged_results = []
    for sql, data in merged_selects.items():
        avg_duration = sum(data['durations']) / len(data['durations'])  
        merged_results.append({
            'sql': sql,
            'count': data['count'],
            'avg_duration': avg_duration,
            'first_timestamp': data['timestamps'][0].strftime("%Y-%m-%d %H:%M:%S.%f"),
            'last_timestamp': data['timestamps'][-1].strftime("%Y-%m-%d %H:%M:%S.%f")
        })

    for result in merged_results:
        print(result)


def topo_data(data):
    
    execution_times = {}
    graph = {}

    for entry in data:
        sql_id = entry['sql_id']
        duration = entry['duration']
        dependencies = entry['dependencies']

        execution_times[sql_id] = duration

        graph[sql_id] = dependencies

    print("execution_times =", execution_times)
    print("graph =", graph)
    return execution_times, graph


