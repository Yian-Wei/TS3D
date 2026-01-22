# This is a sample Python script.
from datetime import datetime, time
import time

from RA.DynamicSQLGraph import DynamicSQLGraph
from RA.sqlParse import build_dependency_graph, draw_graph
from utils.preprocess import pre_process, transfer_data, cal_select, topo_data


def convert_sql_data(data):
    sql_id = data['sql_id']
    timestamp = datetime.strptime(data['time'], '%Y-%m-%d %H:%M:%S.%f')
    epoch_ms = int(timestamp.timestamp() * 1000)  
    duration = int(data['duration'])  
    sql_text = data['sql']
    return (sql_id, epoch_ms, duration, sql_text)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_path = "../data/m2m/logs/sqls.txt"
    group_data, keys = pre_process(file_path)

    # example
    # log_data= """2025-01-06 10:31:03.020828	10.244.0.20:3306	abnormal_select_pod_2	'SELECT o.driver_id, COUNT(p.payment_id) AS payment_count FROM orders o JOIN payments p ON o.order_id = p.order_id GROUP BY o.driver_id;'	1	0.010317087173461914	0
    # 2025-01-06 10:31:03.029022	10.244.0.20:3306	abnormal_select_pod_2	'SELECT o.driver_id, SUM(p.fare) AS total_amount FROM orders o JOIN payments p ON o.order_id = p.order_id GROUP BY o.driver_id;'	1	0.01604604721069336	0
    # 2025-01-06 10:31:03.086807	10.244.0.20:3306	abnormal_select_pod_2	'SELECT driver_id, COUNT(*) AS order_count FROM orders GROUP BY driver_id;'	1	0.05759406089782715	0
    # 2025-01-06 10:31:03.088035	10.244.0.20:3306	abnormal_select_pod_2	'SELECT MAX(fare) AS max_amount FROM payments;'	1	0.0005707740783691406	0
    # 2025-01-06 10:31:03.106368	10.244.0.20:3306	abnormal_select_pod_2	'SELECT o.driver_id, AVG(p.fare) AS average_amount FROM orders o JOIN payments p ON o.order_id = p.order_id GROUP BY o.driver_id;'	1	0.01878523826599121	0
    # 2025-01-06 10:31:03.111973	10.244.0.20:3306	abnormal_select_pod_2	'SELECT COUNT(*) AS total_users FROM users;'	1	0.005342721939086914	0
    # 2025-01-06 10:31:03.116715	10.244.0.20:3306	abnormal_select_pod_2	'SELECT COUNT(*) AS total_payments FROM payments;'	1	0.0005164146423339844	0
    # 2025-01-06 10:31:03.117503	10.244.0.20:3306	abnormal_select_pod_2	'SELECT COUNT(*) AS total_drivers FROM drivers;'	1	0.005965232849121094	0"""


    processed_data, node_dict = transfer_data(group_data.get(keys[0]))

    G = build_dependency_graph(processed_data)

    now = time.time()
    graph = DynamicSQLGraph(timeout_threshold=3000)  
    for sql in processed_data:
        graph.add_sql(*convert_sql_data(sql))

    abnormal = graph.detect_abnormal_chains()

    print("超时链路:", abnormal['timeout_chains'])
    # print("因果环:", abnormal['deadlock_loops'])
    print((time.time() - now) * 1000)



