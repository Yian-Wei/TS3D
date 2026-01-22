import json
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import ast
def read_json(filename):
    with open(filename, "r") as load_f:
        file_dict = json.load(load_f)
    return file_dict


def trp(l, n):
    """Truncate or pad a list"""
    r = l[:n]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def down_sample(logs, labels, sample_ratio):
    print("sampling...")
    total_num = len(labels)
    all_index = list(range(total_num))
    sample_logs = {}
    for key in logs.keys():
        sample_logs[key] = []
    sample_labels = []
    sample_num = int(total_num * sample_ratio)

    for i in tqdm(range(sample_num)):
        random_index = int(np.random.uniform(0, len(all_index)))
        for key in logs.keys():
            sample_logs[key].append(logs[key][random_index])
        sample_labels.append(labels[random_index])
        del all_index[random_index]
    return sample_logs, sample_labels


def sliding_window(data_dir, datatype, window_size, sample_ratio=1):
    event2semantic_vec = read_json(data_dir + "s2s/template_vectors.json")
    num_sessions = 0
    result_logs = {}
    result_logs["Sequentials"] = []
    result_logs["Quantitatives"] = []
    result_logs["Semantics"] = []
    result_logs["Metrics"] = []  
    labels = []
    if datatype == "train":
        data_dir += "s2s/train.csv"
    if datatype == "val":
        data_dir += "s2s/test_normal.csv"
    try:
        df = pd.read_csv(data_dir)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_dir}")
        return result_logs, labels
    metric_cols = [
        'aggregated_log_level','max_response_time','cpu', 'memory', 'qps', 'disk_usage', 'memory_usage', 'iops', 
        'slow_queries', 'select_full_join', 'select_scan', 'threads_created', 
        'aborted_connects', 'aborted_clients', 'table_locks_waited', 
        'innodb_row_lock_waits', 'innodb_buffer_pool_read_requests', 
        'innodb_buffer_pool_reads', 'innodb_buffer_pool_write_requests', 
        'seconds_behind_master', 'connections', 'max_used_connections', 
        'qcache_hits', 'qcache_inserts', 'qcache_lowmem_prunes', 
        'created_tmp_tables', 'created_tmp_disk_tables', 
        'binlog_cache_disk_use', 'binlog_cache_use'
    ]
    df = df.dropna(subset=['event_ids_list'])
    for _, row in df.iterrows():
        num_sessions += 1
        
        try:
            event_ids_str = row['event_ids_list'].strip()[1:-1] 
            line = list(map(lambda n: int(float(n)) - 1, event_ids_str.split(', ')))
        except Exception as e:
            print(f"Error processing event_ids_list: {e}")
            continue
        metrics_data = row[metric_cols].values.tolist()
        for i in range(len(line) - window_size):
            Sequential_pattern = list(line[i : i + window_size])
            Quantitative_pattern = [0] * 32
            log_counter = Counter(Sequential_pattern)

            for key in log_counter:
                Quantitative_pattern[key] = log_counter[key]
            Semantic_pattern = []
            for event in Sequential_pattern:
                if event == 0:
                    Semantic_pattern.append([-1] * 300)
                else:
                    Semantic_pattern.append(event2semantic_vec[str(event)])
            Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
            Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
            Metrics_pattern = np.array(metrics_data,dtype=np.float32) 
            result_logs["Sequentials"].append(Sequential_pattern)
            result_logs["Quantitatives"].append(Quantitative_pattern)
            result_logs["Semantics"].append(Semantic_pattern)
            result_logs["Metrics"].append(Metrics_pattern) 
            labels.append(line[i + window_size])

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    print("File {}, number of sessions {}".format(data_dir, num_sessions))
    print(
        "File {}, number of seqs {}".format(data_dir, len(result_logs["Sequentials"]))
    )

    return result_logs, labels


def session_window(data_dir, datatype, sample_ratio=1):
    event2semantic_vec = read_json("../../data/s2s/template_vectors.json")
    result_logs = {}
    result_logs["Sequentials"] = []
    result_logs["Quantitatives"] = []
    result_logs["Semantics"] = []
    result_logs["Metrics"] = []
    labels = []

    if datatype == "train":
        data_dir += "train.csv"
    elif datatype == "val":
        data_dir += "valid.csv"
    elif datatype == "test":
        data_dir += "test.csv"
    df = pd.read_csv(data_dir)
    metric_cols = [
        "cpu","memory","qps","disk_usage","memory_usage","iops",
        "slow_queries","select_full_join","select_scan","threads_created",
        "aborted_connects","aborted_clients","table_locks_waited",
        "innodb_row_lock_waits","innodb_buffer_pool_read_requests",
        "innodb_buffer_pool_reads","innodb_buffer_pool_write_requests",
        "seconds_behind_master","connections","max_used_connections",
        "qcache_hits","qcache_inserts","qcache_lowmem_prunes",
        "created_tmp_tables","created_tmp_disk_tables",
        "binlog_cache_disk_use","binlog_cache_use","max_response_time",
        "num_events"
    ]
    metric_cols = [col for col in metric_cols if col in df.columns]
    train_df = pd.read_csv(data_dir)
    for i in tqdm(range(len(train_df))):
        ori_seq = [int(eventid) for eventid in ast.literal_eval(train_df["event_ids_list"][i])]
        Sequential_pattern = trp(ori_seq, 50)
        Semantic_pattern = []
        for event in Sequential_pattern:
            if event == 0:
                Semantic_pattern.append([-1] * 46)
            else:
                Semantic_pattern.append(event2semantic_vec[str(event)])
        Quantitative_pattern = [0] * 32
        log_counter = Counter(Sequential_pattern)

        for key in log_counter:
            Quantitative_pattern[key] = log_counter[key]
        metric_values = df.loc[i, metric_cols].values.astype(float)
        Metrics_pattern = metric_values
        Sequential_pattern = np.array(Sequential_pattern)[:, np.newaxis]
        Quantitative_pattern = np.array(Quantitative_pattern)[:, np.newaxis]
        result_logs["Sequentials"].append(Sequential_pattern)
        result_logs["Quantitatives"].append(Quantitative_pattern)
        result_logs["Semantics"].append(Semantic_pattern)
        result_logs["Metrics"].append(Metrics_pattern)
        labels.append(int(train_df["label"][i]))

    if sample_ratio != 1:
        result_logs, labels = down_sample(result_logs, labels, sample_ratio)

    # result_logs, labels = up_sample(result_logs, labels)

    print("Number of sessions({}): {}".format(data_dir, len(result_logs["Semantics"])))
    return result_logs, labels
