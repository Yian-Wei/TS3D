#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import sys
import time
from collections import Counter

sys.path.append("../../")

import ast
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import Counter
from logdeep.dataset.log import log_dataset
from logdeep.dataset.sample import session_window
from logdeep.tools.utils import save_parameters, seed_everything, train_val_split
import json
from logdeep.metric.metrics import get_metrics

def generate(name):
    window_size = 10
    hdfs = {}
    length = 0
    with open("../data/m2m/" + name, "r") as f:
        for ln in f.readlines():
            ln = list(map(lambda n: n - 1, map(int, ln.strip().split())))
            ln = ln + [-1] * (window_size + 1 - len(ln))
            hdfs[tuple(ln)] = hdfs.get(tuple(ln), 0) + 1
            length += 1
    print("Number of sessions({}): {}".format(name, len(hdfs)))
    return hdfs, length


class Predicter:
    def __init__(self, model, options):
        self.data_dir = options["data_dir"]
        self.device = options["device"]
        self.model = model
        self.model_path = options["model_path"]
        self.window_size = options["window_size"]
        self.num_candidates = options["num_candidates"]
        self.num_classes = options["num_classes"]
        self.input_size = options["input_size"]
        self.sequentials = options["sequentials"]
        self.quantitatives = options["quantitatives"]
        self.semantics = options["semantics"]
        self.padding_idx = self.num_classes
        self.metrics = options["metrics"] 
        self.metrics_dim = options["metrics_dim"] 
        self.batch_size = options["batch_size"]
        self.metrics_stats_path = os.path.join(options["save_dir"], 'metrics_stats.json')
        self.metrics_mean, self.metrics_std = self._load_metrics_stats(self.metrics_stats_path)

    def _load_metrics_map(self, name):

        file_name = "test_normal.csv" if name == "test_normal" else "test_abnormal.csv"
        file_path = os.path.join(self.data_dir, "m2m", file_name)
        print(f"Loading Metrics Map from: {file_path}")

        session_data_map = {}
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: Data file not found at {file_path}")
            return {}

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

        window_size = self.window_size 

        for _, row in df.iterrows():
            try:
                event_ids_list = ast.literal_eval(row['event_ids_list'])
            except:
                continue

            session_events = [eid - 1 for eid in event_ids_list]
            session_key = tuple(session_events)

            metrics_vector = row[metric_cols].values.astype(np.float32)
            if self.metrics_mean is not None and self.metrics_std is not None:
                metrics_vector = (metrics_vector - self.metrics_mean) / self.metrics_std

            if session_key not in session_data_map:
                session_data_map[session_key] = {'metrics': metrics_vector, 'count': 1}
            else:
                session_data_map[session_key]['count'] += 1

        print(f"Finished loading {len(session_data_map)} unique sessions for {name}")
        return session_data_map
    
    def _load_metrics_stats(self, stats_path):
        if not self.metrics:
            return None, None
            
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                mean = np.array(stats['mean'], dtype=np.float32)
                std = np.array(stats['std'], dtype=np.float32)
                std_safe = np.where(std == 0, 1.0, std)
                return mean, std_safe
                
        except FileNotFoundError:
            print(f"Warning: Metrics stats file not found at {stats_path}. Metrics will NOT be standardized.")
            return None, None
        except json.decoder.JSONDecodeError:
            print(f"Warning: Metrics stats file at {stats_path} is corrupted. Metrics will NOT be standardized.")
            return None, None
    
    def predict_unsupervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)["state_dict"])
        model.eval()
        print("model_path: {}".format(self.model_path))
    
        normal_data_name = "test_normal" 
        abnormal_data_name = "test_abnormal"

        normal_session_data_map = self._load_metrics_map(normal_data_name)
        abnormal_session_data_map = self._load_metrics_map(abnormal_data_name)

        num_normal = len(normal_session_data_map)

        test_normal_loader = normal_session_data_map.keys()
        test_abnormal_loader = abnormal_session_data_map.keys()

        test_normal_length = sum(v['count'] for v in normal_session_data_map.values())
        test_abnormal_length = sum(v['count'] for v in abnormal_session_data_map.values())
        start_time = time.time()
        all_labels = []
        all_predicted = []
        print("--- Testing Normal Logs ---")
        with torch.no_grad():
            for line in tqdm(test_normal_loader, desc="Testing Normal"):
                session_data = normal_session_data_map.get(line)
                if session_data is None: 
                    continue

                metrics_vector = session_data['metrics']
                metrics_input = torch.tensor(metrics_vector, dtype=torch.float)\
                                    .view(1, self.metrics_dim).to(self.device)

                is_abnormal_session = False
                error_count = 0
                for i in range(len(line) - self.window_size):
                    seq0 = line[i : i + self.window_size]
                    label = line[i + self.window_size]

                    seq0_tensor = torch.tensor(seq0, dtype=torch.long)\
                                    .view(1, self.window_size, 1).to(self.device)
                    seq1_vector = [0] * self.num_classes
                    log_counter = Counter(seq0)
                    for key in log_counter:
                        seq1_vector[int(key)] = log_counter[key]
                    seq1_tensor = torch.tensor(seq1_vector, dtype=torch.float)\
                                        .view(1, self.num_classes, self.input_size).to(self.device)
                
                    features = []
                    if self.sequentials: features.append(seq0_tensor)
                    if self.quantitatives: features.append(seq1_tensor)
                    if self.metrics: features.append(metrics_input)

                    output = model(features=features, device=self.device)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates :]

                    if label not in predicted:
                        error_count += 1

                if error_count / len(line) > 0.1:  
                    is_abnormal_session = True
                if is_abnormal_session:
                    all_predicted.append(1)
                else:
                    all_predicted.append(0)
                all_labels.append(0)
        print("--- Testing Abnormal Logs ---")
        with torch.no_grad():
            for line in tqdm(test_abnormal_loader, desc="Testing Abnormal"):
                session_data = abnormal_session_data_map.get(line)
                if session_data is None: continue
                session_count = session_data['count']

                metrics_vector = session_data['metrics']
                metrics_input = torch.tensor(metrics_vector, dtype=torch.float)\
                                    .view(1, self.metrics_dim).to(self.device)
                is_abnormal_session = False
                for i in range(len(line) - self.window_size):
                    seq0 = line[i : i + self.window_size]
                    label = line[i + self.window_size]

                    seq0_tensor = torch.tensor(seq0, dtype=torch.long)\
                                    .view(1, self.window_size, 1).to(self.device)
                    seq1_vector = [0] * self.num_classes
                    log_counter = Counter(seq0)
                    for key in log_counter:
                        seq1_vector[int(key)] = log_counter[key]
                    seq1_tensor = torch.tensor(seq1_vector, dtype=torch.float)\
                                        .view(1, self.num_classes, self.input_size).to(self.device)
                
                    features = []
                    if self.sequentials: features.append(seq0_tensor)
                    if self.quantitatives: features.append(seq1_tensor)
                    if self.metrics: features.append(metrics_input)

                    output = model(features=features, device=self.device)
                    predicted = torch.argsort(output, 1)[0][-self.num_candidates:]

                    if label not in predicted:
                        is_abnormal_session = True
                        break  

                if is_abnormal_session:
                    all_predicted.append(1)
                else:
                    all_predicted.append(0)
                all_labels.append(1)
                        
 
        metrics = get_metrics(np.array(all_predicted),labels = np.array(all_labels), pred=np.array(all_predicted))
        print(metrics)
        print(f"Total Normal Sessions Tested: {test_normal_length}")
        print(f"Total Abnormal Sessions Tested: {test_abnormal_length}")
        print("Finished Predicting")
        elapsed_time = time.time() - start_time
        print("elapsed_time: {}".format(elapsed_time))

    def predict_supervised(self):
        model = self.model.to(self.device)
        model.load_state_dict(torch.load(self.model_path)["state_dict"])
        model.eval()
        print("model_path: {}".format(self.model_path))
        test_logs, test_labels = session_window(self.data_dir, datatype="test")
        test_dataset = log_dataset(
            logs=test_logs,
            labels=test_labels,
            seq=self.sequentials,
            quan=self.quantitatives,
            sem=self.semantics,
            met=True,
            metrics_mean=self.metrics_mean, 
            metrics_std=self.metrics_std
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True
        )
        all_labels = []
        all_predicted = []
        tbar = tqdm(self.test_loader, desc="\r")
        for i, (log, label) in enumerate(tbar):
            features = []
            for value in log.values():
                features.append(value.clone().to(self.device))
            output = self.model(features=features, device=self.device)

            predicted = torch.argmax(output, dim=1).cpu().numpy()

            label = np.array([y.cpu() for y in label])
            all_labels.extend(label.tolist()) 
            all_predicted.extend(predicted.tolist())
        metrics = get_metrics(np.array(all_predicted),labels = np.array(all_labels), pred=np.array(all_predicted))
        print(metrics)

