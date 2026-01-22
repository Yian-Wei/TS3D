#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


class log_dataset(Dataset):
    def __init__(self, logs, labels, seq=True, quan=False, sem=False, met=True,
                 metrics_mean=None, metrics_std=None):
        
        self.seq = seq
        self.quan = quan
        self.sem = sem
        self.met = met

        self.metrics_mean = metrics_mean
        self.metrics_std = metrics_std
        
        if self.seq:
            self.Sequentials = logs['Sequentials']
        if self.quan:
            self.Quantitatives = logs['Quantitatives']
        if self.sem:
            self.Semantics = logs['Semantics']
        if self.met:
            self.Metrics = logs['Metrics']
        
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        log = dict()
        
        if self.seq:
            log['Sequentials'] = torch.tensor(self.Sequentials[idx],
                                              dtype=torch.long)
        if self.quan:
            log['Quantitatives'] = torch.tensor(self.Quantitatives[idx],
                                                 dtype=torch.float)
        if self.sem:
            log['Semantics'] = torch.tensor(self.Semantics[idx],
                                             dtype=torch.float)
        
        if self.met:
            raw_metrics = self.Metrics[idx]
            
            if self.metrics_mean is not None and self.metrics_std is not None:

                standardized_metrics = (raw_metrics - self.metrics_mean) / self.metrics_std
                
            else:
                standardized_metrics = raw_metrics
            
            log['Metrics'] = torch.tensor(standardized_metrics,
                                          dtype=torch.float)
            
        return log, torch.tensor(self.labels[idx], dtype=torch.long)