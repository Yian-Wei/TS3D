#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys

sys.path.append("../")

from logdeep.models.lstm import deeplog, loganomaly, robustlog
from logdeep.tools.predict import Predicter
from logdeep.tools.train import Trainer
from logdeep.tools.utils import *


# Config Parameters

options = dict()
options["data_dir"] = "../data/"
options["window_size"] = 10
options["device"] = "cpu"

# Smaple
options["sample"] = "sliding_window"
options["window_size"] = 10  # if fix_window

# Features
options["sequentials"] = True
options["quantitatives"] = False
options["semantics"] = False
options["metrics"] = True  
options["metrics_dim"] = 29 

# Model
options["input_size"] = 1
options["hidden_size"] = 64
options["num_layers"] = 2
options["num_classes"] = 32

# Train
options["batch_size"] = 2048
options["accumulation_step"] = 1

options["optimizer"] = "adam"
options["lr"] = 0.001
options["max_epoch"] = 370
options["lr_step"] = (300, 350)
options["lr_decay_ratio"] = 0.1

options["resume_path"] = None
options["model_name"] = "deeplog"
options["save_dir"] = "../result/deeplog/"

# Predict
options["model_path"] = "../result/deeplog/deeplog_last.pth"
options["num_candidates"] = 3

options["mask_fnn"] = 1 
options["gn_mode"] = "gn_wn" #  "gn", "gn_wn", "sqrt_gn"
options["sample_mode"] = "Adaptive" #  "TopK", "random", "Adaptive"
options["keep_ratio"] = 0.3 

seed_everything(seed=1234)


def train():
    Model = deeplog(
        num_templates=32,     
        embedding_dim=64,     
        hidden_size=128,
        num_layers=2,
        num_keys=32,
        num_metrics=29
    )
    trainer = Trainer(Model, options)
    trainer.start_train(options)


def predict():
    Model = deeplog(
        num_templates=32,    
        embedding_dim=64,     
        hidden_size=128,
        num_layers=2,
        num_keys=32,
        num_metrics=29
    )
    predicter = Predicter(Model, options)
    predicter.predict_unsupervised()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "predict"])
    args = parser.parse_args()
    if args.mode == "train":
        train()
    else:
        predict()
