# TS3D


TS3D is a temporal multimodal dataset derived from a distributed database, with 300 million numerical data points and 90 million contextual data records. The dataset has been tested on two representative downstream tasks anomaly detection and root cause discovery. Technical details can be found in the paper [TS3D: A Temporal Multimodal Dataset for
Distributed Database System Analysis]([https://www.vldb.org/pvldb/vol16/p3363-khelifati.pdf](https://icde2026.github.io)), ICDE'26. 

The project is structured as follows:

- Multimodal Anomaly Detection (TS3D/MultimodalAD): builds on existing log- and metric-based backbones, and adds a lightweight fusion module plus a self-adaptive masking strategy.Fusion Module: bidirectional cross-attention + feature fusion between log and numerical embeddings.Self-adaptive Mask: dynamically selects informative numerical features by masking less useful gradients during training.
- Anomaly Detection Metrics TS3D-F1 (TS3D/MultimodalAD/ts3d/metrics): Standard metrics and TS3D-F1.
- Root Cause Analysis (TS3D/RootDiscovery): builds dependency graphs (e.g., table-dependency graphs of timeout SQL templates) for root analysis.

## Getting Started

### Get Data
Download TS3D datasets from [here](https://drive.google.com/drive/folders/1DptJNiqgXF4DlZ5rx-FXRKz_NaMyS34t)

### Data Preparation
After downloading the raw dataset, please preprocess it into the same format as the examples in 
```
data/s2s/
```
You can either preprocess it manually or use the provided script to automate the process.
```
python merge.py
python merge2.py
python datapre.py
```
### Install dependencies
1. Clone this repository.
```
git clone https://github.com/Yian-Wei/TS3D.git
```
2. Create a new Conda environment.
```
conda create -n ts3d python=3.12
conda activate ts3d
```
3. Install Core Dependencies
```
pip install -r requirements.txt
```

### Run TS3D
```
cd demo
# Train
python deeplog.py train
# Test
python deeplog.py test
```
Alternatively, you can run the shell script.
```
sh run.sh
```

### Options
You can customize the model and training configurations by modifying the options dictionary in the code.
For example:
```
options["metrics"] = True  
options["metrics_dim"] = 29
options["mask_fnn"] = 1
```
## Experimental Results

### Multimodal Alignment Window Size Parameter Study
<img width="982" height="582" alt="image" src="https://github.com/user-attachments/assets/48a10efd-ed5d-4d84-a258-762cdfe642f4" />

The size of the window was carefully chosen based on many experiments. We report in 
Figure 1, the results of an experimental analysis on the same 24-hour Multi2Multi dataset to investigate the impact of different window sizes (2-10) on the performance and efficiency of multiple log-based anomaly detection methods (i.e., DeepLog, LogAnomaly, and RobustLog). As shown in Figures 1 (a)–(d), increasing the window size leads to consistent improvements in Precision, Recall, F1, and TS3D-F1 across all methods, with the most notable gains observed when the window size increases from 2 to 6. This is because larger windows capture longer event contexts, enabling models to better learn temporal dependencies and more accurately distinguish normal behaviors from anomalies. When the window size further increases to 8 or 10, the performance gains gradually saturate, indicating that excessively long contexts provide diminishing benefits and may even introduce redundant noise. From an efficiency perspective, Figure 1 (e) indicates that the overall runtime exhibits a slight decreasing trend as the window size increases. This is because, under a fixed temporal span, larger windows result in fewer generated sequence samples, thereby reducing the number of samples processed during model training. 


## Contributors

- Yuanyuan Yao (contact person yoyoyao@zju.edu.cn, yoyo185644@163.com)

