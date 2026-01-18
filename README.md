# TS3D

## Get Started

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

### Multimodal Alignment Window Size Parameter Study
<img width="579" height="317" alt="image" src="https://github.com/user-attachments/assets/4b02a37b-eba4-41ce-b216-a26a8fc12490" />
The size of the window was carefully chosen based on many experiments. We report in 
the Figure, the results of an experimental analysis on the same 24-hour Multi2Multi dataset to investigate the impact of different window sizes (2-10) on the performance and efficiency of multiple log-based anomaly detection methods (i.e., DeepLog, LogAnomaly, and RobustLog). As shown in Figures \ref{fig.parameter_study_rebuttal}(a)–(d), increasing the window size leads to consistent improvements in Precision, Recall, F1, and TS3D-F1 across all methods, with the most notable gains observed when the window size increases from 2 to 6. This is because larger windows capture longer event contexts, enabling models to better learn temporal dependencies and more accurately distinguish normal behaviors from anomalies. When the window size further increases to 8 or 10, the performance gains gradually saturate, indicating that excessively long contexts provide diminishing benefits and may even introduce redundant noise. From an efficiency perspective, Figure \ref{fig.parameter_study_rebuttal}(e) indicates that the overall runtime exhibits a slight decreasing trend as the window size increases. This is because, under a fixed temporal span, larger windows result in fewer generated sequence samples, thereby reducing the number of samples processed during model training. 

