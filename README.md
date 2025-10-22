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

### Options
You can customize the model and training configurations by modifying the options dictionary in the code.
For example:
```
options["metrics"] = True  
options["metrics_dim"] = 29
options["mask_fnn"] = 1
```
