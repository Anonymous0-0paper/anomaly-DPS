# **Dynamic Contrastive Self-Supervised Learning via Kolmogorov-Arnold Networks for Anomaly Detection in Multivariate Data Streams with Concept Drift**

This repository contains the implementation of a novel approach to anomaly detection in multivariate data streams. The method leverages **dynamic contrastive self-supervised learning** combined with **Kolmogorov-Arnold Networks (KANs)** to effectively detect anomalies in environments where data distributions evolve over time (**concept drift**).

## 📥 Datasets

The following datasets were used for evaluation:

### 1. KDD Cup 99
- **Description**: A widely used benchmark dataset containing network traffic with simulated intrusions.
- [📥 Download Dataset](https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)

---

### 2. IoTID20
- **Description**: Features various attacks on IoT devices in home and office environments.
- [📥 Download Dataset](https://www.kaggle.com/datasets/rohulaminlabid/iotid20-dataset)

---

### 3. WUSTL IIoT 2021
- **Description**: Captures realistic network traffic in an industrial IoT environment.
- [📥 Download Dataset](https://ieee-dataport.org/documents/wustl-iiot-2021)

---

### 4. UNSW-NB15
- **Description**: Combines features of normal behavior and modern attack patterns in network traffic.
- [📥 Download Dataset](https://unsw-my.sharepoint.com/personal/z5025758_ad_unsw_edu_au/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fz5025758%5Fad%5Funsw%5Fedu%5Fau%2FDocuments%2FUNSW%2DNB15%20dataset&ga=1)

---

### 5. MSL (Mars Science Laboratory)
- **Description**: Multivariate time series data collected from NASA spacecraft sub-systems, used for anomaly detection.
- [📥 Download Dataset](https://pds-atmospheres.nmsu.edu/data_and_services/atmospheres_data/Mars/Mars.html)

---

### 6. SMAP (Soil Moisture Active Passive)
- **Description**: Sensor data from NASA’s Earth satellite missions; includes labeled anomalies.
- [📥 Download Dataset](https://smap.jpl.nasa.gov/data/)

---

### 7. SMD (Server Machine Dataset)
- **Description**: Real-world server monitoring data with labeled anomalies from a large internet company.
- [📥 Download Dataset](https://github.com/NetManAIOps/OmniAnomaly)

---

### 8. WADI (Water Distribution (Drinking) Testbed for Anomaly Detection)
- **Description**: Industrial water distribution testbed with labeled cyber-physical attacks.
- [📥 Download Dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#wadi)

---

### 9. SWaT (Secure Water Treatment)
- **Description**: Dataset from a fully operational water treatment testbed with real cyber-physical attacks.
- [📥 Download Dataset](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/#swat)

---

## 🛠️ Features

- Dynamic contrastive learning for evolving data streams  
- Integration of Kolmogorov-Arnold Networks for improved model expressiveness  
- Real-time adaptation to concept drift  
- Suitable for high-dimensional, multivariate time-series anomaly detection  

---
## Project Structure

| Top-level Folder/File | Main Function Description | Main Files Contained |
|-----------------------|---------------------------|----------------------|
| **checkpoints** | Stores temporary files during model training | - |
| **data_provider** | Stores data processing programs | `data_loader.py`, `data_factory.py`, `uea.py` |
| **dataset** | Stores model training data | - |
| **exp** | Stores time series task training modules (anomaly detection, filling missing values, etc.) | `exp_basic.py`, `exp_anomaly_detection` |
| **layers** | Stores basic architecture components of the model (encoder/decoder, attention mechanism, embedding layer) | `AutoCorrelation.py`, `Autoformer_EncDec.py`, `Conv_Blocks.py`, `Crossformer_EncDec.py`, `Embed.py`, etc. |
| **models** | Stores the implementation of time series prediction models | `Autoformer.py`, `TimesNet.py`, `Transformer.py`, etc. |
| **performance_charts** | Stores training results | - |
| **scripts** | Stores Shell scripts for automated execution | Multiple `.sh` script files |
| **utils** | Stores utility functions and auxiliary modules | `masking.py`, `print_args.py`, `tools.py` |
| **LICENSE** | Copyright declaration file | - |
| **README.md** | Project documentation | - |
| **requirements.txt** | Project dependency environment configuration | - |
| **run.py** | Main program (parameter setting/model training/test entry) | - |

---
## Execution

```bash
# Clone the repository
git clone git@github.com\:Anonymous0-0paper/anomaly-DPS.git
cd anomaly-DPS
# Install dependencies
pip install -r requirements.txt
```

### Database Source
The dataset is stored in the `dataset` folder on [Google Drive](https://drive.google.com/drive/folders/17rV8ahgAC2RDt6m6e2pRmfOIt_NLAvk0) and needs to be transplanted to the project root directory `anomaly-DPS`.

### Script Execution
All running scripts are located in the `scripts/` directory. Execution example:
```bash
sh scripts/anomaly_detection/MSL/Transformer.sh
```

### Key Model Parameter Descriptio
The core configuration parameters of `run.py` are as follows:

| Parameter Name   | Parameter Meaning     |
|-------|------|
| task_name | 	Task type     |
| is_training| Whether it is training mode     |
| root_path | Dataset root directory path     |
| data_path | 	Data filename     |
| model_id | Model unique identifier (MSL, SMAP, MSD, SWAT, WADI) |
| model | Time series model name (Transformer, TimesNets, etc.) |
| data  | Data processing method |
| features | Prediction mode (M, S, MS) |
| seq_len | Historical time step length (encoder input length） |
| pred_len | Future time step length to be predicted |
| e_layers | Number of encoder layers |
| d_layers | Number of decoder layers |
| enc_in | Encoder input feature dimension |
| dec_in | Decoder input feature dimension |
| c_out | Output feature dimension|


### Developing Your Own Model
Follow these steps:

1. Add the model file to the `models` folder.
2. In the `exp/exp_basic.py` file, include the newly added model in `Exp_Basic.model_dict`.
3. Create the corresponding `scripts` in the scripts folder. 

## Run In the Cluster

### Environment Setup
```shell

git clone https://github.com/Anonymous0-0paper/anomaly-DPS.git

python3 -m venv ~/project/venvs/anomaly-DPS

source ~/project/venvs/anomaly-DPS/bin/activate

pip install --upgrade pip

pip install -r requirements.txt
```

### Model Execution
`run_cluster.sh` is a cluster running script that supports the parallel execution of multiple models on the same dataset, ensuring that the number of available nodes is greater than the number of scripts.
```shell

chmod +x run_cluster.sh

./run_cluster.sh
```
### Logs
Model execution logs are stored in `anomaly-DPS/logs`, GPU memory records are stored in `anomaly-DPS/logs/dataset_name/gpu_mem`.