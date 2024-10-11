# Zero-Shot Cellular Trajectory Map Matching

This repository contains the implementation for the paper "Zero-Shot Cellular Trajectory Map Matching".

## Table of Contents

1. [Requirements](#requirements)
2. [Execution](#execution)
3. [Dataset](#dataset)
4. [License](#license)
5. [Contact](#contact)

## 1. Requirements

The following environment and modules are required:

- Ubuntu 16.04 or later
- Python 3.5 or later (Anaconda3 recommended)
- PyTorch 0.4 (virtualenv recommended)
- CUDA 9.0
- pytorch-lightning

## 2. Execution

### 2.1 Preprocessing

To preprocess the data, run the following commands:

```bash
cd src/script
python construct_road_map.py
```

### 2.2 Training
To train the model, execute:
```bash
cd src/model
python main.py
```
## 3. Dataset
We provide synthetic data for testing and evaluation. You can access it at our Hugging Face dataset repository.
For real trajectory data, we regret that due to privacy protection regulations, we cannot provide the dataset for testing.
## 4. License
This project is licensed under the MPL-2.0 License. 
## 5. Contact
Due to the anonymous submission policy, we cannot provide contact information at this time. If you have any questions or concerns, please open an issue in this repository.