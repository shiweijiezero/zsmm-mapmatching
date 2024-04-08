This is an implementation for paper <u>Zero-Shot Cellular Trajectory Map Matching</u>

---

### Overview

1. [Requirements](#requirements)
2. [Execution](#execution)
3. [Map-matching-Example](#Map-matching-Example)
4. [Dataset](#Dataset)
5. [License](#license)
6. [Contact](#contact)

---

## 1. Requirements

The following modules are required.

- Ubuntu 16.04
- Python >=3.5 (`Anaconda3` recommended)
- PyTorch 0.4 (`virtualenv` recommended)
- Cuda 9.0
- pytorch-lightning

---

## 2. Execution

### 2.1 Preprocess
```bash
$ cd src/script
$ python construct_road_map.py
```

### 2.2 Training

```bash
$ cd src/model
$ python main.py
```

## 3. Map-matching-Example

We provide some figures and experiment results to illustrate the matching of ZSMM in director `src/model`.

---

## 4. Dataset

We provide synthetic data in (https://huggingface.co/datasets/weijiezz/map-matching-dataset/tree/main).
For the real trajectory data, unfortunately, due to privacy protection, we cannot provide the dataset for testing.

---

## 5. License

The code is developed under the MPL-02.0 license.

---

## 6. Contact
If you have any questions or require further clarification, please do not hesitate to send an email to us (E-mail address：shiweijie0311@foxmail.com)
