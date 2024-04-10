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

---

## 3. Dataset

We provide synthetic data in (https://huggingface.co/datasets/weijiezz/map-matching-dataset/tree/main).
For the real trajectory data, unfortunately, due to privacy protection, we cannot provide the dataset for testing.

---

## 4. License

The code is developed under the MPL-02.0 license.

---

## 5. Contact
Due to Anonymous Policy, we could not provide any contact information.
