import os
import time

import torch
import torch.nn.functional as F
import torch.nn as nn
import pytorch_lightning as pl
import tqdm
from model import MyModel
import matplotlib.pyplot as plt
import util
import yaml

# 初始化参数
device="cpu"
config_dir = 'config.yaml'
with open(config_dir) as fin:
    config = yaml.safe_load(fin)
myutil = util.UtilClass()

# 加载测试数据
with open(f"./data/save_output/src_tensor","rb") as f:
    src_tensor=torch.load(f,map_location=device)
with open(f"./data/save_output/output_tensor", "rb") as f:
    output_tensor=torch.load(f,map_location=device)

# 开始测试
precision, recall = myutil.get_acc(config=config,
                                                  src_tensor=src_tensor,
                                                  output_tensor=output_tensor,
                                                  batch_idx=0,
                                                  batch_size=
                                                  # 30,
                                                  src_tensor.shape[0],
                                                  data_type="val",
                                                  save_pic=False)
with open("./validation experiment result.txt", "w") as f:
    f.write(f"precision:{precision}, recall:{recall}")
print(f"precision:{precision}, recall:{recall}")