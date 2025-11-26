# print_car_ids_from_name_parallel.py
import numpy as np
import os
import pickle
from glob import glob
import torch
import matplotlib.pyplot as plt
import pandas as pd
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

current_dir = os.getcwd()

train_path = r'/Workspace/Users/pssx_a_yifshen@corpdir.partner.onmschina.cn/VAE/pkl_normal'
test_path  = r'/Workspace/Users/pssx_a_yifshen@corpdir.partner.onmschina.cn/VAE/pkl_abnormal'

train_pkl_files = glob(train_path+'/*.pkl')
test_pkl_files  = glob(test_path +'/*.pkl')

ind_pkl_files = []
ood_pkl_files = []
car_num_list = []

ood_car_num_list = set()
ind_car_num_list = set()

all_car_dict = {}

def parse_from_name(each_path: str):
    name  = os.path.splitext(os.path.basename(each_path))[0]
    parts = name.split('_')
    if len(parts) < 2:
        return None
    label = parts[0]                  # '0' 正常, '1' 故障
    car   = f"{parts[0]}_{parts[1]}"  # 车号形如 0_0001
    return car, label, each_path

# —— 并行解析文件名 + 进度条 —— #
all_files = train_pkl_files + test_pkl_files
max_workers = min(26, (os.cpu_count() or 8) * 2)  # 可按磁盘性能调大/调小
with ThreadPoolExecutor(max_workers=max_workers) as ex:
    for res in tqdm(ex.map(parse_from_name, all_files, chunksize=4096),
                    total=len(all_files),
                    desc="Indexing PKL files", unit="file", dynamic_ncols=True):
        if res is None:
            continue
        car, label, each_path = res

        if str(label) == '0':
            ind_pkl_files.append(each_path)
            ind_car_num_list.add(car)
        else:
            ood_pkl_files.append(each_path)
            ood_car_num_list.add(car)

        car_num_list.append(car)
        if car not in all_car_dict:
            all_car_dict[car] = [each_path]
        else:
            all_car_dict[car].append(each_path)

# —— 只按升序，不打乱 —— #
ind_sorted = sorted(ind_car_num_list)
print(ind_sorted)
ood_sorted = sorted(ood_car_num_list)
print(ood_sorted)
print(ind_car_num_list, len(ind_car_num_list))
print(ood_car_num_list, len(ood_car_num_list))

# —— 保存方式与路径保持完全一致 —— #
ind_odd_dict = {}
ind_odd_dict["ind_sorted"], ind_odd_dict["ood_sorted"] = ind_sorted, ood_sorted
os.makedirs('../five_fold_utils', exist_ok=True)
np.save('../five_fold_utils/ind_odd_dict1.npz', ind_odd_dict)

# save all the three brands path information（保持一致）
np.save('../five_fold_utils/all_car_dict.npz', all_car_dict)
