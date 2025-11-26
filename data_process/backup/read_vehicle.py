#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
# ====== 1. 配置：改成你的“总文件夹”路径 ======
ROOT_DIR = r"C:\Users\YIFSHEN\Documents\01_InputRawData\top_500_mileage2" # TODO：这里改成包含所有车辆子文件夹的路径
def collect_car_ids(root_dir: str):
    """
    从 root_dir 下读取所有子文件夹名，认为子文件夹名就是车辆编号（如 0_0921）
    返回一个排序后的列表
    """
    if not os.path.isdir(root_dir):
        print(f"[ERROR] 目录不存在：{root_dir}")
        return []
    car_ids = []
    for name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, name)
        if os.path.isdir(full_path):
            # 这里默认所有子文件夹名都是车辆编号，比如 0_0921
            car_ids.append(name)
    # 去重 + 排序
    car_ids = sorted(set(car_ids))
    return car_ids
def main():
    car_ids = collect_car_ids(ROOT_DIR)
    if not car_ids:
        print("[INFO] 没有找到任何子文件夹，或者目录不存在。")
        return
    print(f"[INFO] 共找到 {len(car_ids)} 个车辆编号。")
    # 1) 按你需要的格式：'0_0921','0_0922',...
    joined_str = ",".join(f"'{cid}'" for cid in car_ids)
    print("\n====== 复制下面这一行，用作删除 pkl 的车辆编号列表（无括号版）======")
    print(joined_str)
    # 2) 也顺便给一个 Python 列表格式，方便直接粘贴到代码里：
    list_repr = "[" + ", ".join(f"'{cid}'" for cid in car_ids) + "]"
    print("\n====== 这是 Python 列表格式，可以直接粘到 TARGET_CARS 里 ======")
    print(list_repr)
if __name__ == "__main__":
    main()