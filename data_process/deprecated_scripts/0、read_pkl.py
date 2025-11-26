# Databricks notebook source
import torch
import numpy as np
from numpy.core.multiarray import _reconstruct
from numpy import ndarray, dtype  # 新增导入 dtype

def read_pkl_file(file_path):
    """读取由 slice_and_save_from_parquet_v6_infer 生成的 pkl 文件"""
    if file_path.startswith("dbfs:/"):
        local_path = "/dbfs" + file_path[5:]
    else:
        local_path = file_path
    
    try:
        # 同时允许 _reconstruct、ndarray 和 dtype 全局变量
        with torch.serialization.safe_globals([_reconstruct, ndarray, dtype]):
            win_np, meta = torch.load(local_path, weights_only=True)
        
        print("元数据信息:")
        for key, value in meta.items():
            print(f"  {key}: {value}")
        
        print(f"\n特征数据形状: {win_np.shape}")
        print(f"特征数据前 2 行示例:\n{win_np[:2]}")
        
        return win_np, meta
    
    except Exception as e:
        print(f"读取文件失败: {e}")
        # 尝试完全禁用 weights_only（仅信任文件时使用）
        try:
            print("尝试使用 weights_only=False 加载...")
            win_np, meta = torch.load(local_path, weights_only=False)
            # 验证加载结果
            print("加载成功（weights_only=False）")
            print("元数据信息:")
            for key, value in meta.items():
                print(f"  {key}: {value}")
            print(f"\n特征数据形状: {win_np.shape}")
            return win_np, meta
        except Exception as e2:
            print(f"再次失败: {e2}")
            return None, None

# 读取文件
file_path = "/Workspace/Users/pssx_a_yifshen@corpdir.partner.onmschina.cn/VAE/pkl_abnormal/1_0001_10_262.pkl"
win_data, metadata = read_pkl_file(file_path)