from pathlib import Path
import numpy as np
import pickle
import pandas as pd

def load_pkl(path: str):
    """优先用 torch.load 读取，与保存方式一致；失败则回退到 pickle.load。"""
    try:
        import torch
        return torch.load(path, map_location="cpu")
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)

path = r"C:\Users\YIFSHEN\Documents\01_InputRawData\532_test_pkl_abnormal_new\1_0001_1_0.pkl"   # 替换为你的文件路径

X, meta = load_pkl(path)

# 统一为 numpy
if hasattr(X, "numpy"):
    X = X.numpy()
elif not isinstance(X, np.ndarray):
    X = np.asarray(X)

# 取前100行（若不足100行，则全取）
n = min(100, X.shape[0])
head = X[:n]

# 构建DataFrame方便查看
colnames = [f"f{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(head, columns=colnames)

print("文件：", path)
print("数组形状：", X.shape)
print("展示行数：", n)
print("meta：", meta)
print(df.head(40))  # 控制台先看前10行

# 可选：导出预览为CSV
df.to_csv("preview_head100.csv", index=False)
