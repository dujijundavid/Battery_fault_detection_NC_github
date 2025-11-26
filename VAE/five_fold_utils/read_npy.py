import numpy as np

# 读取 .npy 文件，允许加载 Python 对象
data = np.load(r'C:\Users\YIFSHEN\Documents\VAE\data\five_fold_utils\all_car_dict.npz.npy', allow_pickle=True)

# 查看数据
print(data)
