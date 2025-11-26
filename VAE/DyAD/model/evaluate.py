# import os
# import sys
# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
#
# class Evaluate:
#     def __init__(self, args):
#         self.args = args
#
#     @staticmethod
#     def flatten_list(v, repeat_len):
#         # 标量转list，单元素ndarray/list转值，否则原样
#         if isinstance(v, torch.Tensor):
#             v = v.cpu().numpy()
#         if isinstance(v, (float, int, np.floating, np.integer)):
#             return [v] * repeat_len
#         if isinstance(v, str):
#             return [v] * repeat_len
#         if isinstance(v, (list, np.ndarray)):
#             v = np.array(v)
#             if v.ndim == 0:
#                 return [v.item()] * repeat_len
#             elif v.ndim == 1:
#                 if len(v) == repeat_len:
#                     return list(v)
#                 elif len(v) == 1:
#                     return [v[0]] * repeat_len
#                 else:
#                     # 异常长度，强制截断或填充
#                     if len(v) > repeat_len:
#                         return list(v[:repeat_len])
#                     else:
#                         return list(v) + [v[-1]] * (repeat_len - len(v))
#             else:
#                 # 更高维度，取第一维
#                 v = v.flatten()
#                 if len(v) >= repeat_len:
#                     return list(v[:repeat_len])
#                 else:
#                     return list(v) + [v[-1]] * (repeat_len - len(v))
#         # 其他类型
#         return [v] * repeat_len
#
#     @staticmethod
#     def get_feature_label(data_path, max_group=None):
#         data, label = [], []
#         print(f"Data path: {data_path}")
#         try:
#             file_list = os.listdir(data_path)
#             print("Files in the directory:", file_list)
#         except Exception as e:
#             print(f"Error reading directory: {e}")
#             return np.array(data), np.array(label)
#         for f in tqdm(sorted(os.listdir(data_path), key=lambda x: x.split('_')[0])):
#             if f.endswith(".file"):
#                 temp_label = torch.load(open(os.path.join(data_path, f), 'rb'))
#                 num = len(temp_label['rec_error'])
#                 car_list = Evaluate.flatten_list(temp_label.get('car'), num)
#                 mileage_list = Evaluate.flatten_list(temp_label.get('mileage', [None] * num), num)
#                 label_list = Evaluate.flatten_list(temp_label.get('label'), num)
#                 rec_error_list = Evaluate.flatten_list(temp_label.get('rec_error'), num)
#                 # 新增：尝试读取 soc_range / volt_range；若不存在则用 None 填充
#                 soc_range_list = Evaluate.flatten_list(temp_label.get('soc_range', [None] * num), num)
#                 volt_range_list = Evaluate.flatten_list(temp_label.get('volt_range', [None] * num), num)
#                 # 一对一输出（对齐顺序）
#                 for c, m, l, r, sr, vr in zip(car_list, mileage_list, label_list, rec_error_list,
#                                               soc_range_list, volt_range_list):
#                     # 为了防止列表/元组写入 CSV 变成不可读字符串，这里统一成 "a-b" 或原样
#                     def norm_span(x):
#                         if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2:
#                             try:
#                                 return f"{float(x[0])}-{float(x[1])}"
#                             except Exception:
#                                 return f"{x[0]}-{x[1]}"
#                         return x
#
#                     label.append([c, m, l, r, norm_span(sr), norm_span(vr)])
#             elif f.endswith(".npy"):
#                 data += np.load(os.path.join(data_path, f)).tolist()
#         print(f"Data shape: {np.array(data).shape}")
#         print(f"Label shape: {np.array(label).shape}")
#         return np.array(data), np.array(label, dtype=object)
#
#     @staticmethod
#     def calculate_rec_error(_, label):
#         # 列结构现在是：['car','mileage','label','rec_error','soc_range','volt_range']
#         rec_sorted_index = np.argsort(-label[:, 3].astype(float))  # 仍按重构误差降序
#         res = [label[i] for i in rec_sorted_index]
#         return pd.DataFrame(res, columns=['car', 'mileage', 'label', 'rec_error', 'soc_range', 'volt_range'])
#
#     def main(self):
#         x, label = self.get_feature_label(self.args.feature_path, max_group=20000)
#         print("Loading feature is :", x.shape)
#         print("Loading label is :", label.shape)
#         result = eval('self.calculate_' + self.args.use_flag)(x, label)
#         result.to_csv(os.path.join(self.args.result_path, "train_segment_scores.csv"), index=False)
#
#         x, label = self.get_feature_label(self.args.save_feature_path, max_group=20000)
#         print("Loading test feature is :", x.shape)
#         print("Loading test label is :", label.shape)
#         result = eval('self.calculate_' + self.args.use_flag)(x, label)
#         result.to_csv(os.path.join(self.args.result_path, "test_segment_scores.csv"), index=False)
#
# if __name__ == '__main__':
#     import argparse
#     import json
#
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     parser = argparse.ArgumentParser(description='Train Example')
#     parser.add_argument('--modelparams_path', type=str,
#                         default=os.path.join(r'C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\2025-11-06-21-49-53_fold0\model', 'model_params.json'))
#
#     args = parser.parse_args()
#
#     with open(args.modelparams_path, 'r', encoding='utf-8') as file:
#         p_args = argparse.Namespace()
#         model_params = json.load(file)
#         p_args.__dict__.update(model_params["args"])
#         args = parser.parse_args(namespace=p_args)
#     print("Loaded configs at %s" % args.modelparams_path)
#     print("args", args)
#     Evaluate(args).main()


import os
import sys
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
class Evaluate:
    def __init__(self, args):
        self.args = args
    @staticmethod
    def flatten_list(v, repeat_len):
        # 标量转list，单元素ndarray/list转值，否则原样
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if isinstance(v, (float, int, np.floating, np.integer)):
            return [v] * repeat_len
        if isinstance(v, str):
            return [v] * repeat_len
        if isinstance(v, (list, np.ndarray)):
            v = np.array(v)
            if v.ndim == 0:
                return [v.item()] * repeat_len
            elif v.ndim == 1:
                if len(v) == repeat_len:
                    return list(v)
                elif len(v) == 1:
                    return [v[0]] * repeat_len
                else:
                    # 异常长度，强制截断或填充
                    if len(v) > repeat_len:
                        return list(v[:repeat_len])
                    else:
                        return list(v) + [v[-1]] * (repeat_len - len(v))
            else:
                # 更高维度，取第一维
                v = v.flatten()
                if len(v) >= repeat_len:
                    return list(v[:repeat_len])
                else:
                    return list(v) + [v[-1]] * (repeat_len - len(v))
        # 其他类型
        return [v] * repeat_len
    @staticmethod
    def get_feature_label(data_path, max_group=None):
        data, label = [], []
        # 打印数据路径，确保路径正确
        print(f"Data path: {data_path}")
        # 打印目录下所有文件，确保文件存在
        try:
            file_list = os.listdir(data_path)
            print("Files in the directory:", file_list) # 打印目录下的文件
        except Exception as e:
            print(f"Error reading directory: {e}")
            return np.array(data), np.array(label)
        for f in tqdm(sorted(os.listdir(data_path), key=lambda x: x.split('_')[0])):
            if f.endswith(".file"):
                temp_label = torch.load(open(os.path.join(data_path, f), 'rb'))
                num = len(temp_label['rec_error'])
                car_list = Evaluate.flatten_list(temp_label['car'], num)
                mileage_list = Evaluate.flatten_list(temp_label.get('mileage', [None]*num), num)
                label_list = Evaluate.flatten_list(temp_label['label'], num)
                rec_error_list = Evaluate.flatten_list(temp_label['rec_error'], num)
                # 新增：尝试读取 soc_range / volt_range；若不存在则用 None 填充
                soc_range_list = Evaluate.flatten_list(temp_label.get('soc_range', [None]*num), num)
                volt_range_list = Evaluate.flatten_list(temp_label.get('volt_range', [None]*num), num)
                # 一对一输出（对齐顺序）
                for c, m, l, r, sr, vr in zip(car_list, mileage_list, label_list, rec_error_list,
                                              soc_range_list, volt_range_list):
                    # 为了防止列表/元组写入 CSV 变成不可读字符串，这里统一成 "a-b" 或原样
                    def norm_span(x):
                        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == 2:
                            try:
                                return f"{float(x[0])}-{float(x[1])}"
                            except Exception:
                                return f"{x[0]}-{x[1]}"
                        return x
                    label.append([c, m, l, r, norm_span(sr), norm_span(vr)])
            elif f.endswith(".npy"):
                data += np.load(os.path.join(data_path, f)).tolist()
        print(f"Data shape: {np.array(data).shape}")
        print(f"Label shape: {np.array(label).shape}")
        return np.array(data), np.array(label, dtype=object)
    @staticmethod
    def calculate_rec_error(_, label):
        # 列结构现在是：['car','mileage','label','rec_error','soc_range','volt_range']
        rec_sorted_index = np.argsort(-label[:, 3].astype(float)) # 仍按重构误差降序
        res = [label[i] for i in rec_sorted_index]
        return pd.DataFrame(res, columns=['car', 'mileage', 'label', 'rec_error', 'soc_range', 'volt_range'])
    def main(self):
        x, label = self.get_feature_label(self.args.feature_path, max_group=20000)
        print("Loading feature is :", x.shape)
        print("Loading label is :", label.shape)
        result = eval('self.calculate_' + self.args.use_flag)(x, label)
        # 使用 Parquet 格式存储结果
        result.to_parquet(os.path.join(self.args.result_path, "train_segment_scores.parquet"), index=False)
        print(f"Train segment scores saved to {os.path.join(self.args.result_path, 'train_segment_scores.parquet')}")
        x, label = self.get_feature_label(self.args.save_feature_path, max_group=20000)
        print("Loading test feature is :", x.shape)
        print("Loading test label is :", label.shape)
        result = eval('self.calculate_' + self.args.use_flag)(x, label)
        # 使用 Parquet 格式存储结果
        result.to_parquet(os.path.join(self.args.result_path, "test_segment_scores.parquet"), index=False)
        print(f"Test segment scores saved to {os.path.join(self.args.result_path, 'test_segment_scores.parquet')}")
if __name__ == '__main__':
    import argparse
    import json
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--modelparams_path', type=str,
                        default=os.path.join(r'C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\2025-11-12-01-54-20_fold0\model', 'model_params.json'))
    args = parser.parse_args()
    with open(args.modelparams_path, 'r', encoding='utf-8') as file:
        p_args = argparse.Namespace()
        model_params = json.load(file)
        p_args.__dict__.update(model_params["args"])
        args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.modelparams_path)
    print("args", args)
    Evaluate(args).main()