# import json
# import os
# import sys
# import time
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from utils import collate
# from model import dataset
# from train import extract
# from utils import to_var, collate, Normalizer, PreprocessNormalizer
# from model import tasks
# import pickle
#
# class Extraction:
#     """
#     feature extraction
#     """
#
#     def __init__(self, args, fold_num=0):
#         """
#         :param project: class model.projects.Project object
#         """
#         self.args = args
#         self.fold_num = fold_num
#
#     def main(self):
#         """
#         test: normalized test data
#         task: task, e.g. EvTask、JeveTask
#         model: model
#         """
#         model_params_path = os.path.join(self.args.current_model_path, "model_params.json")
#         with open(model_params_path, 'r', encoding='utf-8') as load_f:
#             prams_dict = json.load(load_f)
#         model_params = prams_dict['args']
#         start_time = time.time()
#         # data_pre = dataset.Dataset(model_params["test_path"])
#         data_pre = dataset.Dataset(model_params["test_path"], train=False, fold_num=self.fold_num)
#         self.normalizer = pickle.load(open(os.path.join(self.args.current_model_path, "norm.pkl"), 'rb'))
#         test = PreprocessNormalizer(data_pre, normalizer_fn=self.normalizer.norm_func)
#
#         task = tasks.Task(task_name=model_params["task"], columns=model_params["columns"])
#
#         # load checkpoint
#         #model_torch = os.path.join(model_params["current_model_path"], "model.torch")
#         # 改用你传入的 --current_model_path 作为模型路径
#         model_torch = os.path.join(self.args.current_model_path, "model.torch")
#
#         model = to_var(torch.load(model_torch)).float()
#         model.encoder_filter = task.encoder_filter
#         model.decoder_filter = task.decoder_filter
#         model.noise_scale = model_params["noise_scale"]
#         data_loader = DataLoader(dataset=test, batch_size=model_params["batch_size"], shuffle=True,
#                                  num_workers=model_params["jobs"], drop_last=False,
#                                  pin_memory=torch.cuda.is_available(),
#                                  collate_fn=collate if model_params["variable_length"] else None)
#
#         print("sliding windows dataset length is: ", len(test))
#         print("model", model)
#
#         # extact feature
#         model.eval()
#         p_bar = tqdm(total=len(data_loader), desc='saving', ncols=100, mininterval=1, maxinterval=10, miniters=1)
#         extract(data_loader, model, task, model_params["save_feature_path"], p_bar, model_params["noise_scale"],
#                 model_params["variable_length"])
#         p_bar.close()
#         print("Feature extraction of all test saved at", model_params["save_feature_path"])
#         print("The total time consuming：", time.time() - start_time)
#
#
# if __name__ == '__main__':
#     import argparse
# #gpu加速，一个gpu设置为0
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     parser = argparse.ArgumentParser(description='Train Example')
#     parser.add_argument('--current_model_path', type=str,
#                         default=r'C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\91_vehicle_train\model')
#     args = parser.parse_args()
#     Extraction(args).main()



import os
import json
import time
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import collate, to_var, Normalizer, PreprocessNormalizer # Normalizer 仅为类型引用，不在测试拟合
from model import dataset
from model import tasks
from train import extract # 复用你训练里的 extract(batch...) 保存特征
class Extraction:
    """
    Feature extraction on test set using the normalizer fitted on training set.
    """
    def __init__(self, args, fold_num=0):
        self.args = args
        self.fold_num = fold_num
    def main(self):
        # === 读取训练时保存的参数（保证模型与norm匹配同一轮训练） ===
        model_params_path = os.path.join(self.args.current_model_path, "model_params.json")
        with open(model_params_path, 'r', encoding='utf-8') as f:
            params_dict = json.load(f)
        model_params = params_dict['args']
        start_time = time.time()
        # === 构造测试集（惰性读取），并应用训练阶段的 normalizer 变换 ===
        data_pre = dataset.Dataset(model_params["test_path"], train=False, fold_num=self.fold_num)
        normalizer_path = os.path.join(self.args.current_model_path, "norm.pkl")
        with open(normalizer_path, "rb") as f:
            normalizer = pickle.load(f)
        test_ds = PreprocessNormalizer(data_pre, normalizer_fn=normalizer.norm_func)
        # === 构造任务（列索引/encoder/decoder/target 过滤） ===
        task = tasks.Task(task_name=model_params["task"], columns=model_params["columns"])
        # === 加载模型到设备，eval + 关闭梯度 ===
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model_torch = os.path.join(self.args.current_model_path, "model.torch")
        model = torch.load(model_torch, map_location=device).float().eval()
        model.encoder_filter = task.encoder_filter
        model.decoder_filter = task.decoder_filter
        model.noise_scale = model_params["noise_scale"]
        # === DataLoader（测试阶段更稳：不shuffle、较小prefetch、关persistent） ===
        batch_size_eval = model_params.get("batch_size_eval", model_params["batch_size"])
        num_workers_eval = min(8, int(model_params.get("jobs", 0))) # 测试阶段不必开太多
        dl_kwargs = dict(
            dataset=test_ds,
            batch_size=batch_size_eval,
            shuffle=False, # 测试/提取不打乱
            num_workers=num_workers_eval,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate if model_params["variable_length"] else None
        )
        if num_workers_eval > 0:
            dl_kwargs.update(dict(
                prefetch_factor=2,
                persistent_workers=False
            ))
        data_loader = DataLoader(**dl_kwargs)
        print("sliding windows dataset length:", len(test_ds))
        print("model:", model.__class__.__name__, "on", device)
        # === 批量提取特征：无梯度，更省显存 ===
        p_bar = tqdm(total=len(data_loader), desc='saving', ncols=100, mininterval=1, maxinterval=10, miniters=1)
        with torch.inference_mode():
            # 如需进一步提速/省显存，可启用 AMP（一般安全）：
            # with torch.cuda.amp.autocast():
            extract(
                data_loader=data_loader,
                model=model.to(device),
                data_task=task,
                feature_path=model_params["save_feature_path"],
                p_bar=p_bar,
                noise_scale=model_params["noise_scale"],
                variable_length=model_params["variable_length"]
            )
        p_bar.close()
        print("Feature extraction of all test saved at", model_params["save_feature_path"])
        print("The total time consuming:", time.time() - start_time)
if __name__ == '__main__':
    import argparse
    # 指定当前要读取的训练产出目录（其中包含 model_params.json / model.torch / norm.pkl）
    parser = argparse.ArgumentParser(description='Feature Extraction')
    parser.add_argument('--current_model_path', type=str,
                        default=r'C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\2025-11-12-01-54-20_fold0\model')
    args = parser.parse_args()
    # 单卡推理
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    Extraction(args).main()