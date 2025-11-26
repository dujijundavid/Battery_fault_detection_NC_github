import numpy as np
import torch

from utils import to_var

import torch
import torch.nn.functional as F

class Label:
    def __init__(
        self,
        column_name,
        training_set=None,
        *,
        min_mileage=5,
        max_mileage=280000,
        device="cuda",
        clip_to_01=True
    ):
        """
        column_name: 目标列名（如 'mileage'）
        training_set: 若不手动传 min/max，用它来自动求 min/max
        min_mileage / max_mileage: 手动指定归一化范围（推荐）
        device: 'cuda' 或 'cpu'
        clip_to_01: 是否把归一化结果裁剪到 [0, 1]
        """
        self.label = column_name
        self.device = device
        self.clip_to_01 = clip_to_01

        if min_mileage is not None and max_mileage is not None:
            self.min_mileage = float(min_mileage)
            self.max_mileage = float(max_mileage)
        else:
            if training_set is None:
                raise ValueError("未提供 min/max 时必须给 training_set 用于自动计算。")
            sample_mileage = [training_set[i][1][self.label] for i in range(len(training_set))]
            self.min_mileage = float(min(sample_mileage))
            self.max_mileage = float(max(sample_mileage))

        # 处理异常区间
        self._range = float(self.max_mileage - self.min_mileage)
        if self._range < 0:
            raise ValueError(f"max_mileage({self.max_mileage}) 应 ≥ min_mileage({self.min_mileage})")

    def _normalize(self, v):
        if self._range == 0:
            norm = 0.0
        else:
            norm = (float(v) - self.min_mileage) / self._range
        if self.clip_to_01:
            # 裁剪到 [0, 1]，避免越界
            norm = 0.0 if norm < 0.0 else (1.0 if norm > 1.0 else norm)
        return norm

    def loss(self, batch, mean_pred, is_mse=True):
        # batch[1][self.label] 是原始里程序列
        label_data = [self._normalize(i) for i in batch[1][self.label]]
        y = torch.tensor(label_data, dtype=torch.float32, device=self.device)
        x = mean_pred.squeeze().to(self.device)
        return F.mse_loss(x, y, reduction='mean') if is_mse else torch.tensor(0.0, device=self.device)

# class Label:
#     def __init__(self, column_name, training_set, sample_length=50):
#         self.label = column_name
#         self.sample_length = sample_length
#         self.sample_mileage = [training_set[i][1][self.label] for i in range(self.sample_length)]
#         self.max_mileage = max(self.sample_mileage)
#         self.min_mileage = min(self.sample_mileage)
#
#     def loss(self, batch, mean_pred, is_mse=True):
#         label_data = []
#         for i in batch[1][self.label]:
#             norm_label = (i - self.min_mileage) / (self.max_mileage - self.min_mileage)
#             label_data.append(norm_label)
#         label = torch.tensor(label_data)
#         x = mean_pred.squeeze().to("cuda")
#         y = label.float().to("cuda")
#         mse = torch.nn.MSELoss(reduction='mean')
#         loss = 0
#         if is_mse:
#             loss = mse(x, y)
#         return loss


class Task:
    """
    Task
    """

    def __init__(self, columns, encoder_dimension=122, decoder_dimension=122,
                 output_dimension=122, task_name='ev'):
        """
        :param columns: columns
        :param encoder_dimension: encoder dimension int
        :param decoder_dimension: decoder dimension int
        :param output_dimension: output dimension int
        :param task_name: task name, e.g. ev
        """
        self.encoder_dimension = encoder_dimension
        self.decoder_dimension = decoder_dimension
        self.output_dimension = output_dimension
        self.task_name = task_name
        self.columns = columns
        self.encoder = []
        self.decoder = []
        self.target = []
        eval(self.task_name.capitalize() + 'Task.set_params')(self)
        eval(self.task_name.capitalize() + 'Task.get_task_idx')(self, columns)

    def encoder_filter(self, input_embedding):
        return eval(self.task_name.capitalize() + 'Task.task_encoder')(self, input_embedding, self.columns)

    def decoder_filter(self, input_embedding):
        self.decoder = self.encoder[:self.decoder_dimension]
        return to_tensor(to_array(input_embedding)[:, :, self.decoder])

    def target_filter(self, input_embedding):
        self.target = self.encoder[self.decoder_dimension:]
        return to_tensor(to_array(input_embedding)[:, :, self.target])

    def task_encoder(self, input_embedding, columns):
        return to_tensor(to_array(input_embedding)[:, :, self.encoder])


class EvTask(Task):
    def set_params(self):
        """
        initialize
        """
        self.encoder_dimension = 6
        self.decoder_dimension = 2
        self.output_dimension = 4

    def get_task_idx(self, columns):
        """
        filter specified columns
        :param columns: column names
        :Return corresponding indexes
        """
        self.encoder = np.array(
            [columns.index("soc"), columns.index("current"),
             columns.index("max_temp"), columns.index("max_single_volt"),
             columns.index("min_single_volt"), columns.index("volt")]).astype(int)
        return self.encoder


class BatterybrandbTask(Task):
    def set_params(self):
        self.encoder_dimension = 7
        self.decoder_dimension = 4
        self.output_dimension = 3

    def get_task_idx(self, columns):
        self.encoder = np.array(
            [columns.index("soc"),
             columns.index("current"),
             columns.index("min_temp"),
             columns.index("max_single_volt"),
             columns.index("min_single_volt"),
             columns.index('volt'),
             columns.index("max_temp")]).astype(int)
        return self.encoder

class BatterybrandaTask(Task):
    def set_params(self):
        self.encoder_dimension = 7
        self.decoder_dimension = 2
        self.output_dimension = 5

    def get_task_idx(self, columns):
        self.encoder = np.array(
            [columns.index("soc"),
             columns.index("current"),
             columns.index("min_temp"),
             columns.index("max_single_volt"),
             columns.index("max_temp"),
             columns.index("min_single_volt"),
             columns.index('volt')]).astype(int)
        return self.encoder

def to_tensor(input_embedding):
    return to_var(torch.from_numpy(np.array(input_embedding)))


def to_array(input_embedding):
    return input_embedding.cpu().numpy()
