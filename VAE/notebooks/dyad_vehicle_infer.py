# # import os
# # import argparse
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
# #
# # def score_vehicle(df, score_mode='mean'):
# #     scores = []
# #     for car, group in df.groupby("car"):
# #         group_sorted = group.sort_values(by="rec_error", ascending=False)
# #         top_k = group_sorted.head(10)["rec_error"].values  # 取前10段
# #         if score_mode == 'max':
# #             score = np.max(top_k)
# #         else:
# #             score = np.mean(top_k)
# #         scores.append((car, score))
# #     return pd.DataFrame(scores, columns=["car", "score"])
# #
# # def evaluate(vehicle_scores, threshold, vehicle_gt_path=None):
# #     vehicle_scores["predict"] = (vehicle_scores["score"] > threshold).astype(int)
# #
# #     if vehicle_gt_path and os.path.exists(vehicle_gt_path):
# #         gt_df = pd.read_csv(vehicle_gt_path)
# #         merged = pd.merge(gt_df, vehicle_scores, on="car", how="inner")
# #         y_true = merged["label"]
# #         y_pred = merged["predict"]
# #         y_score = merged["score"]
# #         metrics = {
# #             "AUC": round(roc_auc_score(y_true, y_score), 4),
# #             "Precision": round(precision_score(y_true, y_pred), 4),
# #             "Recall": round(recall_score(y_true, y_pred), 4),
# #             "F1": round(f1_score(y_true, y_pred), 4),
# #             "Accuracy": round(accuracy_score(y_true, y_pred), 4),
# #         }
# #         return merged, metrics, y_true, y_score
# #     else:
# #         print("[INFO] No label_csv provided or file not found. Only vehicle prediction results saved.")
# #         return vehicle_scores, {}, None, None
# #
# # def plot_score_bar(results, threshold, save_path=None):
# #     plt.figure(figsize=(10, 5))
# #     sorted_results = results.sort_values(by="score", ascending=False)
# #     colors = ['red' if p == 1 else 'green' for p in sorted_results["predict"]]
# #     plt.bar(sorted_results["car"], sorted_results["score"], color=colors)
# #     plt.axhline(threshold, color='blue', linestyle='--', label=f"Threshold = {threshold}")
# #     plt.xticks(rotation=45, ha='right')
# #     plt.ylabel("Vehicle Score")
# #     plt.title("Vehicle Anomaly Scores")
# #     plt.legend()
# #     plt.tight_layout()
# #     if save_path:
# #         plt.savefig(save_path)
# #     plt.show()
# #
# # def plot_roc_curve(y_true, y_score, save_path=None):
# #     fpr, tpr, _ = roc_curve(y_true, y_score)
# #     auc = roc_auc_score(y_true, y_score)
# #     plt.figure(figsize=(6, 5))
# #     plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="darkorange")
# #     plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
# #     plt.xlabel("False Positive Rate")
# #     plt.ylabel("True Positive Rate")
# #     plt.title("ROC Curve")
# #     plt.legend()
# #     plt.tight_layout()
# #     if save_path:
# #         plt.savefig(save_path)
# #     plt.show()
# #
# # def main(args):
# #     if not os.path.exists(args.input_csv):
# #         raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
# #     df = pd.read_csv(args.input_csv)
# #     print(f"[INFO] Loaded segment scores: {df.shape[0]} rows")
# #
# #     print(f"[INFO] Using fixed Top 10 segments per vehicle, scoring mode = {args.score_mode}, threshold = {args.threshold}")
# #     score_df = score_vehicle(df, score_mode=args.score_mode)
# #     results, metrics, y_true, y_score = evaluate(score_df, threshold=args.threshold, vehicle_gt_path=args.label_csv)
# #
# #     os.makedirs(args.output_dir, exist_ok=True)
# #
# #     # 保存整车级预测
# #     results.to_csv(os.path.join(args.output_dir, "vehicle_predictions.csv"), index=False)
# #     print(f"[INFO] Saved vehicle-level predictions to: {os.path.join(args.output_dir, 'vehicle_predictions.csv')}")
# #
# #     # 保存片段级预测（附加所属车辆预测结果）
# #     vehicle_label_map = dict(zip(results["car"], results["predict"]))
# #     df["vehicle_predict"] = df["car"].map(vehicle_label_map)
# #     df.to_csv(os.path.join(args.output_dir, "segment_predictions.csv"), index=False)
# #     print(f"[INFO] Saved segment-level predictions to: {os.path.join(args.output_dir, 'segment_predictions.csv')}")
# #
# #     if metrics:
# #         print("Evaluation metrics:")
# #         for k, v in metrics.items():
# #             print(f"{k}: {v}")
# #
# #         plot_score_bar(results, threshold=args.threshold,
# #                        save_path=os.path.join(args.output_dir, "vehicle_scores.png"))
# #         if y_true is not None and y_score is not None:
# #             plot_roc_curve(y_true, y_score, save_path=os.path.join(args.output_dir, "roc_curve.png"))
# #
# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--input_csv", type=str,
# #         default=r"D:\1研究生阶段资料\10项目\第十九届挑战杯昌平区赛道\2025年度中国青年科技创新“揭榜挂帅”擂台赛BJ02赛题(1)\DyAD(2)\DyAD\dyad_vae_save\2025-08-04-14-51-58_fold0\result\test_segment_scores.csv",
# #         help="Path to test_segment_scores.csv")
# #
# #     parser.add_argument("--label_csv", type=str,
# #         default=r"D:\1研究生阶段资料\10项目\第十九届挑战杯昌平区赛道\2025年度中国青年科技创新“揭榜挂帅”擂台赛BJ02赛题(1)\DyAD(2)\data\label\car_labels.csv",
# #         help="Optional vehicle-level label file with car,true_label")
# #
# #     parser.add_argument("--output_dir", type=str,
# #         default=r"D:\1研究生阶段资料\10项目\第十九届挑战杯昌平区赛道\2025年度中国青年科技创新“揭榜挂帅”擂台赛BJ02赛题(1)\DyAD(2)\DyAD\dyad_vae_save\2025-08-04-14-51-58_fold0\result",
# #         help="Directory to save prediction results")
# #
# #     # 已移除 head_n 参数，因为不再基于比例
# #     parser.add_argument("--score_mode", type=str, choices=["mean", "max"], default="mean", help="Scoring method")
# #     parser.add_argument("--threshold", type=float, default=0.03, help="Threshold for predicting abnormal vehicle")
# #
# #     args = parser.parse_args()
# #     main(args)
#
#
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
# from tqdm import tqdm
#
# # ---------------------- 参数区域：直接修改这里即可 ----------------------
# input_csv = r"C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\2025-10-27-10-17-51_fold0\result\test_segment_scores.csv"  # 输入片段打分文件（test_segment_scores.csv）
# output_dir = r"C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\2025-10-27-10-17-51_fold0\result\infer2"  # 输出结果保存目录
# score_mode = "max"  # 可选："mean" 或 "max"
# threshold = -1  # -1 表示自动选择最优阈值（基于 F1-score）
#
#
# # ------------------------------------------------------------------------
#
# def score_vehicle(df, score_mode='mean'):
#     scores = []
#     for car, group in df.groupby("car"):
#         group_sorted = group.sort_values(by="rec_error", ascending=False)
#         top_k = group_sorted.head(10)["rec_error"].values
#         score = np.max(top_k) if score_mode == 'max' else np.mean(top_k)
#         scores.append((car, score))
#     return pd.DataFrame(scores, columns=["car", "score"])
#
# # def score_vehicle(df, score_mode='mean'):
# #     scores = []
# #     for car, group in df.groupby("car"):
# #         # 对每辆车的所有片段进行处理，而不是仅限于前10段
# #         group_sorted = group.sort_values(by="rec_error", ascending=False)  # 按rec_error降序排列
# #         all_scores = group_sorted["rec_error"].values  # 获取所有片段的rec_error值
# #         score = np.max(all_scores) if score_mode == 'max' else np.mean(all_scores)  # 计算最大值或平均值
# #         scores.append((car, score))
# #     return pd.DataFrame(scores, columns=["car", "score"])
#
#
# def get_vehicle_label_from_segment(df):
#     car_labels = df.groupby("car")["label"].max().reset_index()
#     car_labels.columns = ["car", "label"]
#     return car_labels
#
#
# # 替代旧的 find_best_percent：用 F1-score 最大化选阈值
# def find_best_threshold_by_f1(result_df):
#     best_f1 = 0
#     best_threshold = 0
#     sorted_scores = sorted(result_df["score"].unique())
#
#     for t in sorted_scores:
#         result_df["predict"] = (result_df["score"] > t).astype(int)
#         y_true = result_df["label"]
#         y_pred = result_df["predict"]
#         f1 = f1_score(y_true, y_pred, zero_division=0)
#         if f1 > best_f1:
#             best_f1 = f1
#             best_threshold = t
#
#     print(f"[INFO] Best F1 = {best_f1:.4f} at threshold = {best_threshold:.5f}")
#     return best_threshold
#
#
# def evaluate(vehicle_scores, threshold):
#     vehicle_scores["predict"] = (vehicle_scores["score"] > threshold).astype(int)
#     y_true = vehicle_scores["label"]
#     y_pred = vehicle_scores["predict"]
#     y_score = vehicle_scores["score"]
#     metrics = {
#         "AUC": round(roc_auc_score(y_true, y_score), 4),
#         "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
#         "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
#         "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
#         "Accuracy": round(accuracy_score(y_true, y_pred), 4),
#     }
#     return vehicle_scores, metrics, y_true, y_score
#
#
# def plot_score_bar(results, threshold, save_path=None):
#     plt.figure(figsize=(10, 5))
#     sorted_results = results.sort_values(by="score", ascending=False)
#     colors = ['red' if p == 1 else 'green' for p in sorted_results["predict"]]
#     plt.bar(sorted_results["car"], sorted_results["score"], color=colors)
#     plt.axhline(threshold, color='blue', linestyle='--', label=f"Threshold = {threshold:.4f}")
#     plt.xticks(rotation=45, ha='right')
#     plt.ylabel("Vehicle Score")
#     plt.title("Vehicle Anomaly Scores")
#     plt.legend()
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()
#
#
# def plot_roc_curve(y_true, y_score, save_path=None):
#     fpr, tpr, _ = roc_curve(y_true, y_score)
#     auc = roc_auc_score(y_true, y_score)
#     plt.figure(figsize=(6, 5))
#     plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="darkorange")
#     plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
#     plt.xlabel("False Positive Rate")
#     plt.ylabel("True Positive Rate")
#     plt.title("ROC Curve")
#     plt.legend()
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()
#
#
# # ---------------------- 主程序 ----------------------
# def main():
#     if not os.path.exists(input_csv):
#         raise FileNotFoundError(f"[ERROR] CSV not found: {input_csv}")
#
#     df = pd.read_csv(input_csv)
#     print(f"[INFO] Loaded {df.shape[0]} segment samples.")
#
#     score_df = score_vehicle(df, score_mode=score_mode)
#     label_df = get_vehicle_label_from_segment(df)
#     score_df = pd.merge(score_df, label_df, on="car", how="inner")
#
#     # 使用 F1-score 自动或手动设定阈值
#     final_threshold = find_best_threshold_by_f1(score_df) if threshold < 0 else threshold
#
#     results, metrics, y_true, y_score = evaluate(score_df, final_threshold)
#
#     os.makedirs(output_dir, exist_ok=True)
#     results.to_csv(os.path.join(output_dir, "vehicle_predictions.csv"), index=False)
#     print(f"[INFO] Saved vehicle-level predictions to: {os.path.join(output_dir, 'vehicle_predictions.csv')}")
#
#     vehicle_label_map = dict(zip(results["car"], results["predict"]))
#     df["vehicle_predict"] = df["car"].map(vehicle_label_map)
#     df.to_csv(os.path.join(output_dir, "segment_predictions.csv"), index=False)
#     print(f"[INFO] Saved segment-level predictions to: {os.path.join(output_dir, 'segment_predictions.csv')}")
#
#     print("\n📊 Evaluation Metrics:")
#     for k, v in metrics.items():
#         print(f"{k}: {v}")
#
#     plot_score_bar(results, threshold=final_threshold,
#                    save_path=os.path.join(output_dir, "vehicle_scores.png"))
#     plot_roc_curve(y_true, y_score,
#                    save_path=os.path.join(output_dir, "roc_curve.png"))
#
#
# if __name__ == "__main__":
#     main()


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, roc_curve
from tqdm import tqdm
# ---------------------- 参数区域：直接修改这里即可 ----------------------
input_parquet = r"C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\2025-11-12-01-54-20_fold0\result\test_segment_scores.parquet" # 输入片段打分文件（test_segment_scores.parquet）
output_dir = r"C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\2025-11-12-01-54-20_fold0\infer" # 输出结果保存目录
score_mode = "max" # 可选："mean" 或 "max"
threshold = -1 # -1 表示自动选择最优阈值（基于 F1-score）
# ------------------------------------------------------------------------
def score_vehicle(df, score_mode='mean'):
    scores = []
    for car, group in df.groupby("car"):
        group_sorted = group.sort_values(by="rec_error", ascending=False)
        top_k = group_sorted.head(10)["rec_error"].values
        score = np.max(top_k) if score_mode == 'max' else np.mean(top_k)
        scores.append((car, score))
    return pd.DataFrame(scores, columns=["car", "score"])
# def score_vehicle(df, score_mode='mean'):
# scores = []
# for car, group in df.groupby("car"):
# # 对每辆车的所有片段进行处理，而不是仅限于前10段
# group_sorted = group.sort_values(by="rec_error", ascending=False) # 按rec_error降序排列
# all_scores = group_sorted["rec_error"].values # 获取所有片段的rec_error值
# score = np.max(all_scores) if score_mode == 'max' else np.mean(all_scores) # 计算最大值或平均值
# scores.append((car, score))
# return pd.DataFrame(scores, columns=["car", "score"])
def get_vehicle_label_from_segment(df):
    car_labels = df.groupby("car")["label"].max().reset_index()
    car_labels.columns = ["car", "label"]
    return car_labels
# 替代旧的 find_best_percent：用 F1-score 最大化选阈值
def find_best_threshold_by_f1(result_df):
    best_f1 = 0
    best_threshold = 0
    sorted_scores = sorted(result_df["score"].unique())
    for t in sorted_scores:
        result_df["predict"] = (result_df["score"] > t).astype(int)
        y_true = result_df["label"]
        y_pred = result_df["predict"]
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
    print(f"[INFO] Best F1 = {best_f1:.4f} at threshold = {best_threshold:.5f}")
    return best_threshold
def evaluate(vehicle_scores, threshold):
    # 强制将 y_true 转换为整数类型，并检查是否有异常值
    y_true = vehicle_scores["label"].apply(pd.to_numeric, errors='coerce') # 转换为数字，无法转换的变为 NaN
    # 检查是否有 NaN，输出警告信息
    if y_true.isna().any():
        print("[WARNING] Found non-numeric values in 'label' column. These will be treated as NaN and replaced with 0.")
        print(f"[WARNING] Non-numeric values in 'label':\n{vehicle_scores[y_true.isna()]['label'].unique()}")
    # 填充 NaN 为 0，并转换为整数类型
    y_true = y_true.fillna(0).astype(int)
    # 确保 y_pred 也是整数类型
    y_pred = vehicle_scores["predict"].astype(int)
    # 检查 y_true 和 y_pred 的数据类型是否一致
    assert y_true.dtype == y_pred.dtype, f"Data type mismatch: y_true is {y_true.dtype}, y_pred is {y_pred.dtype}"
    y_score = vehicle_scores["score"]
    # 计算评估指标
    metrics = {
        "AUC": round(roc_auc_score(y_true, y_score), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
    }
    return vehicle_scores, metrics, y_true, y_score
def plot_score_bar(results, threshold, save_path=None):
    plt.figure(figsize=(10, 5))
    sorted_results = results.sort_values(by="score", ascending=False)
    colors = ['red' if p == 1 else 'green' for p in sorted_results["predict"]]
    plt.bar(sorted_results["car"], sorted_results["score"], color=colors)
    plt.axhline(threshold, color='blue', linestyle='--', label=f"Threshold = {threshold:.4f}")
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Vehicle Score")
    plt.title("Vehicle Anomaly Scores")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
def plot_roc_curve(y_true, y_score, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()
# ---------------------- 主程序 ----------------------
def main():
    if not os.path.exists(input_parquet):
        raise FileNotFoundError(f"[ERROR] Parquet not found: {input_parquet}")
    # 读取 Parquet 文件
    df = pd.read_parquet(input_parquet)
    print(f"[INFO] Loaded {df.shape[0]} segment samples.")
    score_df = score_vehicle(df, score_mode=score_mode)
    label_df = get_vehicle_label_from_segment(df)
    score_df = pd.merge(score_df, label_df, on="car", how="inner")
    # 使用 F1-score 自动或手动设定阈值
    final_threshold = find_best_threshold_by_f1(score_df) if threshold < 0 else threshold
    results, metrics, y_true, y_score = evaluate(score_df, final_threshold)
    os.makedirs(output_dir, exist_ok=True)
    results.to_parquet(os.path.join(output_dir, "vehicle_predictions.parquet"), index=False)
    print(f"[INFO] Saved vehicle-level predictions to: {os.path.join(output_dir, 'vehicle_predictions.parquet')}")
    vehicle_label_map = dict(zip(results["car"], results["predict"]))
    df["vehicle_predict"] = df["car"].map(vehicle_label_map)
    df.to_parquet(os.path.join(output_dir, "segment_predictions.parquet"), index=False)
    print(f"[INFO] Saved segment-level predictions to: {os.path.join(output_dir, 'segment_predictions.parquet')}")
    print("\n📊 Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    plot_score_bar(results, threshold=final_threshold,
                   save_path=os.path.join(output_dir, "vehicle_scores.png"))
    plot_roc_curve(y_true, y_score,
                   save_path=os.path.join(output_dir, "roc_curve.png"))
if __name__ == "__main__":
    main()