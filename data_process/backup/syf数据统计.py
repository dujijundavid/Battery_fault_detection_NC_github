import pandas as pd
import os


def process_car_data(input_path, output_path):
    """
    处理车辆数据，按car分组统计所需信息并输出

    参数:
        input_path: 输入CSV文件路径
        output_path: 输出CSV文件路径
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(input_path)

        # 检查必要的列是否存在
        required_columns = ['car', 'mileage', 'label', 'rec_error']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"输入文件缺少必要的列: {', '.join(missing_columns)}")

        # 按car分组处理
        result_list = []

        for car, group in df.groupby('car'):
            # 找到最大rec_error对应的行
            max_error_row = group.loc[group['rec_error'].idxmax()]

            # 该车辆的最大里程
            max_mileage = group['mileage'].max()

            # 收集结果
            result_list.append({
                'car': car,
                'max_rec_error': max_error_row['rec_error'],
                'mileage_at_max_error': max_error_row['mileage'],
                'label_at_max_error': max_error_row['label'],
                'max_mileage': max_mileage
            })

        # 转换为DataFrame并保存
        result_df = pd.DataFrame(result_list)
        result_df.to_csv(output_path, index=False)
        print(f"处理完成，结果已保存至: {output_path}")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


# 使用示例
if __name__ == "__main__":
    # 输入文件路径（请替换为实际路径）
    input_csv_path = "C:/Users/YIFSHEN/Documents/VAE/DyAD/dyad_vae_save/2025-10-27-10-17-51_fold0/result/infer2/segment_predictions.csv"
    # 输出文件路径（请替换为实际路径）
    output_csv_path = "C:/Users/YIFSHEN/Desktop/processed_car_data.csv"


    # 确保输出目录存在
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理数据
    process_car_data(input_csv_path, output_csv_path)