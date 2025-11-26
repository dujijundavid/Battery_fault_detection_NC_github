import pandas as pd


def parquet_to_csv_first_100(parquet_file_path, csv_file_path):
    """
    将Parquet文件的前100行转换为CSV文件（兼容旧版本pyarrow）
    """
    try:
        # 读取整个Parquet文件
        df = pd.read_parquet(parquet_file_path, engine='pyarrow')

        # 提取前100行（如果文件不足100行则取全部）
        df_first_100 = df.head(100)

        # 写入CSV
        df_first_100.to_csv(csv_file_path, index=False)

        print(f"转换成功！已提取前100行并保存至: {csv_file_path}")
        print(f"实际提取行数: {len(df_first_100)}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {parquet_file_path}")
    except Exception as e:
        print(f"转换过程中发生错误：{str(e)}")


if __name__ == "__main__":
    parquet_path = r"C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\2025-11-12-01-54-20_fold0\result\test_segment_scores.parquet"  # 替换为你的Parquet文件路径
    csv_path = r"C:\Users\YIFSHEN\Documents\VAE\DyAD\dyad_vae_save\2025-11-12-01-54-20_fold0\result\output_first_100.csv"  # 替换为输出CSV路径
    parquet_to_csv_first_100(parquet_path, csv_path)