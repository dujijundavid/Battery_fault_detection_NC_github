import argparse
import json
import os
import train
import extract
import evaluate
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch battery Example')
    parser.add_argument('--config_path', type=str,
                        default=r'C:\Users\YIFSHEN\Documents\VAE\DyAD\model_params_battery_brandall.json')
    parser.add_argument('--fold_num', type=int, default=0)

    args = parser.parse_args()

    # 1. 读取 config_path （一定存在）
    with open(args.config_path, 'r', encoding='utf-8') as f:
        model_params = json.load(f)  # 假设是平的dict
    # 2. 将 model_params 合并进 args
    p_args = argparse.Namespace()
    p_args.__dict__.update(model_params)
    args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.config_path)
    print("args", args)

    # 3. 训练
    tr = train.Train_fivefold(args, fold_num=args.fold_num)
    print('train start............................')
    tr.main()
    print('train end............................')

    # 获取模型参数保存路径并保存为 modelparams_path
    modelparams_path = tr.getmodelparams()  # 这里应该返回一个文件路径
    print(f'Model params saved to {modelparams_path}')
    del tr

    # 4. 新增 modelparams_path 参数
    parser.add_argument('--modelparams_path', type=str, default=modelparams_path)
    args = parser.parse_args()

    # 确保路径存在并读取 model_params
    if os.path.exists(args.modelparams_path):
        with open(args.modelparams_path, 'r', encoding='utf-8') as f:
            model_params = json.load(f)
        p_args = argparse.Namespace()
        p_args.__dict__.update(model_params)
        args = parser.parse_args(namespace=p_args)
        print(f'Loaded model params from {args.modelparams_path}')
    else:
        print(f"Error: {args.modelparams_path} does not exist")
        exit(1)

    # 5. 特征提取
    ext = extract.Extraction(args, fold_num=args.fold_num)
    print('feature extraction start............................')
    ext.main()
    print('feature extraction end............................')
    del ext

    # 6. 评估（Anomaly Detection）
    ev = evaluate.Evaluate(args)
    print('anomaly detection start............................')
    ev.main()
    print('anomaly detection end............................')
    del ev
