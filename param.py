# param.py
"""
超参数搜索脚本：
- LSTM / BiLSTM / Transformer 三个模型
- 每个模型对应 results/{LSTM,BiLSTM,Transformer} 子文件夹
- LSTM：默认对 feature_mode ∈ {"A","B","C"} 做网格搜索
- BiLSTM / Transformer：默认只用 feature_mode="C"（你可以按需扩展）
"""

import os
from itertools import product

import pandas as pd
import torch

from config import (
    WORD_EMB_DIM,
    RESULT_DIR,
    NUM_EPOCHS,
)
# param.py 里 import 后面加一段统一搜索空间
GRID_SEQ_LEN = [10, 14]               # 统一：10 日 & 14 日历史窗口
GRID_HIDDEN = [64, 128]           # 统一：小/中/大 三种隐层维度
GRID_LR = [1e-3, 5e-4]          # 统一：含一个稍小学习率
GRID_FEATURE_MODE = ["A", "B", "C"]   # 统一：三种特征模式全部搜索


from data_prepare import prepare_datasets
from models import BaseLSTMModel, BiLSTMAttentionModel, TransformerTimeModel
from train_eval import train_one_model

os.makedirs(RESULT_DIR, exist_ok=True)


def sweep_lstm():
    """
    LSTM 模型的超参数搜索：
      - seq_len
      - hidden_dim
      - lr
      - feature_mode ("A","B","C")
    结果保存到 results/LSTM/hparam_results_LSTM.csv
    """
    subdir = "LSTM"
    result_subdir = os.path.join(RESULT_DIR, subdir)
    os.makedirs(result_subdir, exist_ok=True)

    # 统一的搜索空间
    seq_len_list = GRID_SEQ_LEN
    hidden_list = GRID_HIDDEN
    lr_list = GRID_LR
    feature_mode_list = GRID_FEATURE_MODE

    records = []

    for seq_len, hidden_dim, lr, feature_mode in product(
        seq_len_list, hidden_list, lr_list, feature_mode_list
    ):
        print("\n================ LSTM CONFIG ================")
        print(f"seq_len={seq_len}, hidden_dim={hidden_dim}, lr={lr}, feature_mode={feature_mode}")

        info, dataset, loaders = prepare_datasets(
            seq_len=seq_len,
            feature_mode=feature_mode,
            save_npz=False,
        )
        train_loader, val_loader, test_loader = loaders

        vocab_size = info["vocab_size"]
        num_numeric_features = info["num_numeric_features"]
        thresholds = info["difficulty_thresholds"]

        model = BaseLSTMModel(
            vocab_size=vocab_size,
            num_numeric_features=num_numeric_features,
            word_emb_dim=WORD_EMB_DIM,
            hidden_dim=hidden_dim,
        )

        model_name = f"LSTM_seq{seq_len}_h{hidden_dim}_lr{lr}_feat{feature_mode}"
        best_val_loss, test_metrics = train_one_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            model_name=model_name,
            lr=lr,
            thresholds=thresholds,
            result_subdir=subdir,
            return_val=True,
        )

        record = {
            "model": "LSTM",
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "feature_mode": feature_mode,
            "best_val_loss": best_val_loss,
        }
        record.update({f"test_{k}": v for k, v in test_metrics.items()})
        records.append(record)

    df = pd.DataFrame(records)
    out_path = os.path.join(result_subdir, "hparam_results_LSTM.csv")
    df.to_csv(out_path, index=False)
    print(f"[LSTM] 全部结果已保存到 {out_path}")
    return df


def sweep_bilstm():
    """
    BiLSTM + Attention 的超参数搜索：
      - seq_len
      - hidden_dim
      - lr
      - feature_mode ("A","B","C")
    结果保存到 results/BiLSTM/hparam_results_BiLSTM.csv
    """
    subdir = "BiLSTM"
    result_subdir = os.path.join(RESULT_DIR, subdir)
    os.makedirs(result_subdir, exist_ok=True)

    # 统一的搜索空间
    seq_len_list = GRID_SEQ_LEN
    hidden_list = GRID_HIDDEN
    lr_list = GRID_LR
    feature_mode_list = GRID_FEATURE_MODE

    records = []

    for seq_len, hidden_dim, lr, feature_mode in product(
        seq_len_list, hidden_list, lr_list, feature_mode_list
    ):
        print("\n================ BiLSTM CONFIG ================")
        print(f"seq_len={seq_len}, hidden_dim={hidden_dim}, lr={lr}, feature_mode={feature_mode}")

        info, dataset, loaders = prepare_datasets(
            seq_len=seq_len,
            feature_mode=feature_mode,
            save_npz=False,
        )
        train_loader, val_loader, test_loader = loaders

        vocab_size = info["vocab_size"]
        num_numeric_features = info["num_numeric_features"]
        thresholds = info["difficulty_thresholds"]

        model = BiLSTMAttentionModel(
            vocab_size=vocab_size,
            num_numeric_features=num_numeric_features,
            word_emb_dim=WORD_EMB_DIM,
            hidden_dim=hidden_dim,
        )

        model_name = f"BiLSTM_seq{seq_len}_h{hidden_dim}_lr{lr}_feat{feature_mode}"
        best_val_loss, test_metrics = train_one_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            model_name=model_name,
            lr=lr,
            thresholds=thresholds,
            result_subdir=subdir,
            return_val=True,
        )

        record = {
            "model": "BiLSTM",
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "feature_mode": feature_mode,
            "best_val_loss": best_val_loss,
        }
        record.update({f"test_{k}": v for k, v in test_metrics.items()})
        records.append(record)

    df = pd.DataFrame(records)
    out_path = os.path.join(result_subdir, "hparam_results_BiLSTM.csv")
    df.to_csv(out_path, index=False)
    print(f"[BiLSTM] 全部结果已保存到 {out_path}")
    return df

def sweep_bilstm_ablation():
    """
    BiLSTM 消融实验：
      - 围绕当前最优配置 seq_len=14, feature_mode="B"
      - 只调整 hidden_dim 和 lr 做小范围精修
      - 结果保存到 results/BiLSTM_ablation/hparam_results_BiLSTM_ablation.csv
    """
    subdir = "BiLSTM_ablation"
    result_subdir = os.path.join(RESULT_DIR, subdir)
    os.makedirs(result_subdir, exist_ok=True)

    # 固定在当前最优附近：seq_len=14, feature_mode="B"
    seq_len_list = [14]

    # 更小的隐藏维度；64 作为对照
    hidden_list = [16, 32, 48, 64]

    # 比原来更小的学习率；5e-4 作为对照
    lr_list = [5e-4, 2e-4, 1e-4]

    # 只在 B 模式做消融（文本 + 静态特征）
    feature_mode_list = ["B"]

    records = []

    for seq_len, hidden_dim, lr, feature_mode in product(
        seq_len_list, hidden_list, lr_list, feature_mode_list
    ):
        print("\n================ BiLSTM ABLATION CONFIG ================")
        print(f"seq_len={seq_len}, hidden_dim={hidden_dim}, lr={lr}, feature_mode={feature_mode}")

        # 和之前完全一样的准备流程
        info, dataset, loaders = prepare_datasets(
            seq_len=seq_len,
            feature_mode=feature_mode,
            save_npz=False,
        )
        train_loader, val_loader, test_loader = loaders

        vocab_size = info["vocab_size"]
        num_numeric_features = info["num_numeric_features"]
        thresholds = info["difficulty_thresholds"]

        model = BiLSTMAttentionModel(
            vocab_size=vocab_size,
            num_numeric_features=num_numeric_features,
            word_emb_dim=WORD_EMB_DIM,
            hidden_dim=hidden_dim,
        )

        # 单独起一个名字，避免和原 sweep 混淆
        model_name = f"BiLSTM_ablation_seq{seq_len}_h{hidden_dim}_lr{lr}_feat{feature_mode}"

        best_val_loss, test_metrics = train_one_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            model_name=model_name,
            lr=lr,
            thresholds=thresholds,
            result_subdir=subdir,   # 注意这里用的是子目录名
            return_val=True,
        )

        record = {
            "model": "BiLSTM",
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "feature_mode": feature_mode,
            "best_val_loss": best_val_loss,
        }
        record.update({f"test_{k}": v for k, v in test_metrics.items()})
        records.append(record)

    df = pd.DataFrame(records)
    out_path = os.path.join(result_subdir, "hparam_results_BiLSTM_ablation.csv")
    df.to_csv(out_path, index=False)
    print(f"[BiLSTM ablation] 全部结果已保存到 {out_path}")
    return df


def sweep_transformer():
    """
    Transformer 时间序列模型的超参数搜索：
      - seq_len
      - d_model（这里用 GRID_HIDDEN）
      - lr
      - feature_mode ("A","B","C")
    结果保存到 results/Transformer/hparam_results_Transformer.csv
    """
    subdir = "Transformer"
    result_subdir = os.path.join(RESULT_DIR, subdir)
    os.makedirs(result_subdir, exist_ok=True)

    # 统一的搜索空间
    seq_len_list = GRID_SEQ_LEN
    hidden_list = GRID_HIDDEN        # 这里对应 Transformer 的 d_model
    lr_list = GRID_LR
    feature_mode_list = GRID_FEATURE_MODE

    records = []

    for seq_len, hidden_dim, lr, feature_mode in product(
        seq_len_list, hidden_list, lr_list, feature_mode_list
    ):
        print("\n================ Transformer CONFIG ================")
        print(f"seq_len={seq_len}, d_model={hidden_dim}, lr={lr}, feature_mode={feature_mode}")

        info, dataset, loaders = prepare_datasets(
            seq_len=seq_len,
            feature_mode=feature_mode,
            save_npz=False,
        )
        train_loader, val_loader, test_loader = loaders

        vocab_size = info["vocab_size"]
        num_numeric_features = info["num_numeric_features"]
        thresholds = info["difficulty_thresholds"]

        model = TransformerTimeModel(
            vocab_size=vocab_size,
            num_numeric_features=num_numeric_features,
            d_model=hidden_dim,  # 关键：把统一的 hidden_dim 当做 d_model
        )

        model_name = f"Trans_seq{seq_len}_d{hidden_dim}_lr{lr}_feat{feature_mode}"
        best_val_loss, test_metrics = train_one_model(
            model,
            train_loader,
            val_loader,
            test_loader,
            model_name=model_name,
            lr=lr,
            thresholds=thresholds,
            result_subdir=subdir,
            return_val=True,
        )

        record = {
            "model": "Transformer",
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,  # 这里其实就是 d_model
            "lr": lr,
            "feature_mode": feature_mode,
            "best_val_loss": best_val_loss,
        }
        record.update({f"test_{k}": v for k, v in test_metrics.items()})
        records.append(record)

    df = pd.DataFrame(records)
    out_path = os.path.join(result_subdir, "hparam_results_Transformer.csv")
    df.to_csv(out_path, index=False)
    print(f"[Transformer] 全部结果已保存到 {out_path}")
    return df


if __name__ == "__main__":
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("==== Sweep LSTM ====")
    sweep_lstm()

    print("\n==== Sweep BiLSTM ====")
    sweep_bilstm()

    print("\n==== Sweep Transformer ====")
    sweep_transformer()
    print("\n==== Sweep BiLSTM Ablation ====")
    sweep_bilstm_ablation()




    

