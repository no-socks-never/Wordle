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

    seq_len_list = [10, 14]          # 你可以调整
    hidden_list = [64, 128]
    lr_list = [1e-3, 5e-4]
    feature_mode_list = ["A", "B", "C"]

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
    BiLSTM + Attention：
    默认只跑 feature_mode="C"，你可以按需扩展 feature_mode_list。
    """
    subdir = "BiLSTM"
    result_subdir = os.path.join(RESULT_DIR, subdir)
    os.makedirs(result_subdir, exist_ok=True)

    seq_len_list = [14]
    hidden_list = [64, 128]
    lr_list = [1e-3, 5e-4]
    feature_mode_list = ["A","B","C"]  # 如需也做 A/B，可以改成 ["A","B","C"]

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


def sweep_transformer():
    """
    Transformer 时间序列模型：
    默认只跑 feature_mode="C"。
    """
    subdir = "Transformer"
    result_subdir = os.path.join(RESULT_DIR, subdir)
    os.makedirs(result_subdir, exist_ok=True)

    seq_len_list = [14]
    lr_list = [1e-3, 5e-4]
    feature_mode_list = ["A","B","C"]

    records = []

    for seq_len, lr, feature_mode in product(
        seq_len_list, lr_list, feature_mode_list
    ):
        print("\n================ Transformer CONFIG ================")
        print(f"seq_len={seq_len}, lr={lr}, feature_mode={feature_mode}")

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
        )

        model_name = f"Transformer_seq{seq_len}_lr{lr}_feat{feature_mode}"
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
