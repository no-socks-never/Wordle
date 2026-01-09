import os
import pandas as pd
import torch
import numpy as np

from config import RESULT_DIR, DATA_CSV_PATH, WORD_EMB_DIM
from data_prepare import prepare_datasets, load_raw_data, compute_distribution_and_targets
from models import BaseLSTMModel, BiLSTMAttentionModel, TransformerTimeModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================
# 1. 读取三个模型的最佳配置（基于 val_loss）
# ======================================================
def load_best_row(csv_path):
    df = pd.read_csv(csv_path)
    best = df.loc[df["best_val_loss"].idxmin()]
    return best


def build_model(model_name, best_row, vocab_size, num_numeric):
    """根据 best_row 构造对应模型"""
    seq_len = int(best_row["seq_len"])
    hidden_dim = int(best_row["hidden_dim"])
    lr = float(best_row["lr"])
    feature_mode = best_row["feature_mode"]

    if model_name == "LSTM":
        return BaseLSTMModel(vocab_size, num_numeric,
                             word_emb_dim=WORD_EMB_DIM,
                             hidden_dim=hidden_dim), seq_len, feature_mode

    if model_name == "BiLSTM":
        return BiLSTMAttentionModel(vocab_size, num_numeric,
                                    word_emb_dim=WORD_EMB_DIM,
                                    hidden_dim=hidden_dim), seq_len, feature_mode

    if model_name == "Transformer":
        return TransformerTimeModel(vocab_size, num_numeric,
                                    d_model=hidden_dim), seq_len, feature_mode

    raise ValueError("Unknown model name:", model_name)


# ======================================================
# 2. 用最优模式重新 prepare 数据集（必须用其最佳 seq_len & 特征模式）
# ======================================================
def prepare_for_best(seq_len, feature_mode):
    info, dataset, loaders = prepare_datasets(
        seq_len=seq_len,
        feature_mode=feature_mode,
        save_npz=False
    )
    return info, dataset


# ======================================================
# 3. 从训练保存的文件中加载最佳权重
# ======================================================
def find_best_weight_file(model_subdir, model_name_prefix):
    subdir = f"results/{model_subdir}"
    files = os.listdir(subdir)
    files = [f for f in files if f.endswith(".pt") and model_name_prefix in f]
    if len(files) == 0:
        raise FileNotFoundError(f"No weight found for {model_name_prefix} in {subdir}")
    return os.path.join(subdir, sorted(files)[-1])  # 取最新


# ======================================================
# 4. 对指定单词推理：得到 7 维分布
# ======================================================
def predict_for_word(model, dataset, word, info):
    w2id = info["word2id"]
    if word not in w2id:
        raise ValueError(f"Word {word} not found in vocab!")

    # 找到这个词在 dataset 中的 index
    df = load_raw_data(DATA_CSV_PATH)
    df = compute_distribution_and_targets(df)
    idx = df.index[df["word"] == word].tolist()
    if len(idx) == 0:
        raise ValueError("Word not found in dataset.")
    idx = idx[0]

    # 找到它在序列中的输入位置（target index = start + seq_len）
    seq_len = info["seq_len"]
    target_pos = idx
    start = target_pos - seq_len
    if start < 0:
        raise ValueError(f"Word {word} is too early, no full history window.")

    word_seq = dataset.word_ids[start:target_pos]
    num_seq = dataset.numeric_feats[start:target_pos]

    word_seq = torch.tensor(word_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
    num_seq = torch.tensor(num_seq, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(word_seq, num_seq)
        if isinstance(out, tuple):
            dist = out[0]
        else:
            dist = out
        return dist.squeeze(0).cpu().numpy(), df.loc[idx, ["p1","p2","p3","p4","p5","p6","pX"]].values


# ======================================================
# 5. 主函数：输出结果 CSV
# ======================================================
def main(words_to_check):

    # 读取三个模型最优配置
    # 自动选择LSTM最优模型
    import pandas as pd
    lstm_df = pd.read_csv("results/LSTM/hparam_results_LSTM.csv")
    # 按优先级排序
    lstm_df = lstm_df.sort_values(["test_mae_avg", "test_mae_succ", "test_dist_l1"], ascending=[True, True, True])
    best_lstm = lstm_df.iloc[0]
    # BiLSTM和Transformer仍用原逻辑
    best_bilstm = load_best_row("results/BiLSTM/hparam_results_BiLSTM.csv")
    best_trans = load_best_row("results/Transformer/hparam_results_Transformer.csv")

    # 逐模型构造: 先准备其对应参数的数据集
    models_best = {
        "LSTM": best_lstm,
        "BiLSTM": best_bilstm,
        "Transformer": best_trans
    }

    # 收集结果
    rows = []

    for model_name, best_row in models_best.items():
        # 兼容pandas.Series和原DataFrame行
        if hasattr(best_row, 'to_dict'):
            best_row = best_row.to_dict()
        seq_len = int(best_row["seq_len"])
        feat_mode = best_row["feature_mode"]
        dummy_model, _, _ = build_model(model_name, best_row, 10, 0)
        info, dataset = prepare_for_best(seq_len, feat_mode)
        vocab_size = info["vocab_size"]
        num_numeric = info["num_numeric_features"]
        model, _, _ = build_model(model_name, best_row, vocab_size, num_numeric)
        model = model.to(DEVICE)
        prefix = f"{model_name}_seq{seq_len}_"
        weight_file = find_best_weight_file(model_name, prefix)
        model.load_state_dict(torch.load(weight_file, map_location=DEVICE))
        models_best[model_name] = (model, info, dataset)

    # now iterate words
    result_list = []
    for w in words_to_check:
        row = {"word": w}
        for model_name, content in models_best.items():
            model, info, dataset = content
            pred, gt = predict_for_word(model, dataset, w, info)
            row[f"{model_name}_pred"] = pred.tolist()
            row["ground_truth"] = gt.tolist()
        result_list.append(row)

    out_df = pd.DataFrame(result_list)
    out_path = "results/best_model_distributions.csv"
    out_df.to_csv(out_path, index=False)
    print(f"\n结果已保存到 {out_path}\n")
    print(out_df)


if __name__ == "__main__":
    # 你想要检查的单词列表（可以自行修改）
    words_to_check = ["manly", "molar", "agape", "cacao","chest","homer","angry","label"]
    main(words_to_check)
