# data_prepare.py
"""
数据预处理模块：
- 从 CSV 读取每日 Wordle 统计数据
- 计算 7 维分布、avg_tries、succ_rate
- 基于 avg_tries 做数据驱动的三档难度分桶（约等分）
- 构造三类特征：
    * 文本：答案词序列（后续做 embedding）
    * 静态：distinct_letters / Lg10CD / 首字母是否高频
    * 动态：玩法 & 时间相关特征（num_reported 等 + weekday one-hot）
- 构造序列 Dataset: 使用过去 seq_len 天预测第 t 天
"""

import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_CSV_PATH,
    PROCESSED_DATA_PATH,
    RESULT_DIR,
    SEQ_LEN,
    VAL_RATIO,
    TEST_RATIO,
    BATCH_SIZE,
    RANDOM_SEED,
)

# ==== 静态词频字典（来自你之前整理好的 wordle_word_freq_from_subtlex.csv） ====
WORD_FREQ_CSV = os.path.join(os.path.dirname(__file__), "data/wordle_word_freq_from_subtlex.csv")

DEFAULT_LG10CD = 0.301  # 你当前 follow work 设置的缺省 Lg10CD
HIGH_FREQ_INITIALS = set("SCPTAMBRLD")  # 首字母高频集合，可以之后再微调

if os.path.exists(WORD_FREQ_CSV):
    _freq_df = pd.read_csv(WORD_FREQ_CSV)
    _freq_df["Word"] = _freq_df["Word"].astype(str).str.strip().str.upper()
    FREQ_LEXICON = dict(zip(_freq_df["Word"], _freq_df["Lg10CD_used"]))
else:
    print(f"[data_prepare] Warning: {WORD_FREQ_CSV} not found, 所有单词 Lg10CD 将使用默认值 {DEFAULT_LG10CD}")
    FREQ_LEXICON = {}

np.random.seed(RANDOM_SEED)


# ==================== 1. 读取 & 基础清洗 ====================

def load_raw_data(csv_path: str) -> pd.DataFrame:
    """
    从官方统计 CSV 读取每日 Wordle 数据。
    预期列名（和你给的示例一致）：
        Date, Contest number, Word, Number of reported results, Number in hard mode,
        1 try, 2 tries, 3 tries, 4 tries, 5 tries, 6 tries, 7 or more tries (X)
    """
    df = pd.read_csv(csv_path)

    # 标准化列名
    rename_map = {
        "Date": "date",
        "Contest number": "contest",
        "Word": "word",
        "Number of reported results": "num_reported",
        "Number in hard mode": "num_hard",
        "1 try": "c1",
        "2 tries": "c2",
        "3 tries": "c3",
        "4 tries": "c4",
        "5 tries": "c5",
        "6 tries": "c6",
        "7 or more tries (X)": "cX",
    }
    df = df.rename(columns=rename_map)

    # 日期格式 + 排序
    df["date"] = pd.to_datetime(df["date"])
    df["contest"] = df["contest"].astype(int)
    df["word"] = df["word"].astype(str).str.strip().str.upper()

    df = df.sort_values("contest").reset_index(drop=True)
    return df


# ==================== 2. 分布、avg_tries、succ_rate ====================

def compute_distribution_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """
    根据每天 1~6 次以及 X（失败）统计，构造：
    - p1..p6,pX：7 维概率分布
    - avg_tries：访问 1~7 次的期望尝试次数（X 当作 7）
    - succ_rate：成功率 = 1 - pX
    """
    count_cols = ["c1", "c2", "c3", "c4", "c5", "c6", "cX"]
    counts = df[count_cols].values.astype(float)
    total = counts.sum(axis=1, keepdims=True)
    total[total == 0] = 1.0  # 避免除零

    probs = counts / total  # [N,7]
    for i, name in enumerate(["p1", "p2", "p3", "p4", "p5", "p6", "pX"]):
        df[name] = probs[:, i]

    tries_index = np.arange(1, 8, dtype=float)  # 1..7, X 当作 7
    avg_tries = (probs * tries_index).sum(axis=1)
    succ_rate = 1.0 - probs[:, -1]

    df["avg_tries"] = avg_tries
    df["succ_rate"] = succ_rate
    return df


# ==================== 3. 数据驱动难度分桶（量化分位数） ====================

def compute_difficulty_thresholds_from_df(
    df: pd.DataFrame,
    q_low: float = 1.0 / 3.0,
    q_high: float = 2.0 / 3.0,
) -> Tuple[float, float]:
    """
    使用 avg_tries 的分位数，把样本大致分成三等份：
    - avg < easy_th    -> easy
    - easy_th ~ med_th -> medium
    - >= med_th        -> hard
    """
    avg_vals = df["avg_tries"].values
    easy_th, med_th = np.quantile(avg_vals, [q_low, q_high])
    return float(easy_th), float(med_th)


def avg_to_label_array(avg_tries: np.ndarray,
                       easy_th: float,
                       med_th: float) -> np.ndarray:
    labels = np.zeros_like(avg_tries, dtype=int)
    labels[avg_tries >= easy_th] = 1
    labels[avg_tries >= med_th] = 2
    return labels


def compute_data_driven_thresholds_from_csv(
    csv_path: str = DATA_CSV_PATH,
    q_low: float = 1.0 / 3.0,
    q_high: float = 2.0 / 3.0,
):
    """
    独立小工具：直接从 CSV 计算分位数阈值，并打印各档样本数量。
    方便你在论文里说明“基于分位数近似等分三个难度水平”。
    """
    df = load_raw_data(csv_path)
    df = compute_distribution_and_targets(df)
    easy_th, med_th = compute_difficulty_thresholds_from_df(df, q_low, q_high)
    labels = avg_to_label_array(df["avg_tries"].values, easy_th, med_th)
    unique, counts = np.unique(labels, return_counts=True)
    print("=== Data-driven difficulty thresholds ===")
    print(f"easy_th  (q={q_low:.2f}): {easy_th:.4f}")
    print(f"med_th   (q={q_high:.2f}): {med_th:.4f}")
    print("class counts (0:easy,1:medium,2:hard):")
    for u, c in zip(unique, counts):
        print(f"  label {u}: {c}")
    return easy_th, med_th


# ==================== 4. 静态 & 动态特征 ====================

def compute_static_feats(words: pd.Series) -> np.ndarray:
    """
    三维静态特征：
    0: distinct_letters
    1: Lg10CD（来自 SUBTLEX，缺失用 DEFAULT_LG10CD）
    2: first_letter_high_freq（首字母是否在高频首字母集合中）
    """
    feats = []
    for w in words.astype(str):
        w_clean = w.strip().upper()
        letters = [ch for ch in w_clean if ch.isalpha()]
        distinct = len(set(letters))
        log_cd = FREQ_LEXICON.get(w_clean, DEFAULT_LG10CD)
        if letters:
            first_high = 1.0 if letters[0] in HIGH_FREQ_INITIALS else 0.0
        else:
            first_high = 0.0
        feats.append([distinct, log_cd, first_high])
    return np.asarray(feats, dtype=np.float32)


def compute_dynamic_feats(df: pd.DataFrame) -> np.ndarray:
    """
    动态/行为特征（与玩法、时间趋势相关）：
    - log1p(num_reported)：总上报次数的 log
    - log1p(num_hard)：hard mode 上报次数的 log
    - hard_ratio：num_hard / num_reported
    - contest_norm：contest 编号归一化到 [0,1]
    """
    num_reported = df["num_reported"].astype(float).values
    num_hard = df["num_hard"].astype(float).values
    contest = df["contest"].astype(float).values

    log_reported = np.log1p(num_reported)
    log_hard = np.log1p(num_hard)
    hard_ratio = num_hard / np.maximum(num_reported, 1.0)

    c_min, c_max = contest.min(), contest.max()
    if c_max > c_min:
        contest_norm = (contest - c_min) / (c_max - c_min)
    else:
        contest_norm = np.zeros_like(contest)

    dyn = np.stack([log_reported, log_hard, hard_ratio, contest_norm], axis=1)
    return dyn.astype(np.float32)


def compute_weekday_onehot(dates: pd.Series) -> np.ndarray:
    """
    把日期映射为 weekday one-hot（周一=0,...周日=6）。
    """
    weekday = dates.dt.weekday.values  # 0~6
    n = len(weekday)
    onehot = np.zeros((n, 7), dtype=np.float32)
    onehot[np.arange(n), weekday] = 1.0
    return onehot


# ==================== 5. 序列 Dataset ====================

class WordleSeqDataset(Dataset):
    """
    序列 Dataset：
    输入：过去 seq_len 天的 word_id 序列 + 数值特征序列
    输出：第 t 天的目标分布 / avg_tries / succ_rate / label
    """

    def __init__(
        self,
        word_ids: np.ndarray,
        numeric_feats: np.ndarray,
        dists: np.ndarray,
        avg_tries: np.ndarray,
        succ_rate: np.ndarray,
        labels: np.ndarray,
        seq_len: int = SEQ_LEN,
    ):
        super().__init__()
        assert word_ids.ndim == 1
        assert numeric_feats.shape[0] == word_ids.shape[0]
        assert dists.shape[0] == word_ids.shape[0]

        self.word_ids = word_ids
        self.numeric_feats = numeric_feats
        self.dists = dists
        self.avg_tries = avg_tries
        self.succ_rate = succ_rate
        self.labels = labels
        self.seq_len = seq_len
        self.n_total = len(word_ids)

    def __len__(self) -> int:
        # 为了和你之前的实现保持一致，保留 -1，最后一天不作为预测目标
        return self.n_total - self.seq_len - 1

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx
        end = idx + self.seq_len
        target_idx = end

        word_seq = self.word_ids[start:end]            # [L]
        num_seq = self.numeric_feats[start:end, :]     # [L,F]
        target_dist = self.dists[target_idx]           # [7]
        target_avg = self.avg_tries[target_idx]        # 标量
        target_succ = self.succ_rate[target_idx]       # 标量
        target_label = self.labels[target_idx]         # int

        return {
            "word_seq": torch.tensor(word_seq, dtype=torch.long),
            "num_seq": torch.tensor(num_seq, dtype=torch.float32),
            "target_dist": torch.tensor(target_dist, dtype=torch.float32),
            "target_avg": torch.tensor([target_avg], dtype=torch.float32),
            "target_succ": torch.tensor([target_succ], dtype=torch.float32),
            "target_label": torch.tensor(target_label, dtype=torch.long),
        }


# ==================== 6. 总入口：prepare_datasets ====================

def prepare_datasets(
    seq_len: int = SEQ_LEN,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    feature_mode: str = "C",
    save_npz: bool = True,
) -> Tuple[Dict[str, Any], WordleSeqDataset, Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    完整预处理流程：
    - 读取 & 计算目标
    - 计算数据驱动难度阈值（近似等分三档）
    - 构造静态/动态/时间特征（由 feature_mode 决定使用哪部分）
    - 标准化数值特征
    - 构造序列 Dataset + 按时间顺序划分 train/val/test DataLoader

    feature_mode:
      "A" -> 只用文本（word embedding），不加数值特征
      "B" -> 文本 + 静态特征
      "C" -> 文本 + 静态 + 动态 + weekday one-hot
    """
    assert feature_mode in {"A", "B", "C"}

    df = load_raw_data(DATA_CSV_PATH)
    df = compute_distribution_and_targets(df)

    # 7 维分布
    dist_cols = ["p1", "p2", "p3", "p4", "p5", "p6", "pX"]
    dists = df[dist_cols].values.astype(np.float32)

    # avg_tries / succ_rate
    avg_tries = df["avg_tries"].values.astype(np.float32)
    succ_rate = df["succ_rate"].values.astype(np.float32)

    # 数据驱动阈值 + label
    easy_th, med_th = compute_difficulty_thresholds_from_df(df)
    labels = avg_to_label_array(avg_tries, easy_th, med_th)

    # 词表 & word_id
    words = df["word"].astype(str).str.strip().str.upper()
    vocab = sorted(words.unique().tolist())
    word2id = {w: i for i, w in enumerate(vocab)}
    id2word = {i: w for w, i in word2id.items()}
    word_ids = words.map(word2id).values.astype(np.int64)

    # 特征构造
    static_feats = compute_static_feats(words)                # [N,3]
    dynamic_feats = compute_dynamic_feats(df)                 # [N,4]
    weekday_feats = compute_weekday_onehot(df["date"])        # [N,7]

    if feature_mode == "A":
        numeric_feats = np.zeros((len(df), 0), dtype=np.float32)
    elif feature_mode == "B":
        numeric_feats = static_feats
    else:  # "C"
        numeric_feats = np.concatenate(
            [static_feats, dynamic_feats, weekday_feats], axis=1
        )

    # 数值特征标准化
    if numeric_feats.shape[1] > 0:
        scaler = StandardScaler()
        numeric_feats_scaled = scaler.fit_transform(numeric_feats).astype(np.float32)
        scaler_mean = scaler.mean_.tolist()
        scaler_scale = scaler.scale_.tolist()
    else:
        numeric_feats_scaled = numeric_feats
        scaler_mean = []
        scaler_scale = []

    dataset_len = len(df)
    n_samples = dataset_len - seq_len - 1
    indices = np.arange(n_samples)

    n_train = int(n_samples * (1.0 - val_ratio - test_ratio))
    n_val = int(n_samples * val_ratio)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    dataset = WordleSeqDataset(
        word_ids=word_ids,
        numeric_feats=numeric_feats_scaled,
        dists=dists,
        avg_tries=avg_tries,
        succ_rate=succ_rate,
        labels=labels,
        seq_len=seq_len,
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=test_sampler)

    if save_npz:
        os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
        np.savez(
            PROCESSED_DATA_PATH,
            word_ids=word_ids,
            numeric_feats=numeric_feats_scaled,
            dists=dists,
            avg_tries=avg_tries,
            succ_rate=succ_rate,
            labels=labels,
            seq_len=seq_len,
        )

    info: Dict[str, Any] = {
        "vocab_size": len(vocab),
        "num_numeric_features": numeric_feats_scaled.shape[1],
        "seq_len": seq_len,
        "feature_mode": feature_mode,
        "difficulty_thresholds": (easy_th, med_th),
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "word2id": word2id,
        "id2word": id2word,
    }

    return info, dataset, (train_loader, val_loader, test_loader)


if __name__ == "__main__":
    # 1) 先打印一下数据驱动难度分桶的阈值和各档样本数
    compute_data_driven_thresholds_from_csv()

    # 2) 简单测试数据管道
    info, dataset, loaders = prepare_datasets(seq_len=SEQ_LEN, feature_mode="C", save_npz=False)
    print("vocab_size:", info["vocab_size"])
    print("num_numeric_features:", info["num_numeric_features"])
    print("difficulty_thresholds:", info["difficulty_thresholds"])
    print("num_samples (dataset):", len(dataset))

