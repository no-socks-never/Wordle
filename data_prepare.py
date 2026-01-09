# data_prepare.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from sklearn.preprocessing import StandardScaler

# ===== 路径配置（请根据你的实际情况修改路径） =====
DATA_CSV_PATH = "data/wordle_daily.csv" 
WORD_FREQ_CSV = "data/wordle_word_freq_from_subtlex.csv" 
RESULT_DIR = "results"
BATCH_SIZE = 32
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ==================== 特征提取核心逻辑 ====================

def load_and_clean_data():
    df = pd.read_csv(DATA_CSV_PATH)
    # 统一列名
    df.columns = [c.strip() for c in df.columns]
    rename_map = {
        "Date": "date", "Contest number": "contest", "Word": "word",
        "Number of reported results": "num_reported", "Number in hard mode": "num_hard",
        "1 try": "c1", "2 tries": "c2", "3 tries": "c3", "4 tries": "c4", 
        "5 tries": "c5", "6 tries": "c6", "7 or more tries (X)": "cX",
    }
    df = df.rename(columns=rename_map)
    df["word"] = df["word"].astype(str).str.strip().str.upper()
    
    # 【核心修复】：过滤掉 rprobe (6) 和 clen (4) 等异常长度单词
    # 只有长度为5的纯字母单词才能进入模型训练
    df = df[df["word"].str.match(r'^[A-Z]{5}$')].copy()
    
    # 计算 7 维分布 [cite: 1186-1187]
    cols = ["c1", "c2", "c3", "c4", "c5", "c6", "cX"]
    counts = df[cols].values.astype(float)
    probs = counts / counts.sum(axis=1, keepdims=True)
    
    # 派生指标 [cite: 1189, 1191]
    tries_idx = np.arange(1, 8)
    df["avg_tries"] = (probs * tries_idx).sum(axis=1)
    df["succ_rate"] = 1.0 - probs[:, -1]
    df["p_dist"] = probs.tolist()
    
    return df.sort_values("contest").reset_index(drop=True)

def compute_features(df, mode="C"):
    """构造 Mode A/B/C 三种特征组合"""
    df["log_reported"] = np.log1p(df["num_reported"])
    # Mode A: 基础环境特征
    feats = df[["log_reported"]].copy()
    
    if mode in ["B", "C"]:
        # 静态语言学特征 [cite: 1200-1203]
        # 词频特征（由于文件可能不在，这里加了 try-except 保护）
        try:
            freq_df = pd.read_csv(WORD_FREQ_CSV)
            freq_dict = dict(zip(freq_df["word"].str.upper(), freq_df["Lg10CD"]))
            df["f1_freq"] = df["word"].map(lambda x: freq_dict.get(x, 0.301))
        except:
            df["f1_freq"] = 0.301 # 默认值
            
        df["f2_dist"] = df["word"].map(lambda x: len(set(x)))
        high_freq_inits = set("SCPTAMBRLD")
        df["f3_init"] = df["word"].map(lambda x: 1 if x[0] in high_freq_inits else 0)
        feats = pd.concat([feats, df[["f1_freq", "f2_dist", "f3_init"]]], axis=1)
        
    if mode == "C":
        # 动态环境特征 [cite: 1206]
        df["hard_ratio"] = df["num_hard"] / df["num_reported"]
        df["contest_norm"] = df["contest"] / df["contest"].max()
        df["date"] = pd.to_datetime(df["date"])
        weekdays = pd.get_dummies(df["date"].dt.weekday, prefix="day")
        feats = pd.concat([feats, df[["hard_ratio", "contest_norm"]], weekdays], axis=1)
        
    return feats.values.astype(np.float32)

class WordleSeqDataset(Dataset):
    def __init__(self, word_ids, numeric_feats, dists, avg_tries, seq_len):
        self.word_ids = word_ids
        self.numeric_feats = numeric_feats
        self.dists = dists
        self.avg_tries = avg_tries
        self.seq_len = seq_len

    def __len__(self):
        # 确保数据集长度扣除 seq_len 还有剩余
        return len(self.word_ids) - self.seq_len

    def __getitem__(self, idx):
        target_idx = idx + self.seq_len
        # 输入序列：若 seq_len=0，则输入仅含目标词当天
        if self.seq_len > 0:
            w_in = self.word_ids[idx:target_idx]
            n_in = self.numeric_feats[idx:target_idx]
        else:
            w_in = self.word_ids[target_idx:target_idx+1]
            n_in = self.numeric_feats[target_idx:target_idx+1]
            
        return {
            "word_seq": torch.tensor(w_in, dtype=torch.long),
            "num_seq": torch.tensor(n_in, dtype=torch.float32),
            "target_dist": torch.tensor(self.dists[target_idx], dtype=torch.float32),
            "target_avg": torch.tensor([self.avg_tries[target_idx]], dtype=torch.float32),
        }

def prepare_datasets(seq_len=14, feature_mode="C"):
    df = load_and_clean_data()
    
    # 单词编码：[ord(c)-ord('A')]
    def word_to_ids(w): return [ord(c) - ord('A') for c in w]
    all_word_ids = np.array([word_to_ids(w) for w in df["word"]]) # 这里现在是整齐的 [N, 5] 矩阵了！
    
    numeric_feats = compute_features(df, mode=feature_mode)
    scaler = StandardScaler()
    numeric_feats = scaler.fit_transform(numeric_feats)
    
    dataset = WordleSeqDataset(
        all_word_ids, numeric_feats, 
        np.stack(df["p_dist"].values), df["avg_tries"].values, seq_len
    )
    
    # 按时间顺序切分
    n = len(dataset)
    val_size = int(n * VAL_RATIO)
    test_size = int(n * TEST_RATIO)
    train_size = n - val_size - test_size
    
    indices = list(range(n))
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(train_idx))
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(val_idx))
    test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=SubsetRandomSampler(test_idx))
    
    return {"vocab_size": 26, "num_numeric_features": numeric_feats.shape[1]}, dataset, (train_loader, val_loader, test_loader)