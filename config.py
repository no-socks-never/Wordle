# config.py
"""
全局配置文件：
- 路径
- 超参数
- 难度阈值
"""

import os

# ===== 路径配置 =====
# 原始 Wordle 数据 CSV 文件路径（你可以改成自己的）
DATA_CSV_PATH = os.path.join(os.path.dirname(__file__), "data/wordle_daily.csv")

# 预处理后保存 npz 的路径（可选，不想保存可以不用）
PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "data/processed_wordle.npz")

# 结果输出路径
RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ===== 序列建模超参数 =====
SEQ_LEN = 14          # 使用过去多少天的记录预测下一天
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
RANDOM_SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ===== 模型相关 =====
WORD_EMB_DIM = 32     # 模型一：随机初始化的词向量维度
HIDDEN_DIM_LSTM = 64
HIDDEN_DIM_TRANSFORMER = 64
NUM_TRANSFORMER_LAYERS = 2
NUM_ATTENTION_HEADS = 4

# ===== 难度标签阈值（来自思路） =====
# avg_tries < 3.5  -> easy
# 3.5 <= avg_tries < 4.2 -> medium
# 4.2 <= avg_tries       -> hard
EASY_THRESHOLD = 3.5
MEDIUM_THRESHOLD = 4.2

# 方便 train_eval 里统一 label 名字
LABEL2ID = {"easy": 0, "medium": 1, "hard": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_CLASSES = len(LABEL2ID)
