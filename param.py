# param.py
"""
超参数搜索与消融实验主控脚本：
- 移除 config.py 依赖
- 执行 LSTM / BiLSTM / Transformer 的网格搜索
- 自动计算 Global Mean Baseline
- 支持消融实验 (SEQ_LEN=0)
"""

import os
import pandas as pd
import torch
from itertools import product
from tqdm import tqdm

# 引入我们重构后的模块
from data_prepare import prepare_datasets, RESULT_DIR
from models import BaseLSTMModel, BiLSTMAttentionModel, TransformerTimeModel
from train_eval import train_one_model, get_global_mean_baseline

# ===== 实验配置 (替代原 config.py) =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRID_SEQ_LEN = [0, 10,14]            # 0: 消融实验（无历史）；14: 完整实验
GRID_HIDDEN = [64, 128]
GRID_LR = [1e-3, 5e-4]
GRID_FEATURE_MODE = ["A", "B", "C"]
NUM_EPOCHS = 50

# ==================== 自动化搜索逻辑 ====================

def run_sweep(model_type="Transformer"):
    print(f"\n>>> 开始 {model_type} 模型的超参数搜索与消融实验...")
    records = []
    
    # 组合搜索空间
    search_space = list(product(GRID_SEQ_LEN, GRID_HIDDEN, GRID_LR, GRID_FEATURE_MODE))
    
    for seq_len, hidden_dim, lr, feat_mode in tqdm(search_space):
        # 1. 准备数据
        info, _, (train_loader, val_loader, test_loader) = prepare_datasets(
            seq_len=seq_len, 
            feature_mode=feat_mode
        )
        
        # 2. 构造模型
        vocab_size = info["vocab_size"]
        num_numeric = info["num_numeric_features"]
        
        if model_type == "LSTM":
            model = BaseLSTMModel(vocab_size, num_numeric, hidden_dim=hidden_dim)
        elif model_type == "BiLSTM":
            model = BiLSTMAttentionModel(vocab_size, num_numeric, hidden_dim=hidden_dim)
        else: # Transformer
            model = TransformerTimeModel(vocab_size, num_numeric, d_model=hidden_dim)
            
        # 3. 训练并评估
        model_id = f"{model_type}_seq{seq_len}_h{hidden_dim}_lr{lr}_mode{feat_mode}"
        best_val_loss, test_metrics = train_one_model(
            model, train_loader, val_loader, test_loader, 
            model_name=model_id, lr=lr, epochs=NUM_EPOCHS, device=DEVICE
        )
        
        # 4. 记录结果
        res = {
            "model_type": model_type,
            "seq_len": seq_len,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "feature_mode": feat_mode,
            "best_val_loss": best_val_loss
        }
        res.update({f"test_{k}": v for k, v in test_metrics.items()})
        records.append(res)
        
    # 保存结果（格式与原代码一致，确保不破坏同学画图的代码）
    df = pd.DataFrame(records)
    out_path = os.path.join(RESULT_DIR, f"hparam_results_{model_type}.csv")
    df.to_csv(out_path, index=False)
    return df

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 第一步：计算并打印 Baseline
    # 先随便初始化一个 dataset 获取 loader
    print("正在计算全局均值基准 (Baseline)...")
    info, _, (tr, _, ts) = prepare_datasets(seq_len=14, feature_mode="A")
    baseline = get_global_mean_baseline(tr, ts)
    print(f"Baseline 指标: L1={baseline['baseline_l1']:.4f}, MAE={baseline['baseline_mae']:.4f}")
    
    # 将 Baseline 写入一个小文件，方便你在论文里引用
    with open(os.path.join(RESULT_DIR, "baseline.txt"), "w") as f:
        f.write(str(baseline))

    # 第二步：依次跑三个模型
    for m in ["LSTM", "BiLSTM", "Transformer"]:
        run_sweep(model_type=m)
        
    print("\n[完成] 所有实验已结束，结果保存在 results/ 目录下。")
    

