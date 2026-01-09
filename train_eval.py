# train_eval.py
"""
训练与评估工具：
- 增加 EMD (Wasserstein Distance) 指标
- 增加全局均值基准计算
- 移除对 config.py 的依赖
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
from scipy.stats import wasserstein_distance # 需要环境中有 scipy

# ===== 原 config.py 常量整合 =====
RESULT_DIR = "results"
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
EASY_THRESHOLD = 3.5
MEDIUM_THRESHOLD = 4.2

os.makedirs(RESULT_DIR, exist_ok=True)

# ==================== 核心辅助函数 ====================

def dist_to_avg_succ(dist_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """从 7 维分布计算 avg_tries 和 succ_rate"""
    device = dist_batch.device
    tries_idx = torch.arange(1, 8, dtype=torch.float32, device=device)
    avg_tries = (dist_batch * tries_idx).sum(axis=1)
    # 成功率为前 6 项之和
    succ_rate = dist_batch[:, :6].sum(axis=1)
    return avg_tries, succ_rate

def compute_emd_batch(pred_dist: torch.Tensor, target_dist: torch.Tensor) -> float:
    """计算批次的平均 Earth Mover's Distance"""
    p_np = pred_dist.detach().cpu().numpy()
    t_np = target_dist.detach().cpu().numpy()
    u_values = np.arange(1, 8) # 尝试次数作为坐标
    
    emds = []
    for i in range(len(p_np)):
        # wasserstein_distance(u_values, v_values, u_weights, v_weights)
        val = wasserstein_distance(u_values, u_values, p_np[i], t_np[i])
        emds.append(val)
    return float(np.mean(emds))

# ==================== 训练与评估逻辑 ====================

def evaluate(model, loader, criterion, device, thresholds=None):
    model.eval()
    all_losses = []
    all_dist_l1 = []
    all_dist_emd = []
    pred_avgs, true_avgs = [], []
    
    with torch.no_grad():
        for batch in loader:
            word_seq = batch["word_seq"].to(device)
            num_seq = batch["num_seq"].to(device)
            target_dist = batch["target_dist"].to(device)
            target_avg = batch["target_avg"].to(device).view(-1)
            
            out = model(word_seq, num_seq)
            loss = criterion(out, target_dist)
            
            # 计算指标
            all_losses.append(loss.item())
            all_dist_l1.append(torch.abs(out - target_dist).sum(dim=1).mean().item())
            all_dist_emd.append(compute_emd_batch(out, target_dist))
            
            avg_p, _ = dist_to_avg_succ(out)
            pred_avgs.extend(avg_p.cpu().numpy())
            true_avgs.extend(target_avg.cpu().numpy())
            
    pred_avgs = np.array(pred_avgs)
    true_avgs = np.array(true_avgs)
    mae_avg = np.mean(np.abs(pred_avgs - true_avgs))
    
    metrics = {
        "loss": np.mean(all_losses),
        "dist_l1": np.mean(all_dist_l1),
        "dist_emd": np.mean(all_dist_emd), # 新增专业指标
        "mae_avg": mae_avg,
    }
    return metrics

def train_one_model(model, train_loader, val_loader, test_loader, model_name, lr=LEARNING_RATE, epochs=NUM_EPOCHS, device="cpu"):
    """执行单个模型的完整训练流程"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.to(device)
    
    best_val_loss = float("inf")
    history = []
    
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch["word_seq"].to(device), batch["num_seq"].to(device))
            loss = criterion(out, batch["target_dist"].to(device))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # 记录
        epoch_log = {"epoch": epoch+1, "train_loss": np.mean(train_losses), "val_loss": val_metrics["loss"]}
        history.append(epoch_log)
        
        # 保存最优
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), os.path.join(RESULT_DIR, f"best_{model_name}.pth"))
            
    # 加载最优并在测试集评估
    model.load_state_dict(torch.load(os.path.join(RESULT_DIR, f"best_{model_name}.pth")))
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    # 保存历史记录 CSV（为了给同学画图用）
    pd.DataFrame(history).to_csv(os.path.join(RESULT_DIR, f"history_{model_name}.csv"), index=False)
    
    return best_val_loss, test_metrics

def get_global_mean_baseline(train_loader, test_loader):
    """
    计算朴素基准：直接用训练集的平均分布预测测试集。
    论文里如果能赢过这个，才说明深度学习有用。
    """
    all_train_dists = []
    for batch in train_loader:
        all_train_dists.append(batch["target_dist"].numpy())
    mean_dist = np.concatenate(all_train_dists, axis=0).mean(axis=0)
    
    # 在测试集上算误差
    test_l1 = []
    test_mae = []
    tries_idx = np.arange(1, 8)
    mean_avg = (mean_dist * tries_idx).sum()
    
    for batch in test_loader:
        target_dist = batch["target_dist"].numpy()
        target_avg = batch["target_avg"].numpy().flatten()
        # L1
        test_l1.append(np.abs(mean_dist - target_dist).sum(axis=1).mean())
        # MAE
        test_mae.append(np.abs(mean_avg - target_avg).mean())
        
    return {"baseline_l1": np.mean(test_l1), "baseline_mae": np.mean(test_mae)}