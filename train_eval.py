# train_eval.py
"""
训练与评估工具：
- train_one_epoch / evaluate
- train_one_model: 训练 + 验证集选最优 + 在 test 上评估
  同时保存：
    - history CSV（train/val loss & val mae/acc）
    - loss 曲线图
    - test 集预测 CSV
    - avg_tries 趋势图
    - BiLSTM attention 热力图（随机取一条样本）
"""

import os
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score

from config import (
    RESULT_DIR,
    NUM_EPOCHS,
    LEARNING_RATE,
    EASY_THRESHOLD,
    MEDIUM_THRESHOLD,
)

os.makedirs(RESULT_DIR, exist_ok=True)


# ==================== 一些辅助函数 ====================

def dist_to_avg_succ(dist_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """从 7 维分布计算 avg_tries 和 succ_rate。"""
    device = dist_batch.device
    tries_idx = torch.arange(1, 8, dtype=torch.float32, device=device)  # 1..7
    avg_tries_pred = (dist_batch * tries_idx).sum(dim=1)
    succ_rate_pred = 1.0 - dist_batch[:, -1]
    return avg_tries_pred, succ_rate_pred


def avg_to_label(avg_tries, thresholds=None):
    """
    把 avg_tries 转为离散难度标签 0/1/2。
    默认使用 config 中的 EASY/MEDIUM_THRESHOLD；
    若 thresholds=(easy_th, med_th) 被显式传入，则使用数据驱动阈值。
    """
    if isinstance(avg_tries, torch.Tensor):
        avg_tries = avg_tries.detach().cpu().numpy()
    avg_tries = np.asarray(avg_tries, dtype=float)

    if thresholds is None:
        easy_th, med_th = EASY_THRESHOLD, MEDIUM_THRESHOLD
    else:
        easy_th, med_th = thresholds

    labels = np.zeros_like(avg_tries, dtype=int)
    labels[avg_tries >= easy_th] = 1
    labels[avg_tries >= med_th] = 2
    return labels


# ==================== 画图工具（支持自定义 result_dir） ====================

def plot_loss_curve(history: Dict[str, list], model_name: str, result_dir: str):
    """只保存 history 为 CSV，不再画 loss 曲线图。"""
    df = pd.DataFrame(history)
    csv_path = os.path.join(result_dir, f"{model_name}_history.csv")
    df.to_csv(csv_path, index=False)
    # 不再绘制和保存 png 图

def plot_avg_trend(true_avg, pred_avg, model_name: str, result_dir: str, split_tag: str = "test"):
    """
    （已禁用）原本用于画 avg_tries 趋势图。
    现在不再生成图片，保留空函数以兼容调用。
    """
    return

def plot_attention_heatmap(att_weights, model_name: str, result_dir: str):
    """
    （已禁用）原本用于画 BiLSTM 的 attention 热力图。
    现在不再生成图片，保留空函数以兼容调用。
    """
    return



# ==================== 训练 & 评估 ====================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0.0
    n_samples = 0

    for batch in loader:
        word_seq = batch["word_seq"].to(device)
        num_seq = batch["num_seq"].to(device)
        target_dist = batch["target_dist"].to(device)

        optimizer.zero_grad()
        outputs = model(word_seq, num_seq)
        if isinstance(outputs, tuple):
            pred_dist, _ = outputs
        else:
            pred_dist = outputs

        loss = criterion(pred_dist, target_dist)
        loss.backward()
        optimizer.step()

        batch_size = word_seq.size(0)
        epoch_loss += loss.item() * batch_size
        n_samples += batch_size

    epoch_loss /= max(n_samples, 1)
    return epoch_loss


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
    thresholds=None,
):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    all_true_avg = []
    all_pred_avg = []
    all_true_succ = []
    all_pred_succ = []

    # 新增：分布层面的误差
    all_dist_l1 = []   # 每天的 sum_j |p_j - p_hat_j|
    all_dist_kl = []   # 每天的 KL(p || p_hat)

    eps = 1e-8

    for batch in loader:
        word_seq = batch["word_seq"].to(device)
        num_seq = batch["num_seq"].to(device)
        target_dist = batch["target_dist"].to(device)      # [B,7]
        target_avg = batch["target_avg"].to(device)        # [B,1]
        target_succ = batch["target_succ"].to(device)      # [B,1]

        outputs = model(word_seq, num_seq)
        if isinstance(outputs, tuple):
            pred_dist, _ = outputs
        else:
            pred_dist = outputs                            # [B,7] 概率分布

        # 训练时同一个 loss：分布的 MSE
        loss = criterion(pred_dist, target_dist)

        batch_size = word_seq.size(0)
        total_loss += loss.item() * batch_size
        n_samples += batch_size

        # 标量：平均尝试次数和成功率
        avg_pred, succ_pred = dist_to_avg_succ(pred_dist)

        all_true_avg.append(target_avg.cpu().numpy())      # [B,1]
        all_pred_avg.append(avg_pred.cpu().numpy())        # [B]
        all_true_succ.append(target_succ.cpu().numpy())    # [B,1]
        all_pred_succ.append(succ_pred.cpu().numpy())      # [B]

        # 新增：分布层误差
        # L1：sum_j |p_j - p_hat_j|
        dist_l1_batch = torch.abs(pred_dist - target_dist).sum(dim=1)   # [B]
        all_dist_l1.append(dist_l1_batch.detach().cpu().numpy())

        # KL：sum_j p_j * log(p_j / p_hat_j)
        kl_batch = (target_dist * (
            (target_dist + eps).log() - (pred_dist + eps).log()
        )).sum(dim=1)                                     # [B]
        all_dist_kl.append(kl_batch.detach().cpu().numpy())

    total_loss /= max(n_samples, 1)

    true_avg = np.concatenate(all_true_avg, axis=0).squeeze(-1)
    pred_avg = np.concatenate(all_pred_avg, axis=0)
    true_succ = np.concatenate(all_true_succ, axis=0).squeeze(-1)
    pred_succ = np.concatenate(all_pred_succ, axis=0)

    dist_l1 = np.concatenate(all_dist_l1, axis=0)
    dist_kl = np.concatenate(all_dist_kl, axis=0)

    # 分类指标：基于 avg_tries 做三档分桶
    true_labels = avg_to_label(true_avg, thresholds=thresholds)
    pred_labels = avg_to_label(pred_avg, thresholds=thresholds)

    acc = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average="macro")

    mae_avg = np.mean(np.abs(pred_avg - true_avg))
    rmse_avg = float(np.sqrt(np.mean((pred_avg - true_avg) ** 2)))
    mae_succ = np.mean(np.abs(pred_succ - true_succ))
    rmse_succ = float(np.sqrt(np.mean((pred_succ - true_succ) ** 2)))

    metrics = {
        "loss": total_loss,
        "mae_avg": float(mae_avg),
        "rmse_avg": rmse_avg,
        "mae_succ": float(mae_succ),
        "rmse_succ": rmse_succ,
        "acc": float(acc),
        "macro_f1": float(macro_f1),
        # 新增：分布层指标
        "dist_l1": float(dist_l1.mean()),
        "dist_kl": float(dist_kl.mean()),
    }

    details = {
        "true_avg": true_avg,
        "pred_avg": pred_avg,
        "true_succ": true_succ,
        "pred_succ": pred_succ,
        "true_labels": true_labels,
        "pred_labels": pred_labels,
        "dist_l1": dist_l1,
        "dist_kl": dist_kl,
    }

    return metrics, details



def train_one_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    model_name: str,
    device=None,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LEARNING_RATE,
    thresholds=None,
    result_subdir: str = None,
    return_val: bool = False,
):
    """
    训练一个模型：
    - 每个 epoch 记录 train_loss & val_loss & val_mae
    - 选取 val_loss 最优的参数
    - 用 best 参数在 test 集上评估
    - 把 history / test 预测 / avg 趋势 / attention 图 保存到 result_subdir 下面
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_result_dir = RESULT_DIR
    if result_subdir is not None:
        result_dir = os.path.join(base_result_dir, result_subdir)
    else:
        result_dir = base_result_dir
    os.makedirs(result_dir, exist_ok=True)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_mae_avg": [],
        "val_acc": [],
    }

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = evaluate(model, val_loader, criterion, device, thresholds=thresholds)
        val_loss = val_metrics["loss"]

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_avg"].append(val_metrics["mae_avg"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"[{model_name}] Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MAE(avg): {val_metrics['mae_avg']:.4f} | "
            f"Val Acc: {val_metrics['acc']:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }

    # 保存 loss 曲线
    plot_loss_curve(history, model_name, result_dir)

    # 载入最优参数并在 test 上评估
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    test_metrics, test_details = evaluate(
        model, test_loader, criterion, device, thresholds=thresholds
    )

    print(f"\n[{model_name}] Test metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # 保存 test 预测 CSV，方便后续可视化
    test_df = pd.DataFrame({
        "true_avg": test_details["true_avg"],
        "pred_avg": test_details["pred_avg"],
        "true_succ": test_details["true_succ"],
        "pred_succ": test_details["pred_succ"],
        "true_label": test_details["true_labels"],
        "pred_label": test_details["pred_labels"],
    })
    csv_path = os.path.join(result_dir, f"{model_name}_test_preds.csv")
    test_df.to_csv(csv_path, index=False)

    # 保存 avg_tries 趋势图
    plot_avg_trend(
        test_details["true_avg"],
        test_details["pred_avg"],
        model_name,
        result_dir,
        split_tag="test",
    )

    # 如果是 BiLSTM 模型，额外画 attention 热力图
    with torch.no_grad():
        try:
            for batch in test_loader:
                word_seq = batch["word_seq"].to(device)
                num_seq = batch["num_seq"].to(device)
                outputs = model(word_seq, num_seq)
                if isinstance(outputs, tuple):
                    _, att_weights = outputs
                    # 只取一条样本画图
                    att_w = att_weights[0].detach().cpu().numpy()
                    plot_attention_heatmap(att_w, model_name, result_dir)
                break
        except Exception:
            # 非 BiLSTM 或其它原因失败时直接略过
            pass

    if return_val:
        return best_val_loss, test_metrics
    else:
        return test_metrics

