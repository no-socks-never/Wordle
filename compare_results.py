# compare_results.py
"""
最终结果对比脚本：
- 自动读取 LSTM, BiLSTM, Transformer 的搜索结果
- 整合全局均值基准 (Baseline)
- 生成论文中最终的大对比表
"""

import os
import pandas as pd
import ast

# 基础路径
RESULT_DIR = "results"

def get_best_config(model_name):
    path = os.path.join(RESULT_DIR, f"hparam_results_{model_name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # 按照验证集损失选出该模型最强的配置
    best_idx = df["best_val_loss"].idxmin()
    return df.loc[best_idx]

def generate_final_report():
    print(">>> 正在生成最终实验对比报告...")
    
    # 1. 获取各模型最优表现
    summary = []
    for m in ["LSTM", "BiLSTM", "Transformer"]:
        best = get_best_config(m)
        if best is not None:
            summary.append({
                "Model": m,
                "Best_Config": f"seq{best['seq_len']}_h{best['hidden_dim']}_mode{best['feature_mode']}",
                "Test_MAE_Avg": f"{best['test_mae_avg']:.4f}",
                "Test_L1_Dist": f"{best['test_dist_l1']:.4f}",
                "Test_EMD": f"{best['test_dist_emd']:.4f}" # 加上我们新算的专业指标
            })
    
    # 2. 读取 Baseline 数据
    baseline_path = os.path.join(RESULT_DIR, "baseline.txt")
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            b_dict = ast.literal_eval(f.read())
        summary.append({
            "Model": "Baseline (Global Mean)",
            "Best_Config": "N/A",
            "Test_MAE_Avg": f"{b_dict['baseline_mae']:.4f}",
            "Test_L1_Dist": f"{b_dict['baseline_l1']:.4f}",
            "Test_EMD": "N/A"
        })
    
    # 3. 生成展示表格
    report_df = pd.DataFrame(summary)
    print("\n【核心性能对比表 (用于论文第 5 章)】")
    print(report_df.to_string(index=False))
    
    # 保存为 CSV
    report_df.to_csv(os.path.join(RESULT_DIR, "final_performance_comparison.csv"), index=False)
    print(f"\n[完成] 报告已保存至 {RESULT_DIR}/final_performance_comparison.csv")

if __name__ == "__main__":
    generate_final_report()