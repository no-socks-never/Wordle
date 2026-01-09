import pandas as pd
import numpy as np
from pathlib import Path

# 读取所有hparam文件
results_dir = Path('results')
files = {
    'LSTM': results_dir / 'LSTM' / 'hparam_results_LSTM.csv',
    'BiLSTM': results_dir / 'BiLSTM' / 'hparam_results_BiLSTM.csv',
    'BiLSTM_ablation': results_dir / 'BiLSTM_ablation' / 'hparam_results_BiLSTM_ablation.csv',
    'Transformer': results_dir / 'Transformer' / 'hparam_results_Transformer.csv'
}

# 读取数据
dataframes = {}
for name, path in files.items():
    if path.exists():
        df = pd.read_csv(path)
        df['model_type'] = name
        dataframes[name] = df

# 合并所有数据
all_data = pd.concat(dataframes.values(), ignore_index=True)

# 创建输出目录
output_dir = Path('model_analysis')
output_dir.mkdir(exist_ok=True)

# 生成分析报告
report = []
report.append("="*100)
report.append("模型选择分析报告 - 基于验证集指标")
report.append("="*100)
report.append("")
report.append("【重要说明】")
report.append("根据机器学习最佳实践：")
report.append("1. 验证集（Validation Set）：用于模型选择和超参数调优")
report.append("2. 测试集（Test Set）：仅用于最终性能评估，不应用于选择模型")
report.append("3. 因此，选择模型应该基于验证集指标（best_val_loss等）")
report.append("")

# 检查验证集数据
report.append("【验证集数据检查】")
if 'best_val_loss' in all_data.columns:
    report.append("✓ 发现验证集指标：best_val_loss（验证集上的最佳损失）")
    report.append("")
    
    # 检查是否有其他验证集指标
    val_columns = [col for col in all_data.columns if 'val' in col.lower()]
    if val_columns:
        report.append(f"验证集相关列: {', '.join(val_columns)}")
    report.append("")
else:
    report.append("✗ 未发现验证集指标")
    report.append("")

# 测试集指标
test_columns = [col for col in all_data.columns if 'test' in col.lower()]
report.append(f"测试集指标: {', '.join(test_columns)}")
report.append("")

# 1. 数据概览
report.append("【一、数据概览】")
report.append(f"总实验数量: {len(all_data)}")
report.append(f"模型类型: {', '.join(all_data['model_type'].unique())}")
report.append("")
report.append("各模型实验数量:")
for model, count in all_data['model_type'].value_counts().items():
    report.append(f"  - {model}: {count} 个实验")
report.append("")

# 2. 基于验证集选择最优模型
report.append("【二、基于验证集的最优模型选择】")
report.append("-" * 100)

if 'best_val_loss' in all_data.columns:
    # 按验证集损失排序
    report.append("\n按验证集损失（best_val_loss）排序 - 越小越好：")
    report.append("")
    
    # Top 10 验证集损失最低的模型
    top_val_loss = all_data.nsmallest(10, 'best_val_loss')[
        ['model_type', 'seq_len', 'hidden_dim', 'lr', 'feature_mode', 
         'best_val_loss', 'test_loss', 'test_acc', 'test_macro_f1']
    ].copy()
    
    for idx, (i, row) in enumerate(top_val_loss.iterrows(), 1):
        report.append(f"排名 {idx}:")
        report.append(f"  模型: {row['model_type']}")
        report.append(f"  配置: seq_len={row['seq_len']}, hidden_dim={row['hidden_dim']}, "
                     f"lr={row['lr']}, feature_mode={row['feature_mode']}")
        report.append(f"  验证集损失: {row['best_val_loss']:.6f} ⭐ (选择依据)")
        report.append(f"  测试集损失: {row['test_loss']:.6f}")
        report.append(f"  测试集准确率: {row['test_acc']:.4f}")
        report.append(f"  测试集F1分数: {row['test_macro_f1']:.4f}")
        report.append("")
    
    # 最优模型
    best_by_val = all_data.loc[all_data['best_val_loss'].idxmin()]
    report.append("="*100)
    report.append("★ 基于验证集的最优模型（推荐）:")
    report.append("="*100)
    report.append(f"模型类型: {best_by_val['model_type']}")
    report.append(f"超参数配置:")
    report.append(f"  - 序列长度 (seq_len): {best_by_val['seq_len']}")
    report.append(f"  - 隐藏层维度 (hidden_dim): {best_by_val['hidden_dim']}")
    report.append(f"  - 学习率 (lr): {best_by_val['lr']}")
    report.append(f"  - 特征模式 (feature_mode): {best_by_val['feature_mode']}")
    report.append("")
    report.append("验证集性能（模型选择依据）:")
    report.append(f"  - 最佳验证集损失: {best_by_val['best_val_loss']:.6f} ⭐")
    report.append("")
    report.append("测试集性能（最终评估）:")
    report.append(f"  - 测试集损失: {best_by_val['test_loss']:.6f}")
    report.append(f"  - 测试集准确率: {best_by_val['test_acc']:.4f}")
    report.append(f"  - 测试集F1分数: {best_by_val['test_macro_f1']:.4f}")
    report.append(f"  - 测试集平均MAE: {best_by_val['test_mae_avg']:.4f}")
    report.append(f"  - 测试集平均RMSE: {best_by_val['test_rmse_avg']:.4f}")
    report.append("")
    
    # 各模型的最佳验证集损失
    report.append("【三、各模型的最佳验证集性能】")
    report.append("-" * 100)
    for model_type in sorted(all_data['model_type'].unique()):
        model_data = all_data[all_data['model_type'] == model_type]
        best_model = model_data.loc[model_data['best_val_loss'].idxmin()]
        report.append(f"\n{model_type}:")
        report.append(f"  最佳验证集损失: {best_model['best_val_loss']:.6f}")
        report.append(f"  对应配置: seq_len={best_model['seq_len']}, "
                     f"hidden_dim={best_model['hidden_dim']}, "
                     f"lr={best_model['lr']}, "
                     f"feature_mode={best_model['feature_mode']}")
        report.append(f"  测试集表现: 损失={best_model['test_loss']:.6f}, "
                     f"准确率={best_model['test_acc']:.4f}, "
                     f"F1={best_model['test_macro_f1']:.4f}")
    
    # 保存基于验证集选择的结果
    top_val_loss.to_csv(output_dir / 'top_10_by_validation.csv', index=False)
    
else:
    report.append("⚠️ 警告：未找到验证集指标（best_val_loss）")
    report.append("将使用测试集指标进行分析（不推荐，可能导致过拟合测试集）")
    report.append("")

# 3. 模型内部参数影响（基于验证集）
report.append("\n【四、模型内部参数影响分析（基于验证集）】")
report.append("-" * 100)

for model_type in sorted(all_data['model_type'].unique()):
    model_data = all_data[all_data['model_type'] == model_type]
    report.append(f"\n--- {model_type} 模型 ---")
    
    if 'best_val_loss' in model_data.columns:
        # 序列长度影响
        if 'seq_len' in model_data.columns and model_data['seq_len'].nunique() > 1:
            seq_impact = model_data.groupby('seq_len').agg({
                'best_val_loss': 'mean',
                'test_acc': 'mean',
                'test_macro_f1': 'mean'
            }).round(4)
            report.append(f"\n序列长度 (seq_len) 对验证集损失的影响:")
            for seq_len in sorted(seq_impact.index):
                report.append(f"  seq_len={seq_len}: "
                             f"验证集损失={seq_impact.loc[seq_len, 'best_val_loss']:.6f}, "
                             f"测试集准确率={seq_impact.loc[seq_len, 'test_acc']:.4f}")
            best_seq = seq_impact['best_val_loss'].idxmin()
            report.append(f"  → 最佳序列长度（验证集）: {best_seq}")
        
        # 隐藏层维度影响
        if 'hidden_dim' in model_data.columns and model_data['hidden_dim'].nunique() > 1:
            hidden_impact = model_data.groupby('hidden_dim').agg({
                'best_val_loss': 'mean',
                'test_acc': 'mean',
                'test_macro_f1': 'mean'
            }).round(4)
            report.append(f"\n隐藏层维度 (hidden_dim) 对验证集损失的影响:")
            for hidden_dim in sorted(hidden_impact.index):
                report.append(f"  hidden_dim={hidden_dim}: "
                             f"验证集损失={hidden_impact.loc[hidden_dim, 'best_val_loss']:.6f}, "
                             f"测试集准确率={hidden_impact.loc[hidden_dim, 'test_acc']:.4f}")
            best_hidden = hidden_impact['best_val_loss'].idxmin()
            report.append(f"  → 最佳隐藏层维度（验证集）: {best_hidden}")
        
        # 学习率影响
        if 'lr' in model_data.columns and model_data['lr'].nunique() > 1:
            lr_impact = model_data.groupby('lr').agg({
                'best_val_loss': 'mean',
                'test_acc': 'mean',
                'test_macro_f1': 'mean'
            }).round(4)
            report.append(f"\n学习率 (lr) 对验证集损失的影响:")
            for lr in sorted(lr_impact.index):
                report.append(f"  lr={lr}: "
                             f"验证集损失={lr_impact.loc[lr, 'best_val_loss']:.6f}, "
                             f"测试集准确率={lr_impact.loc[lr, 'test_acc']:.4f}")
            best_lr = lr_impact['best_val_loss'].idxmin()
            report.append(f"  → 最佳学习率（验证集）: {best_lr}")
        
        # 特征模式影响
        if 'feature_mode' in model_data.columns and model_data['feature_mode'].nunique() > 1:
            feat_impact = model_data.groupby('feature_mode').agg({
                'best_val_loss': 'mean',
                'test_acc': 'mean',
                'test_macro_f1': 'mean'
            }).round(4)
            report.append(f"\n特征模式 (feature_mode) 对验证集损失的影响:")
            for feat_mode in sorted(feat_impact.index):
                report.append(f"  feature_mode={feat_mode}: "
                             f"验证集损失={feat_impact.loc[feat_mode, 'best_val_loss']:.6f}, "
                             f"测试集准确率={feat_impact.loc[feat_mode, 'test_acc']:.4f}")
            best_feat = feat_impact['best_val_loss'].idxmin()
            report.append(f"  → 最佳特征模式（验证集）: {best_feat}")

# 4. 总结
report.append("\n\n【五、总结与建议】")
report.append("-" * 100)
report.append("")
report.append("1. 模型选择原则:")
report.append("   ✓ 使用验证集指标（best_val_loss）选择模型")
report.append("   ✓ 测试集指标仅用于最终性能报告")
report.append("   ✗ 不要基于测试集指标选择模型（会导致过拟合测试集）")
report.append("")
report.append("2. 推荐做法:")
if 'best_val_loss' in all_data.columns:
    best_by_val = all_data.loc[all_data['best_val_loss'].idxmin()]
    report.append(f"   选择验证集损失最低的模型:")
    report.append(f"   - 模型: {best_by_val['model_type']}")
    report.append(f"   - 配置: seq_len={best_by_val['seq_len']}, "
                 f"hidden_dim={best_by_val['hidden_dim']}, "
                 f"lr={best_by_val['lr']}, "
                 f"feature_mode={best_by_val['feature_mode']}")
    report.append(f"   - 验证集损失: {best_by_val['best_val_loss']:.6f}")
    report.append(f"   - 测试集表现: 准确率={best_by_val['test_acc']:.4f}, "
                 f"F1={best_by_val['test_macro_f1']:.4f}")
report.append("")
report.append("3. 注意事项:")
report.append("   - 验证集和测试集性能可能有差异，这是正常的")
report.append("   - 如果差异过大，可能需要检查数据分布或模型泛化能力")
report.append("")

report.append("="*100)

# 保存报告
report_text = "\n".join(report)
with open(output_dir / 'model_selection_report.txt', 'w', encoding='utf-8') as f:
    f.write(report_text)

# 保存数据
all_data.to_csv(output_dir / 'all_models_with_validation.csv', index=False)

# 打印报告
print(report_text)

print(f"\n\n分析结果已保存到 {output_dir} 目录:")
print(f"  - model_selection_report.txt: 模型选择分析报告")
print(f"  - all_models_with_validation.csv: 所有模型数据（含验证集指标）")
if 'best_val_loss' in all_data.columns:
    print(f"  - top_10_by_validation.csv: 基于验证集的Top 10模型")

