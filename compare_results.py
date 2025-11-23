# compare_results.py
"""
对比三个模型的调参结果，生成对比报告。
"""

import os
import pandas as pd
from config import RESULT_DIR


def compare_models():
    """对比三个模型的调参结果，找出最佳配置并生成对比报告。"""
    
    # 加载三个模型的调参结果
    lstm_path = os.path.join(RESULT_DIR, "hparam_results_LSTM.csv")
    bilstm_path = os.path.join(RESULT_DIR, "hparam_results_BiLSTM.csv")
    trans_path = os.path.join(RESULT_DIR, "hparam_results_Transformer.csv")
    
    results = {}
    
    # 读取LSTM结果
    if os.path.exists(lstm_path):
        df_lstm = pd.read_csv(lstm_path)
        best_lstm = df_lstm.loc[df_lstm['val_loss'].idxmin()]
        results['LSTM'] = {
            'best': best_lstm,
            'all': df_lstm
        }
    else:
        print(f"Warning: {lstm_path} not found. Run param.py first.")
        return
    
    # 读取BiLSTM结果
    if os.path.exists(bilstm_path):
        df_bilstm = pd.read_csv(bilstm_path)
        best_bilstm = df_bilstm.loc[df_bilstm['val_loss'].idxmin()]
        results['BiLSTM'] = {
            'best': best_bilstm,
            'all': df_bilstm
        }
    else:
        print(f"Warning: {bilstm_path} not found. Run param.py first.")
        return
    
    # 读取Transformer结果
    if os.path.exists(trans_path):
        df_trans = pd.read_csv(trans_path)
        best_trans = df_trans.loc[df_trans['val_loss'].idxmin()]
        results['Transformer'] = {
            'best': best_trans,
            'all': df_trans
        }
    else:
        print(f"Warning: {trans_path} not found. Run param.py first.")
        return
    
    # 生成对比报告
    print("\n" + "="*80)
    print("模型对比报告")
    print("="*80)
    
    # 1. 最佳配置对比
    print("\n【1. 最佳模型配置（基于验证集最小损失）】\n")
    
    comparison_df = pd.DataFrame({
        'LSTM': results['LSTM']['best'][['model', 'seq_len', 'hidden_dim', 'lr', 'use_static', 
                                         'val_loss', 'test_loss', 'test_mae_avg', 'test_acc', 'test_macro_f1']],
        'BiLSTM': results['BiLSTM']['best'][['model', 'seq_len', 'hidden_dim', 'lr', 'use_static', 
                                              'val_loss', 'test_loss', 'test_mae_avg', 'test_acc', 'test_macro_f1']],
        'Transformer': results['Transformer']['best'][['model', 'seq_len', 'd_model', 'lr', 'use_static', 
                                                        'val_loss', 'test_loss', 'test_mae_avg', 'test_acc', 'test_macro_f1']]
    }).T
    
    comparison_df.columns = ['Model', 'Seq_Len', 'Hidden/D_Model', 'LR', 'Use_Static', 
                            'Val_Loss', 'Test_Loss', 'Test_MAE_Avg', 'Test_Acc', 'Test_F1']
    
    print(comparison_df.to_string())
    
    # 2. 静态特征影响分析
    print("\n" + "="*80)
    print("【2. 静态特征工程的影响分析】\n")
    
    for model_name, data in results.items():
        df = data['all']
        with_static = df[df['use_static'] == True]
        without_static = df[df['use_static'] == False]
        
        if len(with_static) > 0 and len(without_static) > 0:
            best_with = with_static.loc[with_static['val_loss'].idxmin()]
            best_without = without_static.loc[without_static['val_loss'].idxmin()]
            
            print(f"\n{model_name}:")
            print(f"  使用静态特征最佳配置:")
            print(f"    - Val Loss: {best_with['val_loss']:.6f}")
            print(f"    - Test MAE (avg): {best_with['test_mae_avg']:.6f}")
            print(f"    - Test Acc: {best_with['test_acc']:.6f}")
            print(f"  不使用静态特征最佳配置:")
            print(f"    - Val Loss: {best_without['val_loss']:.6f}")
            print(f"    - Test MAE (avg): {best_without['test_mae_avg']:.6f}")
            print(f"    - Test Acc: {best_without['test_acc']:.6f}")
            
            mae_improvement = (best_without['test_mae_avg'] - best_with['test_mae_avg']) / best_without['test_mae_avg'] * 100
            acc_improvement = (best_with['test_acc'] - best_without['test_acc']) / best_without['test_acc'] * 100
            print(f"  静态特征带来的改进:")
            print(f"    - MAE改进: {mae_improvement:.2f}%")
            print(f"    - Acc改进: {acc_improvement:.2f}%")
    
    # 3. 三个模型最终效果对比
    print("\n" + "="*80)
    print("【3. 三个模型最终效果对比（测试集指标）】\n")
    
    metrics_comparison = pd.DataFrame({
        '指标': ['Loss (MSE)', 'MAE (avg_tries)', 'RMSE (avg_tries)', 
                'MAE (succ_rate)', 'RMSE (succ_rate)', 'Accuracy', 'Macro F1'],
        'LSTM': [
            results['LSTM']['best']['test_loss'],
            results['LSTM']['best']['test_mae_avg'],
            results['LSTM']['best']['test_rmse_avg'],
            results['LSTM']['best']['test_mae_succ'],
            results['LSTM']['best']['test_rmse_succ'],
            results['LSTM']['best']['test_acc'],
            results['LSTM']['best']['test_macro_f1']
        ],
        'BiLSTM': [
            results['BiLSTM']['best']['test_loss'],
            results['BiLSTM']['best']['test_mae_avg'],
            results['BiLSTM']['best']['test_rmse_avg'],
            results['BiLSTM']['best']['test_mae_succ'],
            results['BiLSTM']['best']['test_rmse_succ'],
            results['BiLSTM']['best']['test_acc'],
            results['BiLSTM']['best']['test_macro_f1']
        ],
        'Transformer': [
            results['Transformer']['best']['test_loss'],
            results['Transformer']['best']['test_mae_avg'],
            results['Transformer']['best']['test_rmse_avg'],
            results['Transformer']['best']['test_mae_succ'],
            results['Transformer']['best']['test_rmse_succ'],
            results['Transformer']['best']['test_acc'],
            results['Transformer']['best']['test_macro_f1']
        ]
    })
    
    print(metrics_comparison.to_string(index=False))
    
    # 找出每个指标的最佳模型
    print("\n【4. 各指标最佳模型】\n")
    metric_names = ['Loss', 'MAE_avg', 'RMSE_avg', 'MAE_succ', 'RMSE_succ', 'Acc', 'F1']
    metric_cols = ['test_loss', 'test_mae_avg', 'test_rmse_avg', 'test_mae_succ', 
                   'test_rmse_succ', 'test_acc', 'test_macro_f1']
    reverse_better = [True, True, True, True, True, False, False]  # True表示越小越好
    
    for metric_name, metric_col, reverse in zip(metric_names, metric_cols, reverse_better):
        if reverse:
            best_model = min(results.items(), key=lambda x: x[1]['best'][metric_col])
        else:
            best_model = max(results.items(), key=lambda x: x[1]['best'][metric_col])
        print(f"  {metric_name}: {best_model[0]} (值: {best_model[1]['best'][metric_col]:.6f})")
    
    # 保存对比报告到CSV
    comparison_path = os.path.join(RESULT_DIR, "model_comparison.csv")
    metrics_comparison.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"\n对比报告已保存到: {comparison_path}")
    
    # 保存详细对比表
    detailed_comparison_path = os.path.join(RESULT_DIR, "detailed_comparison.csv")
    comparison_df.to_csv(detailed_comparison_path, encoding='utf-8-sig')
    print(f"详细对比表已保存到: {detailed_comparison_path}")


if __name__ == "__main__":
    compare_models()

