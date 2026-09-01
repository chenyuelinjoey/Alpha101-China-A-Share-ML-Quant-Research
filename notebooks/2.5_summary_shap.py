# ================================================================
# 阶段 2.5：结果汇总与 SHAP 可解释性
# 
# 功能：
#   1. 加载所有阶段（2.1~2.4）的预测和评估结果
#   2. 排除统计基准模型（Mean_Baseline, Lag1_Baseline, AR(1)）
#      因为这些模型只是捕捉了收益率自相关，不是有效信号
#   3. 在有效模型中识别最佳模型（按 IC 排序）
#   4. 对最佳模型（或代表性模型）进行 SHAP 全局解释
#   5. 生成最终提交文件（使用最佳有效模型）
#   6. 输出完整的模型排名（所有模型）和有效模型排名
# 
# 注意：
#   - 统计基准模型仅用于建立预测下限，不参与最终评选
#   - 有效模型包括：线性模型、树模型、集成模型、深度学习模型
# 
# 运行时间：约 5-10 分钟（SHAP 计算稍慢）
# 依赖：preprocessed_data.pkl + 所有 *preds.pkl/*results.pkl
# ================================================================

import numpy as np
import pandas as pd
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import shap

print("=" * 80)
print("【阶段 2.5】结果汇总与 SHAP 可解释性（修正版）")
print("=" * 80)

# ------------------------------
# 1. 加载所有阶段的结果
# ------------------------------
print("\n[1] 加载各阶段预测和结果...")

# 定义需要加载的文件
stages = [
    ('benchmark', '基准模型'),
    ('tree', '核心树模型'),
    ('ensemble', '集成模型'),
    ('deep', '深度学习与Autoencoder')
]

preds_all = {}
results_all = {}

for prefix, name in stages:
    pred_file = f'/kaggle/working/{prefix}_preds.pkl'
    res_file = f'/kaggle/working/{prefix}_results.pkl'
    try:
        with open(pred_file, 'rb') as f:
            preds = pickle.load(f)
            preds_all.update(preds)
        with open(res_file, 'rb') as f:
            res = pickle.load(f)
            results_all.update(res)
        print(f"   ✅ 加载 {name} ({len(preds)} 个模型)")
    except FileNotFoundError:
        print(f"   ⚠️ 跳过 {name}（文件不存在）")

print(f"\n总计加载 {len(preds_all)} 个预测，{len(results_all)} 个结果")

# ------------------------------
# 2. 定义排除列表（统计基准模型，不参与最终评选）
# ------------------------------
EXCLUDE_MODELS = ['Mean_Baseline', 'Lag1_Baseline', 'AR(1)']
print(f"\n[2] 排除统计基准模型: {EXCLUDE_MODELS}")
print("   这些模型仅用于建立预测下限，不代表有效信号。")

# ------------------------------
# 3. 生成完整排名表（含所有模型）
# ------------------------------
print("\n[3] 生成完整模型排名表（含所有模型）...")
result_df_all = pd.DataFrame(results_all).T
result_df_all = result_df_all[result_df_all['IC'].notna()]
result_df_all = result_df_all.sort_values('IC', ascending=False)

print("\n完整排名（含基准模型）:")
print(result_df_all.head(10).to_string())

# 保存完整排名
result_df_all.to_csv('/kaggle/working/all_models_ranking_full.csv', index=True)

# ------------------------------
# 4. 生成有效模型排名表（排除基准模型）
# ------------------------------
print("\n[4] 生成有效模型排名表（排除统计基准）...")
result_df_valid = result_df_all[~result_df_all.index.isin(EXCLUDE_MODELS)]
result_df_valid = result_df_valid.sort_values('IC', ascending=False)

print("\n🏆 有效模型排名（Top 10）:")
print(result_df_valid.head(10).to_string())

# 保存有效排名
result_df_valid.to_csv('/kaggle/working/all_models_ranking_valid.csv', index=True)

# ------------------------------
# 5. 识别最佳有效模型
# ------------------------------
if len(result_df_valid) == 0:
    print("❌ 无有效模型结果，请确保至少运行了阶段 2.2。")
    raise SystemExit("没有可用的有效模型结果。")

best_model_name = result_df_valid.index[0]
best_ic = result_df_valid.loc[best_model_name, 'IC']
best_mse = result_df_valid.loc[best_model_name, 'MSE']
best_mae = result_df_valid.loc[best_model_name, 'MAE']

print(f"\n🏆 最佳有效模型: {best_model_name}")
print(f"   IC  : {best_ic:.4f}")
print(f"   MSE : {best_mse:.6f}")
print(f"   MAE : {best_mae:.6f}")

# ------------------------------
# 6. SHAP 可解释性分析（对 LightGBM）
# ------------------------------
print("\n[5] SHAP 可解释性分析（基于 LightGBM）...")
# 加载原始数据用于重新训练
with open('/kaggle/working/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
feature_cols = data['feature_cols']

print("   重新训练 LightGBM 用于 SHAP...")
from lightgbm import LGBMRegressor
model_lgb = LGBMRegressor(
    n_estimators=100, 
    learning_rate=0.1, 
    num_leaves=31, 
    max_depth=6,
    random_state=42, 
    n_jobs=-1, 
    verbose=-1
)
model_lgb.fit(X_train, y_train)

# 采样测试集（500 个样本）以加速 SHAP 计算
print("   计算 SHAP 值（采样 500 个样本）...")
X_test_sample = X_test[:500]
explainer = shap.TreeExplainer(model_lgb)
shap_values = explainer.shap_values(X_test_sample)

# 绘制 SHAP 汇总图（特征重要性 + 影响方向）
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols, show=False)
plt.title("SHAP Summary Plot (LightGBM)", fontsize=14)
plt.tight_layout()
plt.savefig('/kaggle/working/shap_summary.png', dpi=150)
print("   ✅ SHAP 汇总图已保存: /kaggle/working/shap_summary.png")

# 绘制 SHAP 条形图（平均绝对 SHAP 值）
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (LightGBM)", fontsize=14)
plt.tight_layout()
plt.savefig('/kaggle/working/shap_bar.png', dpi=150)
print("   ✅ SHAP 条形图已保存: /kaggle/working/shap_bar.png")

# 保存 SHAP 值（可选）
np.save('/kaggle/working/shap_values.npy', shap_values)
print("   ✅ SHAP 值已保存至 /kaggle/working/shap_values.npy")

# 释放内存
del model_lgb, explainer
gc.collect()

# ------------------------------
# 7. 生成最终提交文件（使用最佳有效模型）
# ------------------------------
print("\n[6] 生成最终提交文件...")

# 获取最佳模型的预测值（如果最佳模型在 preds_all 中）
if best_model_name in preds_all:
    final_pred = preds_all[best_model_name]
else:
    # 否则使用 LightGBM 的预测（后备）
    print(f"   ⚠️ 最佳模型 {best_model_name} 的预测未找到，使用 LightGBM 作为后备。")
    if 'LightGBM' in preds_all:
        final_pred = preds_all['LightGBM']
    else:
        # 如果连 LightGBM 都没有，报错
        raise KeyError("没有可用的预测值，请检查阶段 2.2 是否成功运行。")

# 加载测试集日期和代码（从预处理数据）
with open('/kaggle/working/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
test_code = data['test_code']
test_date = data['test_date']

submission = pd.DataFrame({
    'Stkcd': test_code,
    'Trddt': test_date,
    'Predicted_Return': final_pred
})
submission.to_csv('/kaggle/working/submission_final.csv', index=False)
print("📁 提交文件已保存至 /kaggle/working/submission_final.csv")

# ------------------------------
# 8. 输出总结
# ------------------------------
print("\n" + "=" * 80)
print("【阶段 2.5 完成】")
print("=" * 80)
print(f"🏆 最佳有效模型: {best_model_name} (IC={best_ic:.4f})")
print(f"   （统计基准模型 {EXCLUDE_MODELS} 已被排除）")
print(f"📈 SHAP 图: /kaggle/working/shap_summary.png, /kaggle/working/shap_bar.png")
print(f"📊 完整排名表: /kaggle/working/all_models_ranking_full.csv")
print(f"📊 有效排名表: /kaggle/working/all_models_ranking_valid.csv")
print(f"📁 提交文件: /kaggle/working/submission_final.csv")
print("\n" + "=" * 80)
print("所有阶段（2.1 ~ 2.5）执行完毕！")
print("=" * 80)
