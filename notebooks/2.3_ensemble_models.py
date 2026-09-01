# ================================================================
# 阶段 2.3：集成模型（Ensemble Models）
# 
# 功能：将多个基模型的预测结果进行组合，通常能获得比单一模型
#       更稳定、更准确的预测。本阶段实现两种集成策略：
#   1. 平均集成（Averaging）—— 简单高效，降低方差
#   2. Stacking 集成 —— 用元模型学习最优组合方式
# 
# 输入依赖：
#   - preprocessed_data.pkl : 原始数据（用于重新训练 Stacking 基模型）
#   - tree_preds.pkl        : 阶段 2.2 的预测结果（用于平均集成）
# 
# 输出：
#   - ensemble_preds.pkl   : 集成模型的预测值（字典）
#   - ensemble_results.pkl : 集成模型的评估指标（字典）
# 
# 运行时间：约 5-10 分钟
# ================================================================

import numpy as np
import pandas as pd
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

print("=" * 80)
print("【阶段 2.3】集成模型（Ensemble）")
print("=" * 80)

# ------------------------------
# 1. 加载数据
# ------------------------------
print("\n[1] 加载数据...")

# 1.1 加载预处理数据（用于 Stacking 重新训练）
with open('/kaggle/working/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
print(f"训练集: {X_train.shape[0]:,} 行 × {X_train.shape[1]} 列")
print(f"测试集: {X_test.shape[0]:,} 行 × {X_test.shape[1]} 列")

# 1.2 加载阶段 2.2 的预测结果（用于平均集成）
with open('/kaggle/working/tree_preds.pkl', 'rb') as f:
    tree_preds = pickle.load(f)
print(f"加载了 {len(tree_preds)} 个基模型的预测结果")

# ------------------------------
# 2. 评估函数（与前面保持一致）
# ------------------------------
def evaluate(name, y_pred):
    """计算并打印模型的三个核心指标：IC、MSE、MAE"""
    if len(y_pred) != len(y_test):
        min_len = min(len(y_pred), len(y_test))
        y_pred = y_pred[:min_len]
        y_true = y_test[:min_len]
    else:
        y_true = y_test
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    ic = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"{name:25s} | IC: {ic:7.4f} | MSE: {mse:.6f} | MAE: {mae:.6f}")
    return {'IC': ic, 'MSE': mse, 'MAE': mae}

results = {}
preds = {}

# ------------------------------
# 3. 平均集成（Averaging Ensemble）
# ------------------------------
print("\n[2] 平均集成...")

# 3.1 LightGBM + XGBoost 平均
# 理由：两者都是 Boosting 树，但算法细节不同，预测分布有差异
#       平均后可以抵消各自的偏置，提升稳定性
pred_lgb_xgb = (tree_preds['LightGBM'] + tree_preds['XGBoost']) / 2
results['Ensemble_LGB_XGB'] = evaluate('Ensemble_LGB_XGB', pred_lgb_xgb)
preds['Ensemble_LGB_XGB'] = pred_lgb_xgb

# 3.2 全部 4 个树模型平均（LightGBM + XGBoost + CatBoost + RandomForest）
# 理由：增加模型多样性，进一步降低方差
tree_names = ['LightGBM', 'XGBoost', 'CatBoost', 'RandomForest']
all_tree_preds = np.array([tree_preds[name] for name in tree_names])
pred_all_avg = np.mean(all_tree_preds, axis=0)
results['Ensemble_All_Tree'] = evaluate('Ensemble_All_Tree', pred_all_avg)
preds['Ensemble_All_Tree'] = pred_all_avg

# ------------------------------
# 4. Stacking 集成（堆叠泛化）
# ------------------------------
print("\n[3] Stacking 集成...")
# Stacking 原理：
#   1. 第一层（基模型）：多个异构模型对原始数据进行预测
#   2. 第二层（元模型）：以基模型的预测结果为输入，学习最优组合方式
# 优势：比简单平均更灵活，能捕捉基模型预测之间的复杂关系
# 
# 基模型选择：LightGBM + XGBoost + CatBoost（三者各有优势，互补性强）
# 元模型选择：Ridge（简单线性模型，防止过拟合）

base_models = [
    ('lgb', LGBMRegressor(
        n_estimators=80, 
        learning_rate=0.1, 
        num_leaves=31, 
        random_state=42, 
        verbose=-1
    )),
    ('xgb', XGBRegressor(
        n_estimators=80, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42, 
        verbosity=0
    )),
    ('cat', CatBoostRegressor(
        iterations=60, 
        learning_rate=0.1, 
        depth=6, 
        random_seed=42, 
        verbose=False
    ))
]

# cv=5 表示使用 5 折交叉验证生成基模型的预测（防止信息泄露）
stacking = StackingRegressor(
    estimators=base_models, 
    final_estimator=Ridge(alpha=1.0), 
    cv=5
)

print("   训练 Stacking（基模型: LGB+XGB+Cat, 元模型: Ridge）...")
stacking.fit(X_train, y_train)
pred_stacking = stacking.predict(X_test)
results['Stacking'] = evaluate('Stacking', pred_stacking)
preds['Stacking'] = pred_stacking

# 释放内存
del stacking
gc.collect()
print("   Stacking 完成，内存已释放。")

# ------------------------------
# 5. 保存阶段结果
# ------------------------------
print("\n[4] 保存阶段结果...")
with open('/kaggle/working/ensemble_preds.pkl', 'wb') as f:
    pickle.dump(preds, f)
with open('/kaggle/working/ensemble_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("✅ 集成模型结果已保存至:")
print("   - /kaggle/working/ensemble_preds.pkl")
print("   - /kaggle/working/ensemble_results.pkl")

print("\n" + "=" * 80)
print("阶段 2.3 完成！")
print("=" * 80)
