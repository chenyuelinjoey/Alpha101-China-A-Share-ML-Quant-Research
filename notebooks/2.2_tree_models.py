# ================================================================
# 阶段 2.2：核心树模型与线性模型
# 
# 功能：建立“主力”预测模型，包括：
#   - 线性模型（Ridge、Lasso、ElasticNet）—— 提供线性基准
#   - 树模型（RandomForest、LightGBM、XGBoost、CatBoost）—— 捕捉非线性
# 
# 输出：
#   - tree_preds.pkl   : 所有核心模型的预测值（字典）
#   - tree_results.pkl : 所有核心模型的评估指标（字典）
# 
# 运行时间：约 10-15 分钟
# 依赖：preprocessed_data.pkl
# ================================================================

import numpy as np
import pandas as pd
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

print("=" * 80)
print("【阶段 2.2】核心树模型与线性模型")
print("=" * 80)

# ------------------------------
# 1. 加载预处理数据
# ------------------------------
print("\n[1] 加载预处理数据...")
with open('/kaggle/working/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
X_train = data['X_train']          # 训练特征 (n_samples, 102)
y_train = data['y_train']          # 训练标签 (n_samples,)
X_test = data['X_test']            # 测试特征 (n_samples, 102)
y_test = data['y_test']            # 测试标签 (n_samples,)
feature_cols = data['feature_cols']# 特征名称列表
print(f"训练集: {X_train.shape[0]:,} 行 × {X_train.shape[1]} 列")
print(f"测试集: {X_test.shape[0]:,} 行 × {X_test.shape[1]} 列")

# ------------------------------
# 2. 评估函数（与阶段2.1保持一致）
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

# 存储所有预测和结果
results = {}
preds = {}

# ------------------------------
# 3. 定义模型字典（按类别分组）
# ------------------------------
# 3.1 线性模型（带正则化）
linear_models = {
    'Ridge': Ridge(alpha=1.0),                      # L2正则化，稳定且不易过拟合
    'Lasso': Lasso(alpha=0.0005, max_iter=10000),   # L1正则化，可进行特征选择
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)  # L1+L2混合
}

# 3.2 树模型（集成学习）
tree_models = {
    'RandomForest': RandomForestRegressor(
        n_estimators=100,        # 树的数量
        max_depth=10,            # 限制深度防止过拟合
        min_samples_split=50,    # 分裂所需最小样本数（减少过拟合）
        n_jobs=-1,               # 使用所有CPU核心
        random_state=42
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=100,        # 迭代次数
        learning_rate=0.1,       # 学习率
        num_leaves=31,           # 叶子节点数（控制模型复杂度）
        max_depth=6,             # 最大深度
        random_state=42,
        n_jobs=-1,
        verbose=-1               # 关闭训练日志
    ),
    'XGBoost': XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        verbosity=0,             # 关闭日志
        tree_method='hist'       # 使用直方图近似加速且节省内存
    ),
    'CatBoost': CatBoostRegressor(
        iterations=80,           # 迭代次数
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False            # 关闭日志
    )
}

# 合并所有模型（顺序：线性优先，树模型次之）
models = {**linear_models, **tree_models}

print("\n[2] 开始训练核心模型（共 %d 个）..." % len(models))

# ------------------------------
# 4. 逐个训练并评估
# ------------------------------
for name, model in models.items():
    print(f"\n>>> 训练 {name} ...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = evaluate(name, y_pred)
    preds[name] = y_pred
    
    # 释放模型内存（重要！）
    del model
    gc.collect()
    print(f"   {name} 完成，内存已释放。")

# ------------------------------
# 5. 保存阶段结果
# ------------------------------
print("\n[3] 保存阶段结果...")
with open('/kaggle/working/tree_preds.pkl', 'wb') as f:
    pickle.dump(preds, f)
with open('/kaggle/working/tree_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("✅ 核心模型结果已保存至:")
print("   - /kaggle/working/tree_preds.pkl")
print("   - /kaggle/working/tree_results.pkl")

print("\n" + "=" * 80)
print("阶段 2.2 完成！")
print("=" * 80)
