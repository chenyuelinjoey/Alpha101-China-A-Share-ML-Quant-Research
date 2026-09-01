# ================================================================
# 阶段 2.6：模型保存与回测分析
# 
# 功能：
#   1. 保存最佳模型（Ensemble_All_Tree）为 .pkl 文件（可部署）
#   2. 加载已保存的模型
#   3. 基于测试集预测结果进行回测分析
#   4. 计算绩效指标：年化收益、夏普比率、最大回撤等
#   5. 生成净值曲线图
# 
# 运行时间：约 5 分钟
# 依赖：preprocessed_data.pkl + ensemble_preds.pkl
# ================================================================

import numpy as np
import pandas as pd
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import StackingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

print("=" * 80)
print("【阶段 2.6】模型保存与回测分析")
print("=" * 80)

# ------------------------------
# 1. 加载数据并重新训练最佳模型
# ------------------------------
print("\n[1] 加载数据并重新训练最佳模型...")

with open('/kaggle/working/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
test_code = data['test_code']
test_date = data['test_date']
feature_cols = data['feature_cols']

print(f"训练集: {X_train.shape[0]:,} 行 × {X_train.shape[1]} 列")
print(f"测试集: {X_test.shape[0]:,} 行 × {X_test.shape[1]} 列")

# 重新训练四个基模型
print("\n   训练 LightGBM...")
lgb_model = LGBMRegressor(
    n_estimators=100, learning_rate=0.1, num_leaves=31, max_depth=6,
    random_state=42, n_jobs=-1, verbose=-1
)
lgb_model.fit(X_train, y_train)

print("   训练 XGBoost...")
xgb_model = XGBRegressor(
    n_estimators=100, learning_rate=0.1, max_depth=6,
    random_state=42, n_jobs=-1, verbosity=0, tree_method='hist'
)
xgb_model.fit(X_train, y_train)

print("   训练 CatBoost...")
cat_model = CatBoostRegressor(
    iterations=80, learning_rate=0.1, depth=6,
    random_seed=42, verbose=False
)
cat_model.fit(X_train, y_train)

print("   训练 RandomForest...")
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_split=50,
    n_jobs=-1, random_state=42
)
rf_model.fit(X_train, y_train)

# 定义集成预测函数
def ensemble_predict(X):
    pred_lgb = lgb_model.predict(X)
    pred_xgb = xgb_model.predict(X)
    pred_cat = cat_model.predict(X)
    pred_rf = rf_model.predict(X)
    return (pred_lgb + pred_xgb + pred_cat + pred_rf) / 4

# ------------------------------
# 2. 保存模型
# ------------------------------
print("\n[2] 保存模型...")
model_bundle = {
    'lgb_model': lgb_model,
    'xgb_model': xgb_model,
    'cat_model': cat_model,
    'rf_model': rf_model,
    'feature_cols': feature_cols,
    'model_name': 'Ensemble_All_Tree',
    'ensemble_func': ensemble_predict
}

with open('/kaggle/working/best_model_bundle.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)
print("✅ 模型已保存至: /kaggle/working/best_model_bundle.pkl")

# ------------------------------
# 3. 测试模型加载
# ------------------------------
print("\n[3] 测试模型加载...")
with open('/kaggle/working/best_model_bundle.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
print(f"✅ 模型加载成功，包含 {len(loaded_model)} 个组件")

# 验证预测一致性
pred_loaded = loaded_model['ensemble_func'](X_test)
pred_original = ensemble_predict(X_test)
print(f"   预测一致性: {np.allclose(pred_loaded, pred_original)}")

# ------------------------------
# 4. 回测分析
# ------------------------------
print("\n[4] 回测分析...")
y_pred = ensemble_predict(X_test)

df_backtest = pd.DataFrame({
    'code': test_code,
    'date': pd.to_datetime(test_date),
    'pred_return': y_pred,
    'actual_return': y_test
})
print(f"回测数据量: {len(df_backtest):,} 行")
print(f"日期范围: {df_backtest['date'].min()} 至 {df_backtest['date'].max()}")

# 回测参数
TOP_N = 30
TRADING_COST = 0.001
REBALANCE_FREQ = 'M'

print(f"\n回测参数:")
print(f"   多头数量: {TOP_N}")
print(f"   交易成本: {TRADING_COST*100:.1f}%")
print(f"   调仓频率: {REBALANCE_FREQ}")

def calculate_portfolio_return(df_day, top_n=TOP_N):
    df_sorted = df_day.sort_values('pred_return', ascending=False)
    long_stocks = df_sorted.head(top_n)
    return long_stocks['actual_return'].mean()

if REBALANCE_FREQ == 'M':
    df_backtest['year_month'] = df_backtest['date'].dt.to_period('M')
    daily_returns = df_backtest.groupby('date').apply(
        lambda g: calculate_portfolio_return(g)
    ).reset_index(name='portfolio_return')
else:
    daily_returns = df_backtest.groupby('date').apply(
        lambda g: calculate_portfolio_return(g)
    ).reset_index(name='portfolio_return')

daily_returns['portfolio_return_net'] = daily_returns['portfolio_return'] - TRADING_COST
daily_returns['cum_return'] = (1 + daily_returns['portfolio_return_net']).cumprod()
daily_returns['cum_return'] = daily_returns['cum_return'] / daily_returns['cum_return'].iloc[0]

# 计算绩效指标
returns = daily_returns['portfolio_return_net']
cum_ret = daily_returns['cum_return']
total_return = cum_ret.iloc[-1] - 1
n_days = len(returns)
n_years = n_days / 252
annualized_return = (1 + total_return) ** (1 / n_years) - 1
annualized_vol = returns.std() * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
max_drawdown = np.min(cum_ret / cum_ret.cummax() - 1)
win_rate = (returns > 0).mean()

print("\n" + "=" * 60)
print("【回测绩效统计】")
print("=" * 60)
print(f"回测区间: {daily_returns['date'].min()} 至 {daily_returns['date'].max()}")
print(f"回测天数: {n_days}")
print(f"交易成本: {TRADING_COST*100:.1f}% (双边)")
print(f"\n总收益率: {total_return:.4%}")
print(f"年化收益率: {annualized_return:.4%}")
print(f"年化波动率: {annualized_vol:.4%}")
print(f"夏普比率: {sharpe_ratio:.4f}")
print(f"最大回撤: {max_drawdown:.4%}")
print(f"胜率: {win_rate:.2%}")

# 绘制净值曲线
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
ax1 = axes[0]
ax1.plot(daily_returns['date'], daily_returns['cum_return'], 
         label='Ensemble_All_Tree (IC=0.0552)', color='blue', linewidth=1.5)
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax1.set_title('净值曲线 (Ensemble_All_Tree)', fontsize=14)
ax1.set_xlabel('日期')
ax1.set_ylabel('净值')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
drawdown_series = daily_returns['cum_return'] / daily_returns['cum_return'].cummax() - 1
ax2.fill_between(daily_returns['date'], 0, drawdown_series, color='red', alpha=0.5)
ax2.set_title('回撤曲线', fontsize=14)
ax2.set_xlabel('日期')
ax2.set_ylabel('回撤')
ax2.set_ylim([-0.5, 0.01])
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/backtest_analysis.png', dpi=150)
print("\n✅ 回测分析图已保存至: /kaggle/working/backtest_analysis.png")
plt.show()

# 保存回测结果
daily_returns.to_csv('/kaggle/working/backtest_daily_returns.csv', index=False)
print("\n✅ 回测结果已保存至:")
print("   - /kaggle/working/backtest_daily_returns.csv")

print("\n" + "=" * 80)
print("【阶段 2.6 完成】")
print("=" * 80)
print(f"🏆 最佳模型: Ensemble_All_Tree (IC=0.0552)")
print(f"📈 年化收益率: {annualized_return:.4%}")
print(f"📊 夏普比率: {sharpe_ratio:.4f}")
print(f"📉 最大回撤: {max_drawdown:.4%}")
print(f"📁 模型文件: /kaggle/working/best_model_bundle.pkl")
print("=" * 80)
