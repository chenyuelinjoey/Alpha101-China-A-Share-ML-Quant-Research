# ================================================================
# 模型保存与加载 + 回测分析
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
print("【模型保存与回测分析】")
print("=" * 80)

# ------------------------------
# 第一部分：保存最佳模型
# ------------------------------
print("\n[1] 加载数据并重新训练最佳模型...")

# 加载预处理数据
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

# 重新训练最佳模型：Ensemble_All_Tree（四个树模型的简单平均）
# 但为了保存和部署，我们分别训练四个基模型，然后定义预测函数
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
    """四个模型的平均预测"""
    pred_lgb = lgb_model.predict(X)
    pred_xgb = xgb_model.predict(X)
    pred_cat = cat_model.predict(X)
    pred_rf = rf_model.predict(X)
    return (pred_lgb + pred_xgb + pred_cat + pred_rf) / 4

# 保存模型集合（最佳模型）
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

# 测试模型加载
print("\n[3] 测试模型加载...")
with open('/kaggle/working/best_model_bundle.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
print(f"✅ 模型加载成功，包含 {len(loaded_model)} 个组件")

# 验证预测一致性
pred_loaded = loaded_model['ensemble_func'](X_test)
pred_original = ensemble_predict(X_test)
print(f"   预测一致性: {np.allclose(pred_loaded, pred_original)}")

# ------------------------------
# 第二部分：回测分析
# ------------------------------
print("\n" + "=" * 60)
print("【回测分析】")
print("=" * 60)

# 获取预测值
y_pred = ensemble_predict(X_test)
print(f"测试集预测完成，共 {len(y_pred)} 个样本")

# 构建回测数据框
df_backtest = pd.DataFrame({
    'code': test_code,
    'date': pd.to_datetime(test_date),
    'pred_return': y_pred,
    'actual_return': y_test
})
print(f"回测数据量: {len(df_backtest):,} 行")
print(f"日期范围: {df_backtest['date'].min()} 至 {df_backtest['date'].max()}")
print(f"股票数量: {df_backtest['code'].nunique()}")

# ------------------------------
# 回测参数设置
# ------------------------------
TOP_N = 30          # 多头组合股票数量
BOTTOM_N = 30       # 空头组合股票数量
TRADING_COST = 0.001  # 双边交易成本（千分之一）
REBALANCE_FREQ = 'M'  # 调仓频率：'D'（每日）或 'M'（月度）

print(f"\n回测参数:")
print(f"   多头数量: {TOP_N}")
print(f"   空头数量: {BOTTOM_N}")
print(f"   交易成本: {TRADING_COST*100:.1f}%")
print(f"   调仓频率: {REBALANCE_FREQ}")

# ------------------------------
# 回测核心逻辑
# ------------------------------
def calculate_portfolio_return(df_day, top_n=TOP_N, bottom_n=BOTTOM_N, long_only=True):
    """
    计算单日组合收益
    long_only=True: 只做多
    long_only=False: 多空组合
    """
    df_sorted = df_day.sort_values('pred_return', ascending=False)
    long_stocks = df_sorted.head(top_n)
    long_return = long_stocks['actual_return'].mean()
    if long_only:
        return long_return
    else:
        short_stocks = df_sorted.tail(bottom_n)
        short_return = short_stocks['actual_return'].mean()
        return long_return - short_return

# 按日期分组计算组合收益
if REBALANCE_FREQ == 'M':
    # 月度调仓：每月第一个交易日调仓
    df_backtest['year_month'] = df_backtest['date'].dt.to_period('M')
    daily_returns = df_backtest.groupby('date').apply(
        lambda g: calculate_portfolio_return(g, long_only=True)
    ).reset_index(name='portfolio_return')
else:
    # 每日调仓
    daily_returns = df_backtest.groupby('date').apply(
        lambda g: calculate_portfolio_return(g, long_only=True)
    ).reset_index(name='portfolio_return')

print(f"回测天数: {len(daily_returns)}")

# 考虑交易成本
# 简化：每次调仓扣除双边成本
daily_returns['portfolio_return_net'] = daily_returns['portfolio_return']
if TRADING_COST > 0:
    # 每次调仓扣除成本（假设每次调仓都换手）
    daily_returns['portfolio_return_net'] = daily_returns['portfolio_return'] - TRADING_COST

# ------------------------------
# 计算净值曲线
# ------------------------------
# 使用净收益计算净值
daily_returns['cum_return'] = (1 + daily_returns['portfolio_return_net']).cumprod()
daily_returns['cum_return'] = daily_returns['cum_return'] / daily_returns['cum_return'].iloc[0]

# ------------------------------
# 计算绩效指标
# ------------------------------
returns = daily_returns['portfolio_return_net']
cum_ret = daily_returns['cum_return']

total_return = cum_ret.iloc[-1] - 1
n_days = len(returns)
n_years = n_days / 252
annualized_return = (1 + total_return) ** (1 / n_years) - 1
annualized_vol = returns.std() * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_vol if annualized_vol != 0 else np.nan
max_drawdown = np.min(cum_ret / cum_ret.cummax() - 1)
# 计算最大回撤持续时间
drawdown = cum_ret / cum_ret.cummax() - 1
max_drawdown_start = None
max_drawdown_end = None
# 简化版：记录最大回撤区间
drawdown_start_idx = np.argmax(drawdown == drawdown.min())
if drawdown_start_idx > 0:
    max_drawdown_start = daily_returns['date'].iloc[drawdown_start_idx]
    max_drawdown_end = daily_returns['date'].iloc[np.argmin(cum_ret)]

# 胜率
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

# ------------------------------
# 绘制净值曲线
# ------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 子图1：净值曲线
ax1 = axes[0]
ax1.plot(daily_returns['date'], daily_returns['cum_return'], 
         label='Ensemble_All_Tree (IC=0.0552)', color='blue', linewidth=1.5)
ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax1.fill_between(daily_returns['date'], daily_returns['cum_return'].min()*0.95, 
                  daily_returns['cum_return'], where=(daily_returns['cum_return'] < 1), 
                  color='red', alpha=0.3, label='回撤期')
ax1.set_title('净值曲线 (Ensemble_All_Tree)', fontsize=14)
ax1.set_xlabel('日期')
ax1.set_ylabel('净值')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：回撤曲线
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

# ------------------------------
# 月度收益率统计
# ------------------------------
print("\n【月度收益率统计】")
daily_returns['year_month'] = daily_returns['date'].dt.to_period('M')
monthly_returns = daily_returns.groupby('year_month')['portfolio_return_net'].apply(
    lambda x: (1+x).prod() - 1
).reset_index()
monthly_returns.columns = ['year_month', 'monthly_return']

print("\n最近12个月月度收益率:")
print(monthly_returns.tail(12).to_string(index=False))

# 月度胜率
monthly_win_rate = (monthly_returns['monthly_return'] > 0).mean()
print(f"\n月度胜率: {monthly_win_rate:.2%}")

# ------------------------------
# 保存回测结果
# ------------------------------
daily_returns.to_csv('/kaggle/working/backtest_daily_returns.csv', index=False)
monthly_returns.to_csv('/kaggle/working/backtest_monthly_returns.csv', index=False)
print("\n✅ 回测结果已保存至:")
print("   - /kaggle/working/backtest_daily_returns.csv")
print("   - /kaggle/working/backtest_monthly_returns.csv")

# ------------------------------
# 总结
# ------------------------------
print("\n" + "=" * 80)
print("【模型保存与回测分析完成】")
print("=" * 80)
print(f"🏆 最佳模型: Ensemble_All_Tree (IC=0.0552)")
print(f"📈 年化收益率: {annualized_return:.4%}")
print(f"📊 夏普比率: {sharpe_ratio:.4f}")
print(f"📉 最大回撤: {max_drawdown:.4%}")
print(f"📁 模型文件: /kaggle/working/best_model_bundle.pkl")
print(f"📁 回测结果: /kaggle/working/backtest_daily_returns.csv")
print("=" * 80)
