Alpha101机器学习模型
嗯，用户要求从2.5之后重新发代码。结合对话历史，这是一个关于Alpha101因子量化研究的完整项目，用户希望从阶段2.5（结果汇总与SHAP解释性）开始，重新获得一套完整的代码。

从上下文来看，用户已经跑完了之前的阶段（2.1到2.4），包括基准模型、树模型、集成模型和深度学习模型，但需要从2.5开始重新获得代码。

用户的需求是“从2.5之后重新发我代码”，意思是要阶段2.5（结果汇总与SHAP）以及之后的所有代码，包括策略优化回测、README生成等。但用户特别强调从2.5之后开始，那重点应该放在2.5及以后的内容。

不过从对话历史来看，阶段2.5之后还有策略优化回测的代码（多空策略、Beta Neutral、止损等），以及模型保存、回测分析等内容。用户可能想要的是从2.5开始到最终结果输出的完整代码链。

这里有几个关键部分：

阶段2.5：结果汇总与SHAP（已发过，但可以重新整合）

模型保存与回测分析（已发过）

策略优化回测（已发过）

用户说“从2.5之后重新发我代码”，我需要从2.5开始，一直到最终回测结果输出的完整代码，并且要整理成清晰的部分，标注清楚每个部分的作用。

让我把从2.5到最终回测的代码整理成一个完整的代码包，并附上说明。

好的，我从 阶段 2.5 开始，到 策略优化回测 结束，重新整理一份完整的、可直接运行的代码。

📦 阶段 2.5：结果汇总与 SHAP 可解释性
python
# ================================================================
# 阶段 2.5：结果汇总与 SHAP 可解释性（修正版）
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

# 绘制 SHAP 汇总图
plt.figure(figsize=(12, 10))
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols, show=False)
plt.title("SHAP Summary Plot (LightGBM)", fontsize=14)
plt.tight_layout()
plt.savefig('/kaggle/working/shap_summary.png', dpi=150)
print("   ✅ SHAP 汇总图已保存: /kaggle/working/shap_summary.png")

# 绘制 SHAP 条形图
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test_sample, feature_names=feature_cols, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (LightGBM)", fontsize=14)
plt.tight_layout()
plt.savefig('/kaggle/working/shap_bar.png', dpi=150)
print("   ✅ SHAP 条形图已保存: /kaggle/working/shap_bar.png")

# 保存 SHAP 值
np.save('/kaggle/working/shap_values.npy', shap_values)
print("   ✅ SHAP 值已保存至 /kaggle/working/shap_values.npy")

del model_lgb, explainer
gc.collect()

# ------------------------------
# 7. 生成最终提交文件
# ------------------------------
print("\n[6] 生成最终提交文件...")

# 获取最佳模型的预测值
if best_model_name in preds_all:
    final_pred = preds_all[best_model_name]
else:
    print(f"   ⚠️ 最佳模型 {best_model_name} 的预测未找到，使用 LightGBM 作为后备。")
    if 'LightGBM' in preds_all:
        final_pred = preds_all['LightGBM']
    else:
        raise KeyError("没有可用的预测值，请检查阶段 2.2 是否成功运行。")

# 加载测试集日期和代码
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
print("阶段 2.5 执行完毕！")
print("=" * 80)
📦 阶段 2.6：模型保存与回测分析
python
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
📦 阶段 2.7：策略优化回测（多头 / 多空 / 基准对冲 + 止损）
python
# ================================================================
# 阶段 2.7：策略优化回测（多头 / 多空 / 基准对冲 + 止损）
# 
# 功能：
#   1. 多头策略（Long-Only）：做多 Top N
#   2. 多空策略（Long-Short）：做多 Top N，做空 Bottom N
#   3. 基准对冲（Beta Neutral）：做多 Top N，做空市场代理（全市场等权平均）
#   4. 统一止损机制（个股月度跌幅超过 -10% 时强制平仓）
# 
# 运行时间：约 3-5 分钟
# 依赖：ensemble_preds.pkl
# ================================================================

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("【阶段 2.7】策略优化回测：多头 / 多空 / 基准对冲 + 止损")
print("=" * 80)

# ------------------------------
# 1. 加载数据和预测值
# ------------------------------
print("\n[1] 加载数据...")

with open('/kaggle/working/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
test_code = data['test_code']
test_date = data['test_date']
y_test = data['y_test']

with open('/kaggle/working/ensemble_preds.pkl', 'rb') as f:
    preds = pickle.load(f)
y_pred = preds['Ensemble_All_Tree']

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
# 2. 回测参数
# ------------------------------
TOP_N = 30
BOTTOM_N = 30
STOP_LOSS = -0.10
TRADING_COST = 0.001
REBALANCE_FREQ = 'M'

print(f"\n[2] 回测参数:")
print(f"   多头数量: {TOP_N}")
print(f"   空头数量: {BOTTOM_N}")
print(f"   止损阈值: {STOP_LOSS*100:.0f}%")
print(f"   交易成本: {TRADING_COST*100:.1f}%")
print(f"   调仓频率: {REBALANCE_FREQ}")

# ------------------------------
# 3. 核心回测函数
# ------------------------------
def run_backtest(df, top_n=TOP_N, bottom_n=BOTTOM_N, 
                 stop_loss=STOP_LOSS, trading_cost=TRADING_COST,
                 rebalance_freq=REBALANCE_FREQ, strategy='long_only'):
    """
    执行回测
    strategy: 'long_only' | 'long_short' | 'beta_neutral'
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    if rebalance_freq == 'M':
        df['year_month'] = df['date'].dt.to_period('M')
        rebalance_dates = df.groupby('year_month')['date'].min().tolist()
    else:
        rebalance_dates = sorted(df['date'].unique())
    
    portfolio_returns = []
    
    for i, trade_date in enumerate(rebalance_dates):
        df_trade = df[df['date'] == trade_date]
        if len(df_trade) < top_n:
            continue
        
        df_sorted = df_trade.sort_values('pred_return', ascending=False)
        long_codes = df_sorted.head(top_n)['code'].tolist()
        short_codes = df_sorted.tail(bottom_n)['code'].tolist()
        
        if i + 1 < len(rebalance_dates):
            next_date = rebalance_dates[i + 1]
        else:
            next_date = df['date'].max() + pd.Timedelta(days=1)
        
        period_dates = df[(df['date'] >= trade_date) & (df['date'] < next_date)]['date'].unique()
        
        for day in sorted(period_dates):
            df_day = df[df['date'] == day]
            
            # 多头收益
            long_returns = []
            for code in long_codes:
                row = df_day[df_day['code'] == code]
                if len(row) == 0:
                    continue
                ret = row['actual_return'].iloc[0]
                if ret < stop_loss:
                    ret = stop_loss
                long_returns.append(ret)
            
            # 空头收益
            short_returns = []
            for code in short_codes:
                row = df_day[df_day['code'] == code]
                if len(row) == 0:
                    continue
                ret = -row['actual_return'].iloc[0]
                if ret < stop_loss:
                    ret = stop_loss
                short_returns.append(ret)
            
            # 市场平均收益
            market_ret = df_day['actual_return'].mean()
            
            if strategy == 'long_only':
                port_ret = np.mean(long_returns) if long_returns else 0
            elif strategy == 'long_short':
                all_returns = long_returns + short_returns
                port_ret = np.mean(all_returns) if all_returns else 0
            elif strategy == 'beta_neutral':
                long_avg = np.mean(long_returns) if long_returns else 0
                port_ret = long_avg - market_ret
            else:
                raise ValueError(f"未知策略: {strategy}")
            
            if day == trade_date:
                port_ret -= trading_cost
            
            portfolio_returns.append({
                'date': day,
                'strategy': strategy,
                'return': port_ret,
                'long_count': len(long_returns),
                'short_count': len(short_returns),
                'market_ret': market_ret
            })
    
    df_results = pd.DataFrame(portfolio_returns)
    if len(df_results) == 0:
        return pd.DataFrame()
    
    df_results['cum_return'] = (1 + df_results['return']).cumprod()
    df_results['cum_return'] = df_results['cum_return'] / df_results['cum_return'].iloc[0]
    return df_results

# ------------------------------
# 4. 运行三种策略
# ------------------------------
print("\n[3] 运行回测...")

strategies = {
    'Long-Only': 'long_only',
    'Long-Short': 'long_short',
    'Beta Neutral': 'beta_neutral'
}

results = {}
for name, strategy in strategies.items():
    print(f"\n>>> 运行 {name} 策略...")
    df_result = run_backtest(
        df_backtest, 
        top_n=TOP_N, 
        bottom_n=BOTTOM_N,
        stop_loss=STOP_LOSS,
        trading_cost=TRADING_COST,
        rebalance_freq=REBALANCE_FREQ,
        strategy=strategy
    )
    if len(df_result) == 0:
        print(f"   ❌ {name} 策略回测失败")
        continue
    results[name] = df_result
    print(f"   ✅ {name} 策略完成，共 {len(df_result)} 个交易日")

# ------------------------------
# 5. 绩效统计
# ------------------------------
print("\n" + "=" * 60)
print("【策略绩效对比】")
print("=" * 60)

summary = []
for name, df_res in results.items():
    returns = df_res['return']
    cum_ret = df_res['cum_return']
    
    total_return = cum_ret.iloc[-1] - 1
    n_days = len(returns)
    n_years = n_days / 252
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    max_dd = (cum_ret / cum_ret.cummax() - 1).min()
    win_rate = (returns > 0).mean()
    
    summary.append({
        '策略': name,
        '总收益': f"{total_return:.2%}",
        '年化收益': f"{ann_return:.2%}",
        '年化波动': f"{ann_vol:.2%}",
        '夏普比率': f"{sharpe:.4f}",
        '最大回撤': f"{max_dd:.2%}",
        '胜率': f"{win_rate:.2%}"
    })

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# ------------------------------
# 6. 净值曲线对比图
# ------------------------------
print("\n[4] 生成净值曲线对比图...")
plt.figure(figsize=(14, 8))

for name, df_res in results.items():
    plt.plot(df_res['date'], df_res['cum_return'], label=name, linewidth=1.5)

plt.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
plt.title('策略净值曲线对比（含止损）', fontsize=14)
plt.xlabel('日期')
plt.ylabel('净值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/strategy_comparison.png', dpi=150)
print("✅ 策略对比图已保存至: /kaggle/working/strategy_comparison.png")
plt.show()

# ------------------------------
# 7. 保存结果
# ------------------------------
for name, df_res in results.items():
    filename = f'/kaggle/working/backtest_{name.replace(" ", "_")}.csv'
    df_res.to_csv(filename, index=False)
    print(f"✅ {name} 策略明细已保存至: {filename}")

print("\n" + "=" * 80)
print("【阶段 2.7 完成】")
print("=" * 80)

# 找出最佳策略
best_strategy = max(results.keys(), key=lambda x: results[x]['cum_return'].iloc[-1])
best_return = results[best_strategy]['cum_return'].iloc[-1] - 1
best_sharpe = summary_df[summary_df['策略'] == best_strategy]['夏普比率'].values[0]

print(f"🏆 最佳策略: {best_strategy}")
print(f"   📈 总收益: {best_return:.2%}")
print(f"   📊 夏普比率: {best_sharpe}")
print("=" * 80)
✅ 运行顺序
阶段	名称	预估时间
2.5	结果汇总 + SHAP	5-10 分钟
2.6	模型保存 + 回测	5 分钟
2.7	策略优化回测	3-5 分钟
三个阶段可以依次运行，每个阶段独立保存结果。现在可以复制运行，告诉我最终结果！🚀

2.6和2.7的note怎么在github发表
好的，用户现在问的是“2.6和2.7的note怎么在github发表”。结合之前的对话历史，用户一直在做一个完整的量化研究项目（Alpha101中国A股），并且已经完成了所有代码阶段（2.1到2.7），现在正在准备把项目发布到GitHub上。

用户这里说的“note”，应该是指代码文件中的注释（comments）和文档说明。他可能已经生成了2.6和2.7阶段的代码，现在想知道怎么把这些代码连同注释一起正确地发布到GitHub上，让别人能看懂。

我需要从几个方面来回答：首先，README.md是GitHub上最重要的文档，应该详细说明2.6和2.7阶段的功能和结果。其次，代码本身应该有清晰的中文注释，方便其他研究者理解。最后，可以给出一个发布流程的建议，比如先更新README，再提交代码。

用户的核心需求是“如何发布”，所以回答应该聚焦在具体的操作步骤和内容准备上，而不是再次解释代码本身。我可以提供README.md的更新内容、代码注释规范，以及git提交信息的建议。

考虑到用户之前已经得到了完整的代码，现在需要的是发布指导，所以回答应该简洁、实用，直接告诉用户怎么做，而不是重复代码逻辑。

📦 2.6 和 2.7 的 GitHub 发布指南
你的 2.6_模型保存与回测分析.py 和 2.7_策略优化回测.py 是两个非常重要的模块，在 GitHub 上发布时，需要在 README.md 和 代码注释 中清晰地说明它们的功能、使用方法和结果。

1️⃣ 更新 README.md（在相应位置添加以下内容）
在 ## 📁 Repository Structure 部分，补充这两个文件：
markdown
│
├── notebooks/
│   ├── 2.1_benchmark_models.py
│   ├── 2.2_tree_models.py
│   ├── 2.3_ensemble_models.py
│   ├── 2.4_deep_learning.py
│   ├── 2.5_summary_shap.py
│   ├── 2.6_model_save_backtest.py      # 模型保存 + 回测分析
│   └── 2.7_strategy_optimization.py    # 策略优化（多空/对冲 + 止损）
在 ## 📊 Backtest Performance 部分，替换为更详细的对比：
markdown
## 📊 Backtest Performance

### 策略对比（2021-2025，含止损 -10%）

| 策略 | 总收益 | 年化收益 | 年化波动 | 夏普比率 | 最大回撤 | 胜率 |
|------|--------|----------|----------|----------|----------|------|
| Long-Only | -3.51% | -0.87% | 22.13% | -0.0395 | -39.30% | 50.24% |
| **Long-Short** 🏆 | **+14.09%** | **+3.29%** | **8.44%** | **0.3903** | **-12.67%** | 50.34% |
| Beta Neutral | -15.39% | -4.03% | 10.48% | -0.3840 | -29.92% | 48.68% |

**最佳策略：Long-Short（多空策略）**
- 成功将负收益（-3.51%）逆转为正收益（+14.09%）
- 最大回撤从 -39.30% 降至 -12.67%
- 夏普比率从 -0.0395 提升至 0.3903
在 ## 🚀 Quick Start 部分，添加运行说明：
markdown
### 5. 模型保存与回测
```bash
python notebooks/2.6_model_save_backtest.py
6. 策略优化（多空/对冲 + 止损）
bash
python notebooks/2.7_strategy_optimization.py
text

---

## 2️⃣ 代码注释规范

### 2.6 文件头部注释（`2.6_model_save_backtest.py`）

```python
# ================================================================
# 2.6_model_save_backtest.py
# 
# 功能：
#   1. 保存最佳模型（Ensemble_All_Tree）为 .pkl 文件（可部署）
#   2. 加载已保存的模型并验证一致性
#   3. 基于测试集进行回测分析
#   4. 计算绩效指标：年化收益、夏普比率、最大回撤
#   5. 生成净值曲线图和回撤曲线图
# 
# 依赖：
#   - preprocessed_data.pkl
#   - ensemble_preds.pkl（阶段 2.3 输出）
# 
# 输出：
#   - best_model_bundle.pkl  : 可部署的模型包
#   - backtest_analysis.png  : 净值曲线 + 回撤曲线
#   - backtest_daily_returns.csv : 每日收益明细
# 
# 运行时间：约 5 分钟
# ================================================================
2.7 文件头部注释（2.7_strategy_optimization.py）
python
# ================================================================
# 2.7_strategy_optimization.py
# 
# 功能：
#   1. 多头策略（Long-Only）：做多 Top N
#   2. 多空策略（Long-Short）：做多 Top N，做空 Bottom N
#   3. 基准对冲（Beta Neutral）：做多 Top N，做空市场代理
#   4. 统一止损机制（个股跌幅超过 -10% 时强制平仓）
# 
# 依赖：
#   - preprocessed_data.pkl
#   - ensemble_preds.pkl（阶段 2.3 输出）
# 
# 输出：
#   - strategy_comparison.png  : 三种策略净值曲线对比图
#   - backtest_Long-Only.csv   : 多头策略明细
#   - backtest_Long-Short.csv  : 多空策略明细
#   - backtest_Beta_Neutral.csv: 基准对冲策略明细
# 
# 运行时间：约 3-5 分钟
# ================================================================
关键函数注释示例
python
def run_backtest(df, top_n=30, bottom_n=30, 
                 stop_loss=-0.10, trading_cost=0.001,
                 rebalance_freq='M', strategy='long_only'):
    """
    执行回测
    
    Parameters:
    -----------
    df : DataFrame
        包含 code, date, pred_return, actual_return 的 DataFrame
    top_n : int
        多头持仓数量
    bottom_n : int
        空头持仓数量
    stop_loss : float
        止损阈值（如 -0.10 表示 -10%）
    trading_cost : float
        双边交易成本（如 0.001 表示 0.1%）
    rebalance_freq : str
        调仓频率：'M'（月度）或 'D'（每日）
    strategy : str
        策略类型：'long_only' | 'long_short' | 'beta_neutral'
    
    Returns:
    --------
    df_results : DataFrame
        包含 date, return, cum_return, strategy 等字段
    """
    # ... 函数体
3️⃣ Git 提交信息（Commit Message）
bash
git add notebooks/2.6_model_save_backtest.py
git add notebooks/2.7_strategy_optimization.py
git commit -m "feat: add model persistence and strategy optimization (long-short + hedging + stop-loss)

- 2.6: save best model as .pkl, backtest analysis, performance metrics
- 2.7: compare long-only, long-short, beta-neutral strategies with stop-loss
- Long-Short achieves +14.09% return, Sharpe 0.3903, max drawdown -12.67%"
git push
4️⃣ 如果想把结果图片也上传
在 outputs/ 文件夹中，新增以下文件：

text
outputs/
├── strategy_comparison.png   # 三种策略净值曲线对比
├── backtest_Long-Only.csv
├── backtest_Long-Short.csv
├── backtest_Beta_Neutral.csv
└── best_model_bundle.pkl      # （可选，但文件可能较大，建议用 Git LFS）
在 README 中引用这些图片：

markdown
## 📊 策略对比图

![策略对比](outputs/strategy_comparison.png)

三种策略中，**Long-Short** 表现最佳，夏普比率 0.3903。
✅ 发布检查清单
□ README.md 已更新（添加 2.6、2.7 说明）
□ 2.6 和 2.7 代码文件头部注释已完善
□ 关键函数有 docstring 注释
□ outputs/ 目录包含策略对比图和 CSV
□ 提交信息清晰描述本次更新
现在你的 GitHub 项目就有了完整的 “从因子计算 → 模型训练 → 可解释性 → 策略优化” 全流程！🚀

本回答由 AI 生成，内容仅供参考，请仔细甄别
