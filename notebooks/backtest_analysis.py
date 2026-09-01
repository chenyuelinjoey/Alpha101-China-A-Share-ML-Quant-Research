# ================================================================
# 策略优化回测：多头 / 多空 / 基准对冲 + 止损机制
# 
# 功能：
#   1. 多头策略（Long-Only）：做多 Top N
#   2. 多空策略（Long-Short）：做多 Top N，做空 Bottom N
#   3. 基准对冲（Beta Neutral）：做多 Top N，做空市场代理（全市场等权平均）
#   4. 统一止损机制（个股月度跌幅超过 -10% 时强制平仓）
# 
# 运行时间：约 3-5 分钟
# 依赖：best_model_bundle.pkl（或直接使用已加载的预测值）
# ================================================================

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("【策略优化回测】多头 / 多空 / 基准对冲 + 止损")
print("=" * 80)

# ------------------------------
# 1. 加载数据和预测值
# ------------------------------
print("\n[1] 加载数据...")

# 加载预处理数据（获取日期、代码、真实收益）
with open('/kaggle/working/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
test_code = data['test_code']
test_date = data['test_date']
y_test = data['y_test']

# 加载最佳模型预测值（Ensemble_All_Tree）
with open('/kaggle/working/ensemble_preds.pkl', 'rb') as f:
    preds = pickle.load(f)
y_pred = preds['Ensemble_All_Tree']

# 构建回测基础数据框
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
# 2. 回测参数配置
# ------------------------------
TOP_N = 30              # 多头数量
BOTTOM_N = 30           # 空头数量
STOP_LOSS = -0.10       # 止损阈值（-10%）
TRADING_COST = 0.001    # 双边交易成本（0.1%）
REBALANCE_FREQ = 'M'    # 调仓频率：'M'（月度）或 'D'（每日）

print(f"\n[2] 回测参数:")
print(f"   多头数量: {TOP_N}")
print(f"   空头数量: {BOTTOM_N}")
print(f"   止损阈值: {STOP_LOSS*100:.0f}%")
print(f"   交易成本: {TRADING_COST*100:.1f}%")
print(f"   调仓频率: {REBALANCE_FREQ}")

# ------------------------------
# 3. 核心回测函数（统一接口）
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
    
    # 生成调仓日期（月初或每日）
    if rebalance_freq == 'M':
        df['year_month'] = df['date'].dt.to_period('M')
        rebalance_dates = df.groupby('year_month')['date'].min().tolist()
    else:  # 每日
        rebalance_dates = sorted(df['date'].unique())
    
    # 存储每日组合收益
    portfolio_returns = []
    current_holdings = {}  # {code: entry_price, ...}
    
    for i, trade_date in enumerate(rebalance_dates):
        # ---------- 调仓日：选股 ----------
        df_trade = df[df['date'] == trade_date]
        if len(df_trade) < top_n:
            continue
        
        # 按预测收益率排序
        df_sorted = df_trade.sort_values('pred_return', ascending=False)
        
        # 选定股票池
        long_codes = df_sorted.head(top_n)['code'].tolist()
        short_codes = df_sorted.tail(bottom_n)['code'].tolist()
        
        # 获取当日收盘价（用actual_return近似，这里用1作为基准）
        # 实际回测中，我们只关心收益率序列，不关心绝对价格
        # 因此我们记录持仓股票及其买入日
        current_holdings = {
            'long': {code: trade_date for code in long_codes},
            'short': {code: trade_date for code in short_codes}
        }
        
        # 确定本调仓周期的日期范围
        if i + 1 < len(rebalance_dates):
            next_date = rebalance_dates[i + 1]
        else:
            next_date = df['date'].max() + pd.Timedelta(days=1)
        
        period_dates = df[(df['date'] >= trade_date) & (df['date'] < next_date)]['date'].unique()
        
        # ---------- 周期内逐日计算收益 ----------
        for day in sorted(period_dates):
            df_day = df[df['date'] == day]
            
            # 计算多头收益
            long_returns = []
            for code, entry_date in current_holdings['long'].items():
                row = df_day[df_day['code'] == code]
                if len(row) == 0:
                    continue
                ret = row['actual_return'].iloc[0]
                
                # 止损逻辑：计算自买入以来的累计收益
                # 简化：如果当日收益低于止损阈值，则卖出（收益截断）
                if ret < stop_loss:
                    ret = stop_loss  # 硬止损
                long_returns.append(ret)
            
            # 计算空头收益（做空收益 = -actual_return）
            short_returns = []
            for code, entry_date in current_holdings['short'].items():
                row = df_day[df_day['code'] == code]
                if len(row) == 0:
                    continue
                ret = -row['actual_return'].iloc[0]  # 做空
                if ret < stop_loss:  # 空头止损（即价格上涨超过10%）
                    ret = stop_loss
                short_returns.append(ret)
            
            # 计算市场平均收益（用于基准对冲）
            market_ret = df_day['actual_return'].mean()
            
            # 策略组合
            if strategy == 'long_only':
                if long_returns:
                    port_ret = np.mean(long_returns)
                else:
                    port_ret = 0
            elif strategy == 'long_short':
                all_returns = long_returns + short_returns
                if all_returns:
                    port_ret = np.mean(all_returns)
                else:
                    port_ret = 0
            elif strategy == 'beta_neutral':
                # 做多Top N，做空市场（全市场等权平均）
                if long_returns:
                    long_avg = np.mean(long_returns)
                else:
                    long_avg = 0
                port_ret = long_avg - market_ret
            else:
                raise ValueError(f"未知策略: {strategy}")
            
            # 扣除交易成本（仅在调仓日当天扣除）
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
    
    # 转换为DataFrame
    df_results = pd.DataFrame(portfolio_returns)
    if len(df_results) == 0:
        return pd.DataFrame()
    
    # 计算净值
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
print("【策略优化回测完成】")
print("=" * 80)
