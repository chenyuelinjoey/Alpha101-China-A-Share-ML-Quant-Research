# ================================================================
# 阶段 2.1：基准模型（统计 + 轻量机器学习）
# 
# 功能：建立预测能力的“下限”基准，包括：
#   - 最简基线（均值、滞后1期、AR(1)）
#   - 经典核方法（SVR）
#   - 轻量循环神经网络（SimpleRNN）
# 
# 输出：
#   - benchmark_preds.pkl   : 所有基准模型的预测值（字典）
#   - benchmark_results.pkl : 所有基准模型的评估指标（字典）
# 
# 运行时间：约 15-20 分钟
# 依赖：preprocessed_data.pkl
# ================================================================

import numpy as np
import pandas as pd
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.svm import SVR
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("=" * 80)
print("【阶段 2.1】基准模型（统计 + 轻量 ML）")
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
test_code = data['test_code']      # 测试集股票代码
test_date = data['test_date']      # 测试集日期
feature_cols = data['feature_cols']# 特征名称列表
print(f"训练集: {X_train.shape[0]:,} 行 × {X_train.shape[1]} 列")
print(f"测试集: {X_test.shape[0]:,} 行 × {X_test.shape[1]} 列")

# ------------------------------
# 2. 评估函数（统一接口）
# ------------------------------
def evaluate(name, y_pred):
    """
    计算并打印模型的三个核心指标：
      - IC (信息系数) : 预测值与真实值的秩相关系数，衡量排序能力
      - MSE (均方误差) : 预测误差的平方均值，对大误差敏感
      - MAE (平均绝对误差) : 预测误差的绝对值均值，更直观
    注：当 y_pred 为常数时，IC 返回 NaN（不影响后续排序）
    """
    # 安全处理长度不一致的情况（截断至较短长度）
    if len(y_pred) != len(y_test):
        min_len = min(len(y_pred), len(y_test))
        y_pred = y_pred[:min_len]
        y_true = y_test[:min_len]
    else:
        y_true = y_test
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    # 若 y_pred 为常数，corrcoef 返回 nan（正常现象）
    ic = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"{name:25s} | IC: {ic:7.4f} | MSE: {mse:.6f} | MAE: {mae:.6f}")
    return {'IC': ic, 'MSE': mse, 'MAE': mae}

# 存储所有预测和结果
results = {}
preds = {}

# ------------------------------
# 3. 最简基准模型（Baseline Models）
# ------------------------------
print("\n[2] 最简基准模型（Baseline）...")

# 3.1 历史均值：预测未来收益率等于训练集均值（无预测能力，下限基准）
y_mean = np.mean(y_train)
pred_mean = np.full_like(y_test, y_mean)
results['Mean_Baseline'] = evaluate('Mean_Baseline', pred_mean)
preds['Mean_Baseline'] = pred_mean

# 3.2 滞后1期：预测值 = 前一天的实际收益率（捕捉一阶自相关）
#     注意：测试集第一天无前日数据，用训练集最后一天填充
y_shift = np.concatenate([[y_train[-1]], y_test[:-1]])
results['Lag1_Baseline'] = evaluate('Lag1_Baseline', y_shift)
preds['Lag1_Baseline'] = y_shift

# 3.3 AR(1) 模型：一阶自回归 y_t = c + φ * y_{t-1}
#     手动用最小二乘法拟合整个训练集（简化版，不分组）
y_lag = np.concatenate([[y_train[0]], y_train[:-1]])   # 构造滞后项
X_ar = np.column_stack([np.ones_like(y_lag), y_lag])   # 添加截距项
coef = np.linalg.lstsq(X_ar, y_train, rcond=None)[0]  # 求解系数 [c, φ]
pred_ar = coef[0] + coef[1] * np.concatenate([[y_train[-1]], y_test[:-1]])
results['AR(1)'] = evaluate('AR(1)', pred_ar)
preds['AR(1)'] = pred_ar

# ------------------------------
# 4. 支持向量回归（SVR）
# ------------------------------
print("\n[3] 支持向量回归（SVR）...")
# SVR 在大数据集上训练较慢，随机采样 5 万行训练（保持代表性）
sample_size_svr = min(50000, len(X_train))
np.random.seed(42)  # 固定随机种子保证可复现
idx_svr = np.random.choice(len(X_train), sample_size_svr, replace=False)

model_svr = SVR(kernel='rbf', C=1.0, gamma='scale')
model_svr.fit(X_train[idx_svr], y_train[idx_svr])
pred_svr = model_svr.predict(X_test)
results['SVR'] = evaluate('SVR', pred_svr)
preds['SVR'] = pred_svr

# 释放内存
del model_svr
gc.collect()
print("   SVR 训练完成，内存已释放。")

# ------------------------------
# 5. SimpleRNN（单层循环神经网络）
# ------------------------------
print("\n[4] SimpleRNN（单层循环神经网络）...")
# SimpleRNN 用于捕捉时间序列中的短期依赖，作为 LSTM/Transformer 的轻量基准
# 同样采样 5 万行，序列长度设为 10（即用过去 10 个交易日预测下一日）

def create_sequences(X, y, seq_len=10):
    """
    将原始面板数据转换为滑动窗口序列
    参数：
        X : 特征矩阵 (n_samples, n_features)
        y : 标签 (n_samples,)
        seq_len : 窗口长度
    返回：
        X_seq : (n_samples - seq_len, seq_len, n_features)
        y_seq : (n_samples - seq_len,)
    """
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

seq_len_rnn = 10
sample_size_rnn = 50000
np.random.seed(42)
idx_rnn = np.random.choice(len(X_train), sample_size_rnn, replace=False)

X_train_rnn, y_train_rnn = create_sequences(X_train[idx_rnn], y_train[idx_rnn], seq_len_rnn)
X_test_rnn, y_test_rnn = create_sequences(X_test, y_test, seq_len_rnn)
print(f"   训练序列形状: {X_train_rnn.shape}, 测试序列形状: {X_test_rnn.shape}")

# 定义 SimpleRNN 模型（单层 RNN + 全连接输出）
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)          # out: (batch, seq_len, hidden)
        out = out[:, -1, :]           # 取最后一个时间步的输出
        return self.fc(out).squeeze(-1)

def train_rnn(model, X_train, y_train, X_test, y_test, epochs=15, batch_size=256):
    """通用的 RNN 训练函数（适用于 SimpleRNN, LSTM 等）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test)),
        batch_size=batch_size, shuffle=False
    )
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        model.train()
        loss_sum = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"   RNN Epoch {epoch+1}/{epochs}, Loss: {loss_sum/len(train_loader):.6f}")
    
    # 测试集预测
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            Xb = Xb.to(device)
            pred = model(Xb)
            preds.extend(pred.cpu().numpy())
    return np.array(preds)

model_rnn = SimpleRNN(X_train_rnn.shape[2])
pred_rnn = train_rnn(model_rnn, X_train_rnn, y_train_rnn, X_test_rnn, y_test_rnn, epochs=15)

# 单独评估（因为 y_test_rnn 与原始 y_test 长度不同，需单独计算）
mse_rnn = mean_squared_error(y_test_rnn, pred_rnn)
mae_rnn = mean_absolute_error(y_test_rnn, pred_rnn)
ic_rnn = np.corrcoef(y_test_rnn, pred_rnn)[0, 1]
print(f"SimpleRNN                | IC: {ic_rnn:7.4f} | MSE: {mse_rnn:.6f} | MAE: {mae_rnn:.6f}")
results['SimpleRNN'] = {'IC': ic_rnn, 'MSE': mse_rnn, 'MAE': mae_rnn}
preds['SimpleRNN'] = pred_rnn

del model_rnn
gc.collect()
print("   SimpleRNN 训练完成，内存已释放。")

# ------------------------------
# 6. 保存阶段结果
# ------------------------------
print("\n[5] 保存阶段结果...")
with open('/kaggle/working/benchmark_preds.pkl', 'wb') as f:
    pickle.dump(preds, f)
with open('/kaggle/working/benchmark_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("✅ 基准模型结果已保存至:")
print("   - /kaggle/working/benchmark_preds.pkl")
print("   - /kaggle/working/benchmark_results.pkl")
print("\n" + "=" * 80)
print("阶段 2.1 完成！")
print("=" * 80)
