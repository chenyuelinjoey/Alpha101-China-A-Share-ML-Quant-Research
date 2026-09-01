# ================================================================
# 阶段 2.4：深度学习与 Autoencoder 特征提取
# 
# 功能：
#   1. Autoencoder：将 101 维因子压缩至 32 维，测试降维是否保留有效信息
#   2. LSTM：长短期记忆网络，捕捉时间序列依赖
#   3. Transformer：自注意力机制，捕捉长期依赖（可选，默认启用）
# 
# 输出：
#   - deep_preds.pkl   : 深度模型的预测值（字典）
#   - deep_results.pkl : 深度模型的评估指标（字典）
# 
# 运行时间：约 30-40 分钟（若仅跑 Autoencoder 约 5 分钟）
# 依赖：preprocessed_data.pkl
# ================================================================

import numpy as np
import pandas as pd
import pickle
import gc
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

print("=" * 80)
print("【阶段 2.4】深度学习与 Autoencoder 特征提取")
print("=" * 80)

# ------------------------------
# 1. 加载预处理数据
# ------------------------------
print("\n[1] 加载预处理数据...")
with open('/kaggle/working/preprocessed_data.pkl', 'rb') as f:
    data = pickle.load(f)
X_train = data['X_train']
y_train = data['y_train']
X_test = data['X_test']
y_test = data['y_test']
feature_cols = data['feature_cols']
print(f"训练集: {X_train.shape[0]:,} 行 × {X_train.shape[1]} 列")
print(f"测试集: {X_test.shape[0]:,} 行 × {X_test.shape[1]} 列")

# ------------------------------
# 2. 评估函数（统一接口）
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
# 3. Autoencoder 自编码器（特征压缩）
# ------------------------------
print("\n[2] Autoencoder 自编码器（101维 → 32维）...")
# 自编码器通过无监督学习将高维特征压缩成低维表示，
# 然后在此低维空间上训练 LightGBM，测试降维是否保留有效信息。
# 这可以视为一种特征工程方法，有助于去除噪声。

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

def train_autoencoder(X, encoding_dim=32, epochs=20, batch_size=256):
    """训练自编码器，返回编码后的特征和模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = Autoencoder(X.shape[1], encoding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        loss_sum = 0
        for (Xb,) in loader:
            Xb = Xb.to(device)
            encoded, decoded = model(Xb)
            loss = criterion(decoded, Xb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
        if (epoch+1) % 10 == 0:
            print(f"   AE Epoch {epoch+1}/{epochs}, Loss: {loss_sum/len(loader):.6f}")
    # 提取编码器
    encoder = model.encoder
    encoder.eval()
    with torch.no_grad():
        X_enc = encoder(torch.FloatTensor(X).to(device)).cpu().numpy()
    return X_enc, model

# 使用部分数据训练 AE（加速）
sample_size_ae = min(50000, len(X_train))
np.random.seed(42)
idx_ae = np.random.choice(len(X_train), sample_size_ae, replace=False)
print(f"   使用 {sample_size_ae:,} 个样本训练 Autoencoder...")
X_train_ae, ae_model = train_autoencoder(X_train[idx_ae], encoding_dim=32, epochs=20)

# 对全量训练集和测试集编码
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = ae_model.encoder.to(device)
encoder.eval()
with torch.no_grad():
    X_train_enc = encoder(torch.FloatTensor(X_train).to(device)).cpu().numpy()
    X_test_enc = encoder(torch.FloatTensor(X_test).to(device)).cpu().numpy()
print(f"   编码后训练集形状: {X_train_enc.shape}, 测试集形状: {X_test_enc.shape}")

# 在压缩特征上训练 LightGBM
print("   在压缩特征上训练 LightGBM...")
lgb_ae = LGBMRegressor(n_estimators=80, learning_rate=0.1, num_leaves=31, random_state=42, verbose=-1)
lgb_ae.fit(X_train_enc, y_train)
pred_ae_lgb = lgb_ae.predict(X_test_enc)
results['AE_LightGBM'] = evaluate('AE_LightGBM', pred_ae_lgb)
preds['AE_LightGBM'] = pred_ae_lgb

del ae_model, encoder, lgb_ae
gc.collect()
print("   Autoencoder 完成，内存已释放。")

# ------------------------------
# 4. 时序序列数据构造（用于 LSTM / Transformer）
# ------------------------------
def create_sequences(X, y, seq_len=20):
    """将面板数据转换为滑动窗口序列"""
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

seq_len = 20
sample_size_lstm = 80000  # 采样数量（可调，80k 在 GPU 上约 10 分钟）
np.random.seed(42)
idx_lstm = np.random.choice(len(X_train), sample_size_lstm, replace=False)
X_train_seq, y_train_seq = create_sequences(X_train[idx_lstm], y_train[idx_lstm], seq_len)
X_test_seq, y_test_seq = create_sequences(X_test, y_test, seq_len)
print(f"\n[3] 序列数据形状: 训练 {X_train_seq.shape}, 测试 {X_test_seq.shape}")

# ------------------------------
# 5. 通用深度学习训练函数
# ------------------------------
def train_deep_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=256, name="Model"):
    """
    通用的深度学习模型训练函数（支持 LSTM、Transformer 等）
    使用 MSE 损失和 Adam 优化器，学习率 0.001
    """
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
            print(f"   {name} Epoch {epoch+1}/{epochs}, Loss: {loss_sum/len(train_loader):.6f}")
    
    # 测试集预测
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in test_loader:
            Xb = Xb.to(device)
            pred = model(Xb)
            preds.extend(pred.cpu().numpy())
    return np.array(preds)

# ------------------------------
# 6. LSTM 模型
# ------------------------------
print("\n[4] 训练 LSTM...")
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]      # 取最后一个时间步的输出
        return self.fc(out).squeeze(-1)

lstm_model = LSTMModel(X_train_seq.shape[2])
pred_lstm = train_deep_model(lstm_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
                             epochs=20, name="LSTM")
results['LSTM'] = evaluate('LSTM', pred_lstm)
preds['LSTM'] = pred_lstm
del lstm_model
gc.collect()
print("   LSTM 完成，内存已释放。")

# ------------------------------
# 7. Transformer 模型（自注意力机制）
# ------------------------------
print("\n[5] 训练 Transformer...")
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        # 位置编码：形状 (1, seq_len, d_model) 以便广播
        self.register_buffer('pos_encoder', self._get_positional_encoding(1, 1024, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)
    
    def _get_positional_encoding(self, batch_size, max_len, d_model):
        """生成固定的正弦位置编码"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        return pe
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)                      # (batch, seq_len, d_model)
        # 添加位置编码（截取相应长度）
        x = x + self.pos_encoder[:, :x.size(1), :]  # (batch, seq_len, d_model)
        x = self.transformer(x)                     # (batch, seq_len, d_model)
        x = x[:, -1, :]                             # 取最后一个时间步
        return self.fc(x).squeeze(-1)

trans_model = TransformerModel(X_train_seq.shape[2])
pred_trans = train_deep_model(trans_model, X_train_seq, y_train_seq, X_test_seq, y_test_seq,
                              epochs=15, name="Transformer")
results['Transformer'] = evaluate('Transformer', pred_trans)
preds['Transformer'] = pred_trans
del trans_model
gc.collect()
print("   Transformer 完成，内存已释放。")

# ------------------------------
# 8. 保存阶段结果
# ------------------------------
print("\n[6] 保存阶段结果...")
with open('/kaggle/working/deep_preds.pkl', 'wb') as f:
    pickle.dump(preds, f)
with open('/kaggle/working/deep_results.pkl', 'wb') as f:
    pickle.dump(results, f)
print("✅ 深度学习与 Autoencoder 结果已保存至:")
print("   - /kaggle/working/deep_preds.pkl")
print("   - /kaggle/working/deep_results.pkl")

print("\n" + "=" * 80)
print("阶段 2.4 完成！")
print("=" * 80)
