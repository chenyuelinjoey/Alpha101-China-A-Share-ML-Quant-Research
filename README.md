# Alpha101-China-A-Share-ML-Quant-Research

**A complete quantitative research framework for China A-share market using Alpha101 factors and machine learning ensemble models with SHAP interpretability**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle)](https://kaggle.com)

---

## 📊 Key Results

| Metric | Result |
|--------|--------|
| 🏆 **Best Model** | **Ensemble_All_Tree** (LightGBM + XGBoost + CatBoost + RandomForest) |
| 📈 **Best IC** | **0.0552** |
| 📉 **Best MSE** | 0.000547 |
| 📊 **Best MAE** | 0.016060 |
| 🔍 **Most Important Factor** | **alpha012** (Price-Volume Divergence) |
| 💰 **Best Strategy (Long-Short)** | **+14.09%** total return (2021-2025) |
| 📊 **Sharpe Ratio (Long-Short)** | **0.3903** |

---

## 🧩 Technical Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                              │
├─────────────────────────────────────────────────────────────────┤
│    CSMAR A-share data → CSI 300 constituents → Lag processing   │
│    True VWAP = amount / volume → 101 Alpha factors              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Model Layer                              │
├─────────────────────────────────────────────────────────────────┤
│    Linear: Ridge, Lasso, ElasticNet                             │
│    Tree: LightGBM, XGBoost, CatBoost, RandomForest              │
│    Ensemble: Ensemble_All_Tree (LGB+XGB+Cat+RF)                 │
│    Deep: LSTM, Transformer                                      │
│    Feature Extraction: Autoencoder                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Interpretation Layer                        │
├─────────────────────────────────────────────────────────────────┤
│   SHAP global feature importance analysis                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Backtest Layer                            │
├─────────────────────────────────────────────────────────────────┤
│   Long-Only / Long-Short / Beta Neutral / Stop-loss             │
│   Performance metrics: Sharpe, Max DD, Win rate                 │
└─────────────────────────────────────────────────────────────────┘
