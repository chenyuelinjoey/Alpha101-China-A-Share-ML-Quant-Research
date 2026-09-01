# Alpha101-China-A-Share-ML-Quant-Research

**Short Description:** A complete quantitative research framework for China A-share market using Alpha101 factors and ML ensemble models with SHAP interpretability.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Key Results

- ✅ 101 Alpha Factors with true VWAP
- ✅ 19 ML Models (Ridge, LGB, XGB, CatBoost, RF, LSTM, Transformer)
- ✅ Best Model: Ensemble_All_Tree (IC=0.0552) 🏆
- ✅ SHAP Feature Importance Analysis
- ✅ Backtesting with transaction costs
- ✅ Deployable model bundle

## 📁 Repository Structure

Alpha101-China-A-Share-ML-Quant-Research/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 2.1_benchmark_models.py
│   ├── 2.2_tree_models.py
│   ├── 2.3_ensemble_models.py
│   ├── 2.4_deep_learning.py
│   ├── 2.5_summary_shap.py
│   └── backtest_analysis.py
│
└── outputs/
    ├── all_models_ranking.csv
    ├── shap_bar.png
    ├── shap_summary.png
    ├── backtest_analysis.png
    └── submission_final.csv
