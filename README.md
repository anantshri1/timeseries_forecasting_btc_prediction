# Bitcoin Time Series Forecasting: LSTM vs Transformer vs TFT

A deep learning project comparing three sequence models — a stacked LSTM, a custom Transformer encoder, and a Temporal Fusion Transformer (TFT) — on two forecasting tasks using hourly Bitcoin data from 2018 to 2026.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Data](#2-data)
3. [Feature Engineering](#3-feature-engineering)
4. [Forecasting Targets](#4-forecasting-targets)
5. [Modelling Pipeline](#5-modelling-pipeline)
6. [Models](#6-models)
7. [Results](#7-results)
8. [Key Findings](#8-key-findings)
9. [References](#9-references)

---

## 1. Project Overview

This project addresses two distinct forecasting problems on hourly BTC/USDT data:

- **Task A — Return Forecasting**: predict the next-hour log return `L(t+1)`
- **Task B — Volatility Forecasting**: predict the next-hour 24h realised volatility `σ(t+1)`

Three architectures are compared:
- A stacked LSTM (TensorFlow/Keras)
- A custom Transformer encoder with multi-head self-attention (TensorFlow/Keras)
- A Temporal Fusion Transformer via `neuralforecast` (PyTorch/Lightning), with automatic hyperparameter tuning via Ray Tune

The project is motivated by prior work on deep learning for cryptocurrency forecasting. Feature engineering follows the methodology of Livieris et al. (2021) and Zhang et al. (2024), with additional engineered features not present in those works (candle structure, taker ratio, Bollinger Band position). Model selection and architecture choices follow the framework of Chen et al. (2024).

---

## 2. Data

**Source**: [Bitcoin Historical Datasets 2018–2024](https://www.kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024) via KaggleHub  
**Frequency**: 1-hour OHLCV candles  
**Period**: January 2018 – March 2026  
**Total rows**: ~71,000 hourly observations

Raw columns used: `Open`, `High`, `Low`, `Close`, `Volume`, `Taker buy base asset volume`, `Number of trades`, `Open time`.

```python
import kagglehub
path = kagglehub.dataset_download("novandraanugrah/bitcoin-historical-datasets-2018-2024")
df = pd.read_csv(path + '/btc_1h_data_2018_to_2025.csv')
```

---

## 3. Feature Engineering

All features are derived from the raw OHLCV data. No external data (e.g. sentiment, on-chain) is used. The feature set is informed by Livieris et al. (2021) and Zhang et al. (2024), with the following additions.

### 3.1 Return and Volatility

```python
df['log_return']        = np.log(df['Close'] / df['Close'].shift(1))
df['realised_vol_24h']  = df['log_return'].rolling(24).std()
df['realised_vol_168h'] = df['log_return'].rolling(168).std()
```

Log return is defined as in Chen et al. (2024):

> L(t) = log(P(t) / P(t−1))

where P(t) is the close price at period t. Realised volatility at horizons of 24h and 168h (one week) captures short- and medium-term volatility regimes. The ARCH test (p < 0.001) confirmed statistically significant volatility clustering in the series, motivating volatility as a forecasting target in its own right.

### 3.2 Candle Structure

```python
df['candle_body'] = (df['Close'] - df['Open']) / df['Open']
df['upper_wick']  = (df['High'] - df[['Open','Close']].max(axis=1)) / df['Open']
df['lower_wick']  = (df[['Open','Close']].min(axis=1) - df['Low']) / df['Open']
df['hl_range']    = (df['High'] - df['Low']) / df['Open']
```

These price action features encode intra-candle dynamics not captured by close-price-only approaches.

### 3.3 Volume Signals

```python
df['taker_ratio']      = df['Taker buy base asset volume'] / df['Volume']
df['volume_ma_ratio']  = df['Volume'] / df['Volume'].rolling(24).mean()
```

`taker_ratio` measures buy-side pressure (proportion of volume initiated by buyers). `volume_ma_ratio` captures volume spikes relative to the recent 24h average.

### 3.4 Momentum Indicators

```python
# RSI (14-period)
delta = df['Close'].diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = -delta.clip(upper=0).rolling(14).mean()
df['rsi'] = 100 - (100 / (1 + gain / loss))

# MACD
ema12 = df['Close'].ewm(span=12).mean()
ema26 = df['Close'].ewm(span=26).mean()
df['macd']        = ema12 - ema26
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_hist']   = df['macd'] - df['macd_signal']

# Bollinger Band position
ma20  = df['Close'].rolling(20).mean()
std20 = df['Close'].rolling(20).std()
df['bb_position'] = (df['Close'] - ma20) / (2 * std20)
```

### 3.5 Full Feature List

| Feature | Description |
|---------|-------------|
| `log_return` | Hourly log return |
| `realised_vol_24h` | 24h rolling return std |
| `realised_vol_168h` | 168h rolling return std |
| `candle_body` | Signed intra-candle return |
| `upper_wick` | Upper wick relative to open |
| `lower_wick` | Lower wick relative to open |
| `hl_range` | High-low range relative to open |
| `taker_ratio` | Buy-side volume proportion |
| `volume_ma_ratio` | Volume relative to 24h average |
| `rsi` | 14-period RSI |
| `macd` | MACD line |
| `macd_signal` | MACD signal line |
| `macd_hist` | MACD histogram |
| `bb_position` | Position within Bollinger Bands |
| `Number of trades` | Raw trade count per hour |

**Note on feature inclusion**: following the autoregressive modelling convention (Box & Jenkins, 1970), `log_return` and `realised_vol_24h` appear as both input features and forecasting targets. The input window contains values at times `t-n` through `t`; the target is the value at `t+1`. There is no data leakage.

---

## 4. Forecasting Targets

### Task A — Log Return

```python
A['target'] = A['log_return'].shift(-1)
```

The next-hour log return. This is a regression task. The ADF test (p < 0.001) confirmed stationarity. The target is **not** scaled — log returns are already near-zero and bounded, and the output layer (`Dense(1)`, no activation) is unconstrained.

### Task B — 24h Realised Volatility

```python
B['target'] = np.log(B['realised_vol_24h'].shift(-1))
```

The log of next-hour 24h realised volatility. Log-transforming the target is standard practice in volatility modelling — the raw series has skewness ~50, while the log-transformed series is approximately symmetric. Predictions are inverse-transformed (`np.exp`) before metric computation.

---

## 5. Modelling Pipeline

### 5.1 Train / Validation / Test Split

All splits are strictly temporal — **no shuffling at any point**.

```python
n         = len(df_model)
train_end = int(n * 0.70)   # ~50,000 hours
val_end   = int(n * 0.85)   # ~10,700 hours each for val and test

train_df = df_model.iloc[:train_end]
val_df   = df_model.iloc[train_end:val_end]
test_df  = df_model.iloc[val_end:]
```

| Split | Period | Size |
|-------|--------|------|
| Train | 2018–2022 | ~50,000 hrs |
| Validation | 2022–2024 | ~10,700 hrs |
| Test | 2024–2026 | ~10,700 hrs |

### 5.2 Feature Scaling

`StandardScaler` is fit **only on the training set** and applied to validation and test sets. This prevents data leakage from future observations into the scaling parameters.

```python
scaler  = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols])
X_val   = scaler.transform(val_df[feature_cols])
X_test  = scaler.transform(test_df[feature_cols])
```

### 5.3 Sequence Construction

For LSTM and Transformer, input sequences of length `WINDOW` are constructed via a sliding window:

```python
def create_windows(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)
```

- **Task A (log return)**: `WINDOW = 48` hours
- **Task B (volatility)**: `WINDOW = 24` hours (one full cycle of the 24h rolling window). AutoTFT selected `input_size=48` as optimal for Task B.

The TFT (via `neuralforecast`) handles windowing internally.

### 5.4 Evaluation Metrics

- **MAE** and **RMSE** for both tasks (in original units after inverse transform for Task B)
- **Directional Accuracy** for Task A only: proportion of hours where `sign(y_pred) == sign(y_true)`. A random predictor achieves 50%; any result above 52% is considered meaningful in the literature.

---

## 6. Models

### 6.1 LSTM

A stacked two-layer LSTM implemented in TensorFlow/Keras.

```python
def build_lstm(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='mae'
    )
    return model
```

- Loss: MAE (chosen over MSE/Huber to avoid mean-collapse on near-zero log returns)
- Early stopping: patience=10, monitor val_loss, restore best weights
- LR scheduler: ReduceLROnPlateau, factor=0.5, patience=3

### 6.2 Custom Transformer

A custom Transformer encoder with sinusoidal positional encoding, implemented from scratch in TensorFlow/Keras following Vaswani et al. (2017).

Architecture:
- Input projection: `Dense(embed_dim=128)`
- Sinusoidal positional encoding
- 2× Transformer blocks: multi-head self-attention (4 heads) + FFN (256 units) + residual connections + LayerNorm
- Output: last timestep → `Dropout(0.2)` → `Dense(1)`

Same training configuration as LSTM.

### 6.3 Temporal Fusion Transformer (TFT)

Implemented via `neuralforecast` (Nixtla). The TFT architecture (Lim et al., 2021) combines gating layers, an LSTM recurrent encoder, and multi-head attention for interpretable multi-horizon forecasting.

For Task A, a fixed TFT configuration is used. For Task B, **AutoTFT** with Ray Tune is used to search over:

```python
config = {
    "input_size":          tune.choice([24, 48]),
    "hidden_size":         tune.choice([32, 64]),
    "n_head":              tune.choice([2, 4]),
    "learning_rate":       tune.loguniform(1e-4, 1e-2),
    "scaler_type":         tune.choice(['standard', 'robust']),
    "max_steps":           tune.choice([500, 1000]),
    "windows_batch_size":  tune.choice([32, 64]),
    "random_seed":         tune.randint(1, 10),
}
```

Best configuration selected by AutoTFT: `input_size=48` (48h window outperformed 24h for volatility forecasting).

Evaluation uses `nf.cross_validation` with `step_size=1` across the full test period.

---

## 7. Results

### Task A — Log Return Forecasting

| Model | MAE | RMSE | Directional Accuracy |
|-------|-----|------|----------------------|
| LSTM | 0.003193| 0.004937 | **51.85%** |
| Transformer | 0.003179 | 0.004912 | 51.32% |
| TFT | 0.003216 | 0.004983 | 51.42% |

LSTM achieves the highest directional accuracy. All three models produce predictions with substantially lower variance than the true return series (y_true std ≈ 0.0049), consistent with the low signal-to-noise ratio of hourly cryptocurrency returns. A random baseline achieves 50% directional accuracy.

**Note**: visual inspection of predicted vs actual series reveals that the TFT captures the stochastic structure of returns more faithfully than the LSTM — its predictions exhibit variance closer to the true series. Directional accuracy as a scalar metric does not fully capture this.

### Task B — 24h Volatility Forecasting

All metrics are in original (non-log) units after inverse transformation.

| Model | MAE | RMSE |
|-------|-----|------|
| LSTM | 0.000400 | 0.000607 |
| Transformer (bias corrected) | 0.000554 | 0.000788 |
| TFT (AutoTFT) | **0.000156** | **0.000363** |

TFT wins by a large margin on volatility. The Transformer result required post-hoc bias correction (level shift of ~0.4 in log space due to early convergence); the corrected metrics are reported. The LSTM result is reported directly without correction.

---

## 8. Key Findings

**LSTM wins on return direction, TFT wins on volatility magnitude.** These results are consistent with the nature of each target:

- Hourly log returns are close to a random walk with very low autocorrelation. All models achieve directional accuracy only marginally above 50%, consistent with the efficient market hypothesis at short horizons. The LSTM's sequential inductive bias is sufficient for this task; additional attention capacity does not help.

- Realised volatility is strongly autocorrelated (confirmed by ARCH test and ACF of squared returns). The TFT's combination of LSTM encoding, attention, and automatic hyperparameter tuning allows it to exploit this persistence more effectively than the other architectures.

**Loss function matters more than architecture for return forecasting.** Initial experiments with MSE and Huber loss produced near-constant predictions (y_pred std ≈ 0.00003 vs y_true std ≈ 0.0049). Switching to MAE reduced this variance collapse substantially. This is a known failure mode of squared loss on financial returns where the optimal MSE predictor is the conditional mean, which for log returns is approximately zero.

**Log-transforming skewed targets is non-negotiable.** Raw 24h realised volatility has skewness ~50. Log-transforming before training and inverse-transforming at evaluation produced stable training and well-calibrated predictions.

---

## 9. References

- Chen, Y. et al. (2024). *Deep Learning Models for Bitcoin Price Prediction*. ACM. https://doi.org/10.1145/3650215.3650232
- Livieris, I. E. et al. (2021). *Time-series forecasting of Bitcoin prices using high-dimensional features: a machine learning approach*. Neural Computing and Applications. https://doi.org/10.1007/s00521-020-05129-6
- Zhang, X. et al. (2025). *Bitcoin Forecasting with Classical Time Series Models on Prices and Volatility*. arXiv:2511.06224
- Lim, B. et al. (2021). *Temporal fusion transformers for interpretable multi-horizon time series forecasting*. International Journal of Forecasting, 37(4), 1748–1764.
- Vaswani, A. et al. (2017). *Attention is all you need*. NeurIPS.
- Box, G. E. P. & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
- Fischer, T. & Krauss, C. (2018). *Deep learning with long short-term memory networks for financial market predictions*. European Journal of Operational Research, 270(2), 654–669.
