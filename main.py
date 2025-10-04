import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings('ignore')

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  

mode = 'train'  # Установите 'train' или 'predict'

df_candles = pd.concat([
    pd.read_csv('candles.csv'),
    pd.read_csv('candles_2.csv')
], ignore_index=True)

df_candles['begin'] = pd.to_datetime(df_candles['begin'], utc=True)
df_candles = df_candles.sort_values(['ticker', 'begin'])
df_candles['date_index'] = df_candles.groupby('ticker').cumcount()

train_df = df_candles[df_candles['begin'] < '2025-09-08']
test_df = df_candles[df_candles['begin'] >= '2025-09-08']

for df in [train_df, test_df]:
    df['lag_close_1'] = df.groupby('ticker')['close'].shift(1)
    df['lag_close_2'] = df.groupby('ticker')['close'].shift(2)
    df['lag_close_3'] = df.groupby('ticker')['close'].shift(3)
    df['ma_5'] = df.groupby('ticker')['close'].rolling(window=5).mean().reset_index(0, drop=True)
    df['ma_20'] = df.groupby('ticker')['close'].rolling(window=20).mean().reset_index(0, drop=True)
    df['range'] = df['high'] - df['low']
    df['lag_volume_1'] = df.groupby('ticker')['volume'].shift(1)
    df['lag_volume_2'] = df.groupby('ticker')['volume'].shift(2)
    df['volatility'] = df.groupby('ticker')['close'].rolling(window=5).std().reset_index(0, drop=True)
    delta = df.groupby('ticker')['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))
    exp1 = df.groupby('ticker')['close'].ewm(span=12, adjust=False).mean().reset_index(0, drop=True)
    exp2 = df.groupby('ticker')['close'].ewm(span=26, adjust=False).mean().reset_index(0, drop=True)
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df.groupby('ticker')['macd'].ewm(span=9, adjust=False).mean().reset_index(0, drop=True)
    df['macd_hist'] = df['macd'] - df['macd_signal']

for col in ['range', 'volatility', 'rsi', 'macd', 'macd_signal', 'macd_hist']:
    upper_bound = train_df[col].quantile(0.99)
    train_df[col] = train_df[col].clip(upper=upper_bound)
    test_df[col] = test_df[col].clip(upper=upper_bound)

train_df = train_df.dropna()
test_df_last = test_df  

features = ['lag_close_1', 'lag_close_2', 'lag_close_3', 'ma_5', 'ma_20', 'range', 'lag_volume_1', 'lag_volume_2', 'volatility', 'rsi', 'macd', 'macd_signal', 'macd_hist']
X_train = train_df[features].values
X_test = test_df_last[features].values

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

horizons = range(1, 21)
y_train = {f'y_{i}': train_df.groupby('ticker')['close'].pct_change(periods=i).shift(-i).fillna(0).values for i in horizons}

if mode == 'train':
    X_train_split, X_val, y_train_split, y_val = {}, {}, {}, {}
    for i in horizons:
        X_train_split[f'X_{i}'], X_val[f'X_{i}'], y_train_split[f'y_{i}'], y_val[f'y_{i}'] = train_test_split(
            X_train_scaled, y_train[f'y_{i}'], test_size=0.2, random_state=42
        )

    models = {f'model_{i}': XGBRegressor(
        random_state=42,
        n_estimators=500,
        learning_rate=0.03,
        max_depth=15,
        min_child_weight=1e-3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        scale_pos_weight=1.5,
        tree_method='gpu_hist',
        predictor='gpu_predictor'
    ) for i in horizons}

    for i in tqdm(horizons, desc="Обучение и сохранение моделей"):
        models[f'model_{i}'].fit(X_train_split[f'X_{i}'], y_train_split[f'y_{i}'])
        with open(f'model_{i}.pkl', 'wb') as f:
            pickle.dump(models[f'model_{i}'], f)

elif mode == 'predict':
    models = {}
    for i in horizons:
        with open(f'model_{i}.pkl', 'rb') as f:
            models[f'model_{i}'] = pickle.load(f)

    pred_test = np.column_stack([models[f'model_{i}'].predict(X_test_scaled) for i in horizons])

    result_df_all = test_df_last[['ticker', 'begin']].copy()
    for i in range(20):
        result_df_all[f'p{i+1}'] = pred_test[:, i]
    result_df_all['weekday'] = result_df_all['begin'].dt.weekday
    mask_weekend = (result_df_all['weekday'] == 5) | (result_df_all['weekday'] == 6)
    for i in range(20):
        result_df_all.loc[mask_weekend, f'p{i+1}'] = 0
    final_result_all = result_df_all[['ticker'] + [f'p{i+1}' for i in range(20)]]
    final_result_all.to_csv('predictions_all.csv', index=False)
    print("Предсказания со всеми днями сохранены в predictions_all.csv")

    result_df_workdays = result_df_all[~mask_weekend].copy()
    final_result_workdays = result_df_workdays[['ticker'] + [f'p{i+1}' for i in range(20)]]
    final_result_workdays.to_csv('predictions_workdays.csv', index=False)
    print("Предсказания только для рабочих дней сохранены в predictions_workdays.csv")
