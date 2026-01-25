import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

tvl_path = "ethereum_tvl_2023-01-01_2026-01-01.csv"
price_path = "kline_ETHUSDT_D_20230101_20260101_spot.csv"

tvl_df = pd.read_csv(tvl_path)
price_df = pd.read_csv(price_path)

tvl_df['date'] = pd.to_datetime(tvl_df['date']).dt.date
price_df['date'] = pd.to_datetime(price_df['datetime']).dt.date

price_df = price_df[['date', 'close']].rename(columns={'close': 'eth_price'})

df = pd.merge(tvl_df, price_df, on='date', how='inner')

df['price_neutral_tvl'] = df['tvl_usd'] / df['eth_price']
df['price_neutral_tvl_2dec'] = df['price_neutral_tvl'].round(2)

df['eth_return'] = df['eth_price'].pct_change()
df['pntvl_change'] = df['price_neutral_tvl_2dec'].pct_change()

# =========================================================
# 8. ⭐ 背离强度指标（核心）
# 负相关结构下：
# ETH 涨 & PNTVL 跌 → divergence > 0
# ETH 跌 & PNTVL 涨 → divergence < 0
# =========================================================
df['divergence_strength'] = df['eth_return'] - df['pntvl_change']

# =========================================================
# 9. ⭐ 信号生成
# =========================================================
threshold = 0.01   # 可调参数

df['signal'] = 0

# 做多：ETH 跌 + PNTVL 涨 + 背离足够强
df.loc[
    (df['eth_return'] < 0) &
    (df['pntvl_change'] > 0) &
    (df['divergence_strength'] < -threshold),
    'signal'
] = 1

# 做空：ETH 涨 + PNTVL 跌 + 背离足够强
df.loc[
    (df['eth_return'] > 0) &
    (df['pntvl_change'] < 0) &
    (df['divergence_strength'] > threshold),
    'signal'
] = -1

# =========================================================
# 10. ⭐ T+1 执行（防未来函数）
# =========================================================
df['position'] = df['signal'].shift(1).fillna(0)

# =========================================================
# 17. 打印结果（前20 + 后20）
# =========================================================
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)
pd.set_option('display.float_format', '{:.4f}'.format)

print(df[['date','eth_price','price_neutral_tvl_2dec',
          'eth_return','pntvl_change',
          'divergence_strength','signal','position']])
