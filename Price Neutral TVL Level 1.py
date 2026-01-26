import pandas as pd
import numpy as np

# ========= 1. 读取 CSV =========
tvl_path = "ethereum_tvl_2023-01-01_2026-01-01.csv"
price_path = "kline_ETHUSDT_D_20230101_20260101_spot.csv"

tvl_df = pd.read_csv(tvl_path)
price_df = pd.read_csv(price_path)

# ========= 2. 统一日期 =========
tvl_df['date'] = pd.to_datetime(tvl_df['date']).dt.date
price_df['date'] = pd.to_datetime(price_df['datetime']).dt.date

# ========= 3. 精简价格列 =========
price_df = price_df[['date', 'close']].rename(columns={'close': 'eth_price'})

# ========= 4. 合并 TVL 与价格 =========
df = pd.merge(tvl_df, price_df, on='date', how='inner')

# ========= 5. 计算价格中性 TVL =========
df['price_neutral_tvl'] = df['tvl_usd'] / df['eth_price']

# ========= 6. 保留小数点后 2 位 =========
df['price_neutral_tvl_2dec'] = df['price_neutral_tvl'].round(2)

# ========= 打印（前后截断显示） =========
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.float_format', '{:.2f}'.format)

print(df)
