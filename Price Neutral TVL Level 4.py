import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"

# =========================================================
# 1. 读取数据
# =========================================================
tvl_path = "ethereum_tvl_2023-01-01_2025-01-01.csv"
price_path = "kline_ETHUSDT_D_20230101_20250101.csv"

tvl_df = pd.read_csv(tvl_path)
price_df = pd.read_csv(price_path)

tvl_df['date'] = pd.to_datetime(tvl_df['date']).dt.date
price_df['date'] = pd.to_datetime(price_df['datetime']).dt.date

price_df = price_df[['date', 'close']].rename(columns={'close': 'eth_price'})

df_base = pd.merge(tvl_df, price_df, on='date', how='inner')

# =========================================================
# 2. 构造指标
# =========================================================
df_base['price_neutral_tvl'] = df_base['tvl_usd'] / df_base['eth_price']
df_base['price_neutral_tvl_2dec'] = df_base['price_neutral_tvl'].round(2)

df_base['eth_return'] = df_base['eth_price'].pct_change()
df_base['pntvl_change'] = df_base['price_neutral_tvl_2dec'].pct_change()

df_base['divergence_strength'] = (
    df_base['eth_return'] - df_base['pntvl_change']
)

# =========================================================
# 3. 参数范围
# =========================================================
z_values = np.arange(0.6, 3.0, 0.1)     # Z-score 阈值
window_values = [30, 45, 60, 75, 90,120]    # 滑动窗口

heatmap = pd.DataFrame(
    index=window_values,
    columns=z_values
)

# =========================================================
# 4. 双参数扫描
# =========================================================
for window in window_values:

    df_base['div_mean'] = df_base['divergence_strength'].rolling(window).mean()
    df_base['div_std'] = df_base['divergence_strength'].rolling(window).std()
    df_base['divergence_z'] = (
        df_base['divergence_strength'] - df_base['div_mean']
    ) / df_base['div_std']

    for z in z_values:

        df = df_base.copy()
        df['signal'] = 0

        df.loc[
            (df['eth_return'] < 0) &
            (df['pntvl_change'] > 0) &
            (df['divergence_z'] < -z),
            'signal'
        ] = 1

        df.loc[
            (df['eth_return'] > 0) &
            (df['pntvl_change'] < 0) &
            (df['divergence_z'] > z),
            'signal'
        ] = -1

        # T+1 执行
        df['position'] = df['signal'].shift(1).fillna(0)

        # 策略收益
        df['strategy_return'] = df['position'] * df['eth_return']
        df['strategy_return'] = df['strategy_return'].fillna(0)

        # Sharpe
        if df['strategy_return'].std() == 0:
            sharpe = np.nan
        else:
            sharpe = (
                df['strategy_return'].mean() /
                df['strategy_return'].std()
            ) * np.sqrt(365)

        heatmap.loc[window, z] = sharpe

# =========================================================
# 5. Plotly 热力图
# =========================================================
fig = go.Figure(
    data=go.Heatmap(
        z=heatmap.values.astype(float),
        x=heatmap.columns.astype(str),
        y=heatmap.index.astype(str),
        colorscale='RdYlGn',
        colorbar=dict(title='Sharpe Ratio')
    )
)

fig.update_layout(
    title='Parameter Plateau Heatmap (Sharpe Ratio)',
    xaxis_title='Z-Score Threshold',
    yaxis_title='Rolling Window',
    width=1000,
    height=600
)

fig.show()
