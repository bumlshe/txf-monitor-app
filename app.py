import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from datetime import datetime, date
import calendar

# ------------------------------------------------------------------------------
# 1. 手機端頁面配置
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="台指期 歸原指標監控",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    div[data-testid="stMetricValue"] {
        font-size: 22px;
    }
    .stButton>button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 2. 規格與結算日設定
# ------------------------------------------------------------------------------
SPECS = {
    '小台': {'point_val': 50},
    '大台': {'point_val': 200},
    '微台': {'point_val': 10}
}

def get_settlement_date(year, month):
    c = calendar.monthcalendar(year, month)
    wednesdays = [week[calendar.WEDNESDAY] for week in c if week[calendar.WEDNESDAY] != 0]
    return date(year, month, wednesdays[2])

def is_settlement_day(target_date):
    d = target_date.date() if isinstance(target_date, (pd.Timestamp, datetime)) else target_date
    return d == get_settlement_date(d.year, d.month)

# ------------------------------------------------------------------------------
# 3. 歸原指標與回測引擎
# ------------------------------------------------------------------------------
def calculate_indicators(df, n1=1, n2=1, n3=5, xb=-36, xs=-14, ma_period=60):
    n = len(df)
    highs = df['high'].values
    lows = df['low'].values
    
    h_base = pd.Series(highs).rolling(window=n3, min_periods=1).mean().values
    l_base = pd.Series(lows).rolling(window=n3, min_periods=1).mean().values
    
    alpha1 = 2.0 / (n1 + 1.0)
    alpha2 = 2.0 / (n2 + 1.0)
    
    upper_arr = np.zeros(n)
    lower_arr = np.zeros(n)
    upper_arr[0] = h_base[0]
    lower_arr[0] = l_base[0]
    
    for t in range(1, n):
        upper_arr[t] = upper_arr[t - 1] + alpha1 * (h_base[t] - upper_arr[t - 1])
        lower_arr[t] = lower_arr[t - 1] + alpha2 * (l_base[t] - lower_arr[t - 1])
        
    df = df.copy()
    df['res_line'] = upper_arr + xb
    df['sup_line'] = lower_arr + xs
    df['trend_ma'] = pd.Series(df['close'].values).rolling(window=ma_period, min_periods=1).mean().values
    return df

def run_strategy(df, contract_type, lots, n1, n2, n3, xb, xs, tp_pts, sl_pts, use_ma, ma_period):
    df = calculate_indicators(df, n1, n2, n3, xb, xs, ma_period)
    records = []
    trades = []

    multiplier = lots * SPECS[contract_type]['point_val']
    pst = 0
    order_price = 0.0
    tp_point = 0.0
    sl_point = 0.0
    cum_points = 0.0

    for i in range(len(df)):
        curr_row = df.iloc[i]
        curr_date = curr_row['time']
        curr_open = float(curr_row['open'])
        curr_high = float(curr_row['high'])
        curr_low = float(curr_row['low'])
        curr_close = float(curr_row['close'])
        curr_res = float(curr_row['res_line'])
        curr_sup = float(curr_row['sup_line'])
        curr_ma = float(curr_row['trend_ma'])

        is_settle = is_settlement_day(curr_date)
        action_desc = "持倉續抱" if pst != 0 else "觀望等待訊號"
        closed_by_tp_sl = False

        # 停損停利機制 (以盤中高低價與開盤價判定)
        if pst == 1:
            if tp_pts > 0 and (curr_open >= tp_point or curr_high >= tp_point):
                fill_p = curr_open if curr_open >= tp_point else tp_point
                p = fill_p - order_price
                cum_points += p
                trades.append({'time': curr_date, 'type': '多單停利', 'pnl_ntd': p * multiplier})
                pst = 0
                action_desc = f"【多單停利】達標 +{tp_pts} 點"
                closed_by_tp_sl = True
            elif sl_pts > 0 and (curr_open <= sl_point or curr_low <= sl_point):
                fill_p = curr_open if curr_open <= sl_point else sl_point
                p = fill_p - order_price
                cum_points += p
                trades.append({'time': curr_date, 'type': '多單停損', 'pnl_ntd': p * multiplier})
                pst = 0
                action_desc = f"【多單停損】觸及 -{sl_pts} 點"
                closed_by_tp_sl = True
        elif pst == -1:
            if tp_pts > 0 and (curr_open <= tp_point or curr_low <= tp_point):
                fill_p = curr_open if curr_open <= tp_point else tp_point
                p = order_price - fill_p
                cum_points += p
                trades.append({'time': curr_date, 'type': '空單停利', 'pnl_ntd': p * multiplier})
                pst = 0
                action_desc = f"【空單停利】達標 +{tp_pts} 點"
                closed_by_tp_sl = True
            elif sl_pts > 0 and (curr_open >= sl_point or curr_high >= sl_point):
                fill_p = curr_open if curr_open >= sl_point else sl_point
                p = order_price - fill_p
                cum_points += p
                trades.append({'time': curr_date, 'type': '空單停損', 'pnl_ntd': p * multiplier})
                pst = 0
                action_desc = f"【空單停損】觸及 -{sl_pts} 點"
                closed_by_tp_sl = True

        raw_buy = (curr_close > curr_res)
        raw_sell = (curr_close < curr_sup)
        cond_buy = raw_buy and (curr_close > curr_ma if use_ma else True)
        cond_sell = raw_sell and (curr_close < curr_ma if use_ma else True)

        if is_settle and pst != 0:
            p = (curr_close - order_price) if pst == 1 else (order_price - curr_close)
            cum_points += p
            trades.append({'time': curr_date, 'type': '結算平倉', 'pnl_ntd': p * multiplier})
            pst = 0
            action_desc = "結算日收盤平倉"
        elif not closed_by_tp_sl:
            if cond_buy and pst <= 0:
                if pst == -1:
                    cum_points += (order_price - curr_close)
                    trades.append({'time': curr_date, 'type': '翻多平空', 'pnl_ntd': (order_price - curr_close) * multiplier})
                pst = 1
                order_price = curr_close
                tp_point = order_price + tp_pts
                sl_point = order_price - sl_pts
                action_desc = "【突破多方】進多單"
            elif cond_sell and pst >= 0:
                if pst == 1:
                    cum_points += (curr_close - order_price)
                    trades.append({'time': curr_date, 'type': '翻空平多', 'pnl_ntd': (curr_close - order_price) * multiplier})
                pst = -1
                order_price = curr_close
                tp_point = order_price - tp_pts
                sl_point = order_price + sl_pts
                action_desc = "【跌破空方】進空單"

        records.append({
            'time': curr_date, 'open': curr_open, 'high': curr_high, 'low': curr_low, 'close': curr_close,
            'res_line': curr_res, 'sup_line': curr_sup, 'trend_ma': curr_ma,
            'position': pst, 'entry_price': order_price if pst != 0 else np.nan,
            'action_desc': action_desc, 'cum_pnl': cum_points * multiplier
        })

    return pd.DataFrame(records), pd.DataFrame(trades)

# ------------------------------------------------------------------------------
# 4. 資料來源 (FinMind 台指連續日線)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_wtx_data():
    url = "https://api.finmindtrade.com/api/v4/data"
    params = {
        "dataset": "TaiwanFuturesDaily",
        "data_id": "TX",
        "start_date": "2020-01-01"
    }
    try:
        res = requests.get(url, params=params, timeout=20)
        data = res.json()
        if data.get("msg") == "success":
            df = pd.DataFrame(data["data"])
            df['volume'] = df['volume'].astype(float)
            idx = df.groupby('date')['volume'].idxmax()
            df_c = df.loc[idx].copy()
            df_c = df_c.rename(columns={
                "date": "time", "open": "open", "max": "high",
                "min": "low", "close": "close", "volume": "volume"
            })
            df_c['time'] = pd.to_datetime(df_c['time'])
            return df_c.sort_values('time').reset_index(drop=True)
    except Exception:
        pass
    return None

# ------------------------------------------------------------------------------
# 5. UI 與參數設定
# ------------------------------------------------------------------------------
st.title("📊 台指期 歸原指標監控")

# 側邊欄控制
with st.sidebar:
    st.header("⚙️ 參數設定")
    contract = st.selectbox("合約規格", ["小台", "大台", "微台"], index=0, key="contract")
    lots = st.number_input("口數", min_value=1, max_value=20, value=1, key="lots")
    
    st.subheader("歸原通道參數")
    col_n1, col_n2 = st.columns(2)
    n1 = col_n1.number_input("N1", value=1, key="n1")
    n2 = col_n2.number_input("N2", value=1, key="n2")
    n3 = st.number_input("N3", value=5, key="n3")
    xb = st.number_input("XB (多方閥值)", value=-36, key="xb")
    xs = st.number_input("XS (空方閥值)", value=-14, key="xs")

    st.subheader("風控與濾網")
    tp_pts = st.number_input("停利點數 (0表關閉)", value=400, step=10, key="tp_pts")
    sl_pts = st.number_input("停損點數 (0表關閉)", value=250, step=10, key="sl_pts")
    use_ma = st.checkbox("啟用 60MA 順勢濾網", value=True, key="use_ma")
    ma_period = st.number_input("MA 週期", value=60, key="ma_period")
    
    st.subheader("顯示設定")
    show_bars = st.slider("K線顯示根數", min_value=15, max_value=60, value=25, key="show_bars")

col_btn1, col_btn2 = st.columns(2)
with col_btn1:
    if st.button("🔄 重新計算策略", use_container_width=True):
        st.rerun()
with col_btn2:
    if st.button("📥 抓取最新 FinMind 數據", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

df_raw = fetch_wtx_data()

if df_raw is not None and len(df_raw) > 0:
    # 策略運算
    df_calc, df_trades = run_strategy(
        df_raw, contract, lots, n1, n2, n3, xb, xs, tp_pts, sl_pts, use_ma, ma_period
    )
    last_row = df_calc.iloc[-1]
    
    # 頂部狀態指標
    pos = int(last_row['position'])
    pos_desc = "空手觀望" if pos == 0 else ("多單持有 🟢" if pos == 1 else "空單持有 🔴")
    total_pnl = df_trades['pnl_ntd'].sum() if len(df_trades) > 0 else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("部位狀態", pos_desc, f"最新收盤: {last_row['close']:.0f}")
        st.metric("多方參考(橘)", f"{last_row['res_line']:.2f}")
    with col2:
        st.metric("累積損益", f"{total_pnl:+,.0f} 元", f"{len(df_trades)} 筆成交")
        st.metric("空方參考(藍)", f"{last_row['sup_line']:.2f}")

    is_settle = is_settlement_day(last_row['time'])
    settle_txt = " ⚠️【今日為台指結算日】" if is_settle else ""
    st.info(f"📅 **日期：{last_row['time'].strftime('%Y-%m-%d')}**{settle_txt}\n\n👉 **動作建議：{last_row['action_desc']}**")

    # K 線圖 (修正台股紅漲綠跌 + 依滑桿調整顯示根數)
    st.subheader("📈 通道與 K 線走勢")
    plot_df = df_calc.tail(show_bars).copy()
    
    fig = go.Figure()
    # 台股配色：收盤 > 開盤為紅線 (漲)，收盤 < 開盤為綠線 (跌)
    fig.add_trace(go.Candlestick(
        x=plot_df['time'].dt.strftime('%m/%d'),
        open=plot_df['open'], high=plot_df['high'],
        low=plot_df['low'], close=plot_df['close'], name='K線',
        increasing_line_color='#FF3333', increasing_fillcolor='#FF3333',
        decreasing_line_color='#00CC66', decreasing_fillcolor='#00CC66'
    ))
    fig.add_trace(go.Scatter(
        x=plot_df['time'].dt.strftime('%m/%d'), y=plot_df['res_line'],
        mode='lines', name='多方軌道', line=dict(color='#FFA726', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=plot_df['time'].dt.strftime('%m/%d'), y=plot_df['sup_line'],
        mode='lines', name='空方軌道', line=dict(color='#26C6DA', width=2)
    ))
    if use_ma:
        fig.add_trace(go.Scatter(
            x=plot_df['time'].dt.strftime('%m/%d'), y=plot_df['trend_ma'],
            mode='lines', name=f'{ma_period}MA', line=dict(color='#E040FB', dash='dot', width=1.5)
        ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0E121A',
        plot_bgcolor='#0E121A',
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_rangeslider_visible=False,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📑 查看平倉歷史明細"):
        if len(df_trades) > 0:
            show_trades = df_trades.tail(15).iloc[::-1].copy()
            show_trades['time'] = show_trades['time'].dt.strftime('%Y-%m-%d')
            show_trades['pnl_ntd'] = show_trades['pnl_ntd'].apply(lambda x: f"{x:+,.0f}")
            show_trades.columns = ['出場日期', '出場動作', '獲利(元)']
            st.dataframe(show_trades, use_container_width=True, hide_index=True)
        else:
            st.write("目前區間尚無平倉交易。")
else:
    st.error("暫時無法取得 FinMind API 數據，請檢查網路連線後點擊上方按鈕重試。")