import requests
import pandas as pd
import numpy as np
import mplfinance as mpf
from sklearn.ensemble import HistGradientBoostingClassifier
import itertools
import warnings
import time

warnings.filterwarnings('ignore')

# ==============================================================================
# 🎯 调优目标与搜索空间
# ==============================================================================

TARGET_PROFIT = 60.0  # 目标收益率

# 固定参数
FIXED_LOOKBACK = 400
FIXED_TRAIN_MIN = 100

PARAM_GRID = {
    # --- 交易执行层参数 ---
    'trailing':   [1.5, 2, 2.5],       # 止盈宽松度
    'buy_conf':   [0.6, 0.65],    # 买入信心门槛
    'target_up':  [1.2, 1.6, 2],       # 贪婪度
    'stop_down':  [0.8, 1, 1.2],                  # 容忍度
    'risk_trig':  [0.6, 0.75, 0.9]        # 胆量
}

# ==============================================================================
# 🛠️ GLOBAL SETTINGS (基础固定配置)
# ==============================================================================
INITIAL_CAPITAL = 100000.0   
COMMISSION_RATE = 0.00015    
MIN_COMMISSION  = 5.0        
START_DATE      = "2025-08-01" 
MAX_BULLETS     = 3      

# 卖出风控
HARD_STOP_LOSS  = -0.10      
SOFT_STOP_LOSS  = -0.05      
TRAILING_START  = 0.08       
AI_RISK_ALERT   = 0.85       

# AI固定参数
TRAIN_WINDOW    = 5   
LOOK_BACK_WINDOW = 80      

# 技术指标参数
ATR_PERIOD      = 14         
RSI_PERIOD      = 14         
MACD_FAST       = 12         
MACD_SLOW       = 26         
MACD_SIGNAL     = 9          
VOL_MA          = 5          

# ==============================================================================
# 📊 目标 ETF
# ==============================================================================
TARGET_CODE = "sh518880"
TARGET_NAME = "黄金ETF"
BASE_MA_PERIOD = 20  

# ==============================================================================
# 📉 核心逻辑
# ==============================================================================

def get_data_tencent(code, lookback_days):
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{lookback_days},qfq"
    try:
        resp = requests.get(url, timeout=5).json()
        raw = resp['data'][code].get('qfqday', resp['data'][code].get('day', []))
        if not raw: return None
        df = pd.DataFrame(raw).iloc[:, :6]
        df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
        cols = ['开盘', '收盘', '最高', '最低', '成交量']
        for c in cols: df[c] = pd.to_numeric(df[c])
        df['日期'] = pd.to_datetime(df['日期'])
        return df
    except: return None

def calculate_features(df, ma_period):
    df.sort_values('日期', inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    df['MA_Trend'] = df['收盘'].rolling(ma_period).mean()
    df['Trend_OK'] = df['收盘'] > df['MA_Trend']
    
    high_low = df['最高'] - df['最低']
    high_close = np.abs(df['最高'] - df['收盘'].shift())
    low_close = np.abs(df['最低'] - df['收盘'].shift())
    df['ATR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(ATR_PERIOD).mean()
    df['NATR'] = df['ATR'] / df['收盘']
    df['Bias'] = (df['收盘'] - df['MA_Trend']) / df['MA_Trend']
    
    exp1 = df['收盘'].ewm(span=MACD_FAST, adjust=False).mean()
    exp2 = df['收盘'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['DIF'] = exp1 - exp2
    df['DEA'] = df['DIF'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['Norm_MACD'] = (df['DIF'] - df['DEA']) * 2 / df['收盘']
    
    delta = df['收盘'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['Vol_Ratio'] = df['成交量'] / df['成交量'].rolling(VOL_MA).mean()
    
    bb_std = df['收盘'].rolling(20).std()
    bb_up, bb_low = df['MA_Trend'] + 2 * bb_std, df['MA_Trend'] - 2 * bb_std
    df['BB_Pos'] = (df['收盘'] - bb_low) / (bb_up - bb_low)
    
    low_list = df['最低'].rolling(9).min()
    high_list = df['最高'].rolling(9).max()
    rsv = (df['收盘'] - low_list) / (high_list - low_list) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['KDJ_J'] = 3 * df['K'] - 2 * df['D']
    
    df['ROC'] = df['收盘'].pct_change(periods=10) * 100
    obv_val = np.where(df['收盘'] > df['收盘'].shift(1), df['成交量'], 
              np.where(df['收盘'] < df['收盘'].shift(1), -df['成交量'], 0))
    df['OBV_Slope'] = pd.Series(obv_val).cumsum().pct_change(periods=5)

    return df.dropna()

def plot_candlestick(df, trade_logs, title, params):
    plot_data = df.copy()
    plot_data['日期'] = pd.to_datetime(plot_data['日期'])
    plot_data.set_index('日期', inplace=True)
    plot_data.rename(columns={'开盘': 'Open', '最高': 'High', '最低': 'Low', '收盘': 'Close', '成交量': 'Volume'}, inplace=True)
    
    buy_signals = [np.nan] * len(plot_data)
    sell_signals = [np.nan] * len(plot_data)
    
    for t in trade_logs:
        date = pd.to_datetime(t['日期'])
        if date in plot_data.index:
            loc = plot_data.index.get_loc(date)
            if '买入' in t['操作']:
                buy_signals[loc] = plot_data.iloc[loc]['Low'] * 0.98
            elif '卖出' in t['操作']:
                sell_signals[loc] = plot_data.iloc[loc]['High'] * 1.02

    ap_ma = mpf.make_addplot(plot_data['Close'].rolling(20).mean(), color='orange', width=1.5)
    ap_buy = mpf.make_addplot(buy_signals, type='scatter', markersize=80, marker='^', color='red')
    ap_sell = mpf.make_addplot(sell_signals, type='scatter', markersize=80, marker='v', color='green')
    
    sub_title = f"\nBest Params: TR={params['trailing']} | UP={params['target_up']} | STOP={params['stop_down']} | RISK={params['risk_trig']}"
    
    print(f"   🖼️ 正在绘制最佳结果图表...")
    mpf.plot(plot_data, type='candle', style='yahoo', 
             title=f"{title} AI Tuning Result" + sub_title,
             addplot=[ap_ma, ap_buy, ap_sell], volume=True, figratio=(12, 6), tight_layout=True)

# ==============================================================================
# 🏎️ 极速回测内核
# ==============================================================================
def run_backtest_silent(code, full_df, params):
    p_trailing  = params['trailing']
    p_buy_conf  = params['buy_conf']
    p_target_up = params['target_up']
    p_stop_down = params['stop_down']
    p_risk_trig = params['risk_trig']
    
    backtest_data = full_df[full_df['日期'] >= START_DATE].copy()
    if len(backtest_data) == 0: return -100, []

    # 训练
    feature_cols = ['Norm_MACD', 'RSI', 'Bias', 'Vol_Ratio', 'NATR', 'BB_Pos', 'KDJ_J', 'ROC', 'OBV_Slope']
    start_idx = backtest_data.index[0]
    total_len = len(full_df)
    full_df['AI_Buy_Prob'] = 0.0
    full_df['AI_Sell_Prob'] = 0.0
    
    for i in range(start_idx, total_len, TRAIN_WINDOW):
        if i < FIXED_TRAIN_MIN: continue
        start_train_index = max(0, i - LOOK_BACK_WINDOW)
        train_df = full_df.iloc[start_train_index:i] 
        
        X_train = train_df[feature_cols].iloc[:-5]
        closes = train_df['收盘'].values
        atrs = train_df['ATR'].values
        highs = train_df['最高'].values
        lows = train_df['最低'].values
        v_len = len(train_df) - 5
        
        buy_y, sell_y = [], []
        for k in range(v_len):
            c, a = closes[k], atrs[k]
            t_up = c + a * p_target_up     
            s_down = c - a * p_stop_down   
            risk_line = c - a * p_risk_trig 
            
            is_buy = 0
            if np.max(highs[k+1:k+6]) >= t_up: is_buy = 1 
            elif np.min(lows[k+1:k+6]) <= s_down: is_buy = 0
            buy_y.append(is_buy)
            
            is_risk = 1 if np.min(lows[k+1:k+6]) <= risk_line else 0
            sell_y.append(is_risk)
            
        if len(X_train) < 50: continue
        
        m_buy = HistGradientBoostingClassifier(max_depth=4, random_state=42).fit(X_train, buy_y)
        m_sell = HistGradientBoostingClassifier(max_depth=4, random_state=42).fit(X_train, sell_y)
        
        end_p = min(i + TRAIN_WINDOW, total_len)
        X_pred = full_df[feature_cols].iloc[i:end_p]
        if len(X_pred) > 0:
            full_df.loc[X_pred.index, 'AI_Buy_Prob'] = m_buy.predict_proba(X_pred)[:, 1]
            full_df.loc[X_pred.index, 'AI_Sell_Prob'] = m_sell.predict_proba(X_pred)[:, 1]

    # 模拟
    cash = INITIAL_CAPITAL
    hold_shares = 0
    avg_cost = 0 
    max_price_since_entry = 0 
    current_units = 0 
    trade_logs = []
    
    sim_data = full_df[full_df['日期'] >= START_DATE].copy()
    
    for idx, row in sim_data.iterrows():
        close = row['收盘']
        atr = row['ATR']
        trend_ok = row['Trend_OK']
        prob_buy = row['AI_Buy_Prob']
        prob_sell = row['AI_Sell_Prob']
        
        if current_units > 0 and hold_shares > 0:
            max_price_since_entry = max(max_price_since_entry, row['最高'])
            current_pnl_pct = (close - avg_cost) / avg_cost
            trailing_stop_price = max_price_since_entry - (atr * p_trailing)
            
            sell_trigger = False
            is_clearance = False
            sell_reason = ""
            
            if current_pnl_pct <= HARD_STOP_LOSS: sell_trigger=True; is_clearance=True; sell_reason="硬止损"
            elif current_pnl_pct <= SOFT_STOP_LOSS: sell_trigger=True; sell_reason="弱止损"
            elif (not trend_ok) or (prob_sell > AI_RISK_ALERT): sell_trigger=True; sell_reason="风控"
            elif max_price_since_entry > avg_cost * (1+TRAILING_START) and row['最低'] <= trailing_stop_price:
                sell_trigger=True; sell_reason="移动止盈"
            
            if sell_trigger:
                shares_to_sell = hold_shares if is_clearance else int(hold_shares/current_units/100)*100
                if shares_to_sell == 0: shares_to_sell = hold_shares
                if shares_to_sell > 0:
                    fee = max(shares_to_sell * close * COMMISSION_RATE, MIN_COMMISSION)
                    cash += (shares_to_sell * close) - fee
                    pnl = (close - avg_cost) / avg_cost * 100
                    # 🔥 修改点：增加买信和卖信记录
                    trade_logs.append({
                        "日期":str(row['日期'].date()), 
                        "操作":"卖出", 
                        "价格":close, 
                        "盈亏":pnl, 
                        "说明":sell_reason,
                        "买信": prob_buy,
                        "卖信": prob_sell
                    })
                    hold_shares -= shares_to_sell
                    if is_clearance: current_units = 0
                    else: current_units -= 1
                    if hold_shares == 0: avg_cost=0; max_price_since_entry=0; current_units=0
                    continue

        if current_units < MAX_BULLETS:
            is_buy = False
            risk_pass = (prob_sell < AI_RISK_ALERT)
            buy_note = ""
            if current_units == 0:
                if trend_ok and prob_buy > p_buy_conf and risk_pass: is_buy=True; buy_note="首仓"
            else:
                if trend_ok and prob_buy > p_buy_conf and risk_pass and (close > avg_cost): is_buy=True; buy_note="加仓"
            
            if is_buy:
                money_use = min(INITIAL_CAPITAL/MAX_BULLETS, cash)
                fee_est = max(money_use * COMMISSION_RATE, MIN_COMMISSION)
                if money_use > fee_est + 100:
                    shares = int((money_use - fee_est)/close/100)*100
                    if shares > 0:
                        cost = shares * close
                        fee = max(cost * COMMISSION_RATE, MIN_COMMISSION)
                        if hold_shares==0: new_avg=close
                        else: new_avg=((hold_shares*avg_cost)+cost)/(hold_shares+shares)
                        cash -= (cost + fee)
                        hold_shares += shares
                        avg_cost = new_avg
                        current_units += 1
                        max_price_since_entry = max(max_price_since_entry, close)
                        # 🔥 修改点：增加买信和卖信记录
                        trade_logs.append({
                            "日期":str(row['日期'].date()), 
                            "操作":"买入", 
                            "价格":close, 
                            "盈亏":0, 
                            "说明":buy_note,
                            "买信": prob_buy,
                            "卖信": prob_sell
                        })

    final_asset = cash + (hold_shares * sim_data.iloc[-1]['收盘'])
    total_ret = (final_asset - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    return total_ret, trade_logs, sim_data

# ==============================================================================
# 🧠 自动调优主控
# ==============================================================================
def auto_optimize():
    print(f"🚀 开始AI性格特训：目标收益率 > {TARGET_PROFIT}%")
    print("=" * 75)
    
    print(f"📥 拉取公共数据 (Lookback={FIXED_LOOKBACK})...")
    df_raw = get_data_tencent(TARGET_CODE, FIXED_LOOKBACK)
    if df_raw is None: 
        print("数据获取失败"); return
    df_common = calculate_features(df_raw, BASE_MA_PERIOD)
    
    # 生成组合
    keys, values = zip(*PARAM_GRID.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_combs = len(combinations)
    print(f"📦 共有 {total_combs} 种性格组合待测试...\n")
    
    best_ret = -999
    best_params = {}
    best_logs = []
    best_data = None
    
    for i, params in enumerate(combinations):
        p_str = (f"止盈:{params['trailing']} | 买信:{params['buy_conf']} | "
                 f"贪婪:{params['target_up']} | 容忍:{params['stop_down']} | 胆量:{params['risk_trig']}")
        
        print(f"   [{i+1}/{total_combs}] {p_str} ... ", end="")
        
        ret, logs, sim_data = run_backtest_silent(TARGET_CODE, df_common.copy(), params)
        print(f"收益: {ret:.2f}%")
        
        if ret > best_ret:
            best_ret = ret
            best_params = params
            best_logs = logs
            best_data = sim_data
            
        if ret >= TARGET_PROFIT:
            print("\n" + "🎉" * 20)
            print(f"✅ 找到神级参数！收益率达到 {ret:.2f}%")
            break
            
    print("\n" + "="*75)
    print(f"🏆 最佳 AI 性格参数 (收益 {best_ret:.2f}%)")
    print(f"   止盈ATR倍数:  {best_params['trailing']}")
    print(f"   买入信心阈值: {best_params['buy_conf']}")
    print(f"   [训练]贪婪度:   {best_params['target_up']}")
    print(f"   [训练]容忍度:   {best_params['stop_down']}")
    print(f"   [训练]胆量:     {best_params['risk_trig']}")
    print("="*75 + "\n")
    
    # 🔥 修改点：打印表头增加了 买信 和 卖信
    print(f"{'日期':<12} {'操作':<6} {'价格':<8} {'盈亏':<8} {'买信':<6} {'卖信':<6} {'说明'}")
    print("-" * 75)
    for t in best_logs:
        # 🔥 修改点：打印具体数值
        print(f"{t['日期']:<12} {t['操作']:<6} {t['价格']:<8.3f} {t['盈亏']:<8.2f}% {t['买信']:<6.2f} {t['卖信']:<6.2f} {t['说明']}")
        
    if len(best_logs) > 0 and best_data is not None:
        plot_candlestick(best_data, best_logs, TARGET_NAME, best_params)

if __name__ == "__main__":
    auto_optimize()
