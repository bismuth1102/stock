import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. 策略参数配置 (在这里调整你的模型参数)
# ==============================================================================
STRATEGY_CONFIG = {
    # 标的设置
    'code': 'sh518880',
    'name': '黄金ETF',
    
    # 核心策略参数 (根据之前的最优解填写)
    'trailing_atr':  2,   # 止盈宽松度
    'buy_conf':      0.6,  # 买入信心门槛
    'target_up':     1.2,   # [训练] 贪婪度
    'risk_trig':     0.75,   # [训练] 胆量/风控敏感度
    
    # 固定参数
    'stop_down':     1.0,   # [固定] 容忍度
    'lookback_days': 600,   # 数据回溯长度
    'max_bullets':   3      # 总子弹数 (三发模式)
}

# ==============================================================================
# 2. 当前持仓状态 (请务必诚实填写，否则建议不准！)
# ==============================================================================
MY_PORTFOLIO = {
    'total_capital':   30000.0,  # 总投入本金 (比如10万)
    'current_cash':    30000.0,   # 当前账户里的可用现金
    'hold_shares':     900,     # 当前持仓股数 (如果没有填0)
    'avg_cost':        10.883,     # 当前持仓成本价 (如果没有填0)
    'highest_price':   11.288,     # 持仓期间见过的最高价 (用于移动止盈，若刚买填当前价)
    'units_used':      1          # 已使用的子弹数 (0, 1, 2, 3)
}

# 交易费率设置
COMMISSION = 0.00015
MIN_COMM = 5.0

# ==============================================================================
# 3. 核心逻辑 (无需修改)
# ==============================================================================

def get_data(code, lookback):
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{lookback},qfq"
    try:
        resp = requests.get(url, timeout=3).json()
        raw = resp['data'][code].get('qfqday', resp['data'][code].get('day', []))
        df = pd.DataFrame(raw).iloc[:, :6]
        df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
        for c in df.columns[1:]: df[c] = pd.to_numeric(df[c])
        df['日期'] = pd.to_datetime(df['日期'])
        return df
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None

def calc_indicators(df):
    # 基础指标
    df['MA20'] = df['收盘'].rolling(20).mean()
    df['Trend_OK'] = df['收盘'] > df['MA20']
    
    # ATR
    h_l = df['最高'] - df['最低']
    h_c = (df['最高'] - df['收盘'].shift()).abs()
    l_c = (df['最低'] - df['收盘'].shift()).abs()
    df['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    df['NATR'] = df['ATR'] / df['收盘']
    
    # AI特征
    df['Norm_MACD'] = (df['收盘'].ewm(span=12).mean() - df['收盘'].ewm(span=26).mean()).ewm(span=9).mean() * 2 / df['收盘']
    delta = df['收盘'].diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    df['Bias'] = (df['收盘'] - df['MA20']) / df['MA20']
    df['Vol_Ratio'] = df['成交量'] / df['成交量'].rolling(5).mean()
    
    # 增强特征
    bb_std = df['收盘'].rolling(20).std()
    df['BB_Pos'] = (df['收盘'] - (df['MA20']-2*bb_std)) / (4*bb_std)
    
    low9 = df['最低'].rolling(9).min()
    high9 = df['最高'].rolling(9).max()
    rsv = (df['收盘'] - low9) / (high9 - low9) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['KDJ_J'] = 3 * df['K'] - 2 * df['K'].ewm(com=2).mean()
    
    df['ROC'] = df['收盘'].pct_change(10) * 100
    obv = np.where(df['收盘']>df['收盘'].shift(), df['成交量'], 
          np.where(df['收盘']<df['收盘'].shift(), -df['成交量'], 0))
    df['OBV_Slope'] = pd.Series(obv).cumsum().pct_change(5)
    
    return df.dropna()

def train_and_predict(df, cfg):
    # 准备训练数据
    feature_cols = ['Norm_MACD', 'RSI', 'Bias', 'Vol_Ratio', 'NATR', 'BB_Pos', 'KDJ_J', 'ROC', 'OBV_Slope']
    
    # 这里的逻辑是：拿过去所有数据训练，然后预测“最后一行”的状态
    train_df = df.iloc[:-1].copy() # 排除最后一天作为训练集
    last_row = df.iloc[[-1]].copy() # 最后一天用来预测
    
    X_train = train_df[feature_cols].iloc[:-5] # 标签需要未来5天，所以特征要再切掉5天
    
    # 构建标签
    closes = train_df['收盘'].values
    atrs = train_df['ATR'].values
    highs = train_df['最高'].values
    lows = train_df['最低'].values
    
    buy_y, sell_y = [], []
    v_len = len(train_df) - 5
    
    for k in range(v_len):
        c, a = closes[k], atrs[k]
        # 使用配置参数打标签
        t_up = c + a * cfg['target_up']
        s_down = c - a * cfg['stop_down']
        risk_trig = c - a * cfg['risk_trig']
        
        # 买入标签
        is_buy = 0
        if np.max(highs[k+1:k+6]) >= t_up: is_buy = 1
        elif np.min(lows[k+1:k+6]) <= s_down: is_buy = 0
        buy_y.append(is_buy)
        
        # 卖出/风险标签
        is_risk = 1 if np.min(lows[k+1:k+6]) <= risk_trig else 0
        sell_y.append(is_risk)
        
    # 训练
    m_buy = HistGradientBoostingClassifier(max_depth=4).fit(X_train, buy_y)
    m_sell = HistGradientBoostingClassifier(max_depth=4).fit(X_train, sell_y)
    
    # 预测最新一天
    buy_prob = m_buy.predict_proba(last_row[feature_cols])[:, 1][0]
    sell_prob = m_sell.predict_proba(last_row[feature_cols])[:, 1][0]
    
    return buy_prob, sell_prob

def make_decision():
    cfg = STRATEGY_CONFIG
    pf = MY_PORTFOLIO
    
    print(f"🚀 正在分析 {cfg['name']} ({cfg['code']})...")
    df = get_data(cfg['code'], cfg['lookback_days'])
    if df is None: return
    
    df = calc_indicators(df)
    last_row = df.iloc[-1]
    last_date = last_row['日期'].date()
    
    print(f"📅 最新数据日期: {last_date}")
    print(f"   收盘价: {last_row['收盘']:.3f} | MA20: {last_row['MA20']:.3f} | ATR: {last_row['ATR']:.3f}")
    
    # 1. 获取AI预测
    buy_prob, sell_prob = train_and_predict(df, cfg)
    print(f"🤖 AI预测: 买入信心 {buy_prob:.2f} | 风险概率 {sell_prob:.2f}")
    
    # 2. 决策逻辑
    action = "观望"
    reason = ""
    amount = 0
    shares = 0
    
    close = last_row['收盘']
    atr = last_row['ATR']
    trend_ok = last_row['Trend_OK']
    
    # --- 卖出检查 ---
    if pf['hold_shares'] > 0:
        pnl_pct = (close - pf['avg_cost']) / pf['avg_cost']
        # 计算动态止盈价
        trailing_price = pf['highest_price'] - (atr * cfg['trailing_atr'])
        
        sell_trigger = False
        
        if pnl_pct <= -0.10:
            sell_trigger = True; reason = "硬止损(-10%)清仓"
        elif pnl_pct <= -0.05:
            sell_trigger = True; reason = "弱止损(-5%)退弹"
        elif (not trend_ok) or (sell_prob > 0.85): # AI高危
            sell_trigger = True; reason = f"风控撤退 (趋势:{trend_ok}, AI险:{sell_prob:.2f})"
        elif (pf['highest_price'] > pf['avg_cost'] * 1.08) and (last_row['最低'] <= trailing_price):
            sell_trigger = True; reason = f"移动止盈 (破{trailing_price:.3f})"
            
        if sell_trigger:
            action = "卖出"
            # 卖出一发子弹的量
            if "清仓" in reason:
                shares = pf['hold_shares']
            else:
                if pf['units_used'] > 0:
                    shares = int(pf['hold_shares'] / pf['units_used'] / 100) * 100
                else:
                    shares = pf['hold_shares']
            
            if shares == 0: shares = pf['hold_shares'] # 防止碎股
            amount = shares * close
            
            print("\n" + "="*40)
            print(f"📢 建议操作: 【{action}】")
            print(f"📉 卖出数量: {shares} 股")
            print(f"💰 预计回笼: {amount:.2f} 元")
            print(f"💡 理由: {reason}")
            print("="*40)
            return

    # --- 买入检查 ---
    if pf['units_used'] < cfg['max_bullets']:
        buy_signal = False
        
        # 风控检查
        risk_pass = (sell_prob < 0.85)
        
        if pf['units_used'] == 0:
            # 首仓
            if trend_ok and (buy_prob > cfg['buy_conf']) and risk_pass:
                buy_signal = True; reason = "首仓进场"
        else:
            # 加仓 (必须浮盈)
            if trend_ok and (buy_prob > cfg['buy_conf']) and risk_pass and (close > pf['avg_cost']):
                buy_signal = True; reason = f"加仓 (第{pf['units_used']+1}发)"
            elif close <= pf['avg_cost']:
                reason = "未满足浮盈加仓条件 (当前亏损中)"
        
        if buy_signal:
            action = "买入"
            # 计算买入金额：总本金 / 总子弹数
            target_amount_per_bullet = pf['total_capital'] / cfg['max_bullets']
            money_to_use = min(target_amount_per_bullet, pf['current_cash'])
            
            fee = max(money_to_use * COMMISSION, MIN_COMM)
            if money_to_use > fee + 100:
                shares = int((money_to_use - fee) / close / 100) * 100
                amount = shares * close
                
                print("\n" + "="*40)
                print(f"📢 建议操作: 【{action}】")
                print(f"📈 买入数量: {shares} 股")
                print(f"💸 动用资金: {amount:.2f} 元")
                print(f"💡 理由: {reason} (AI信心:{buy_prob:.2f})")
                print("="*40)
                return
            else:
                print(f"\n⚠️ 信号触发，但现金不足以买入一手。")
                return

    # --- 无操作 ---
    print("\n" + "="*40)
    print(f"🧘 建议操作: 【观望 / 持股】")
    if pf['hold_shares'] > 0:
        trailing_price = pf['highest_price'] - (atr * cfg['trailing_atr'])
        print(f"🛡️ 当前止盈保护线: {trailing_price:.3f}")
        print(f"📊 浮动盈亏: {(close - pf['avg_cost'])/pf['avg_cost']*100:.2f}%")
    else:
        print(f"💤 空仓等待机会 (AI买入分:{buy_prob:.2f} < 门槛{cfg['buy_conf']})")
    print("="*40)

if __name__ == "__main__":
    make_decision()
