import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
import json
import sys
from datetime import datetime, timedelta, timezone
from openai import OpenAI

warnings.filterwarnings('ignore')

# ==============================================================================
# ⚙️ 全局配置
# ==============================================================================
FIXED_TRAIN_WINDOW = 150  
GLOBAL_LOOKBACK_DAYS = 400  
GLOBAL_MAX_BULLETS = 3      # 资金最大份数 (限制加仓次数)
GLOBAL_STOP_DOWN = 1.0      # 训练集打标用的固定止损 (ATR倍数)

# ==============================================================================
# 🔑 API 配置
# ==============================================================================
DEEPSEEK_API_KEY = "sk-5e49822fcb8649d88c847667cb41642d" 
BASE_URL = "https://api.deepseek.com"
SERPER_API_KEY = "fa0849fb7ce91463287f65b7354caff449f3cab9"

# ==============================================================================
# 策略配置
# ==============================================================================
STRATEGIES = {
    'gt_ndx': {
        'code': 'sh513100',
        'name': '国泰纳斯达克ETF',
        'news_keywords': '纳斯达克 美股 科技股 人工智能',
        'trailing_atr': 3,
        'target_up': 1.6,
        'risk_trig': 0.9,
        'buy_conf': 0.6,
        'portfolio': {
            'hold_shares': 6900,
            'avg_cost': 1.429,
            'highest_price': 2.001,
            'units_used': 2
        }
    },
    'gold': {
        'code': 'sh518880',
        'name': '黄金ETF',
        'news_keywords': '黄金 美联储 降息 战争 避险',
        'trailing_atr': 2,
        'target_up': 1.2,
        'risk_trig': 0.9,
        'buy_conf': 0.6,
        'portfolio': {
            'hold_shares': 900,
            'avg_cost': 10.895,
            'highest_price': 11.977,
            'units_used': 1
        }
    },
     'ai': {
        'code': 'sz159819',
        'name': '人工智能ETF',
        'news_keywords': '人工智能 算力 英伟达 纳斯达克 科技股',
        'trailing_atr': 3,
        'target_up': 1.2,
        'risk_trig': 0.9,
        'buy_conf': 0.6,
        'portfolio': {
            'hold_shares': 0,
            'avg_cost': 0,
            'highest_price': 0,
            'units_used': 0
        }
    },
     'metal': {
        'code': 'sh560860',
        'name': '工业有色ETF',
        'trailing_atr': 3,
        'target_up': 1.2,
        'risk_trig': 1.5,
        'buy_conf': 0.55,
        'news_keywords': '铜价 铝价 大宗商品 制造业',
        'portfolio': {
            'hold_shares': 4400, 
            'avg_cost': 1.976,
            'highest_price': 2.228,
            'units_used': 2
        }
    }
}

COMMISSION = 0.00015
MIN_COMM = 5.0

# ==============================================================================
# 1. 数据处理模块
# ==============================================================================

def get_data(code, lookback):
    """从腾讯接口获取数据"""
    url = f"http://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={code},day,,,{lookback},qfq"
    try:
        resp = requests.get(url, timeout=3).json()
        raw = resp['data'][code].get('qfqday', resp['data'][code].get('day', []))
        df = pd.DataFrame(raw).iloc[:, :6]
        df.columns = ['日期', '开盘', '收盘', '最高', '最低', '成交量']
        for c in df.columns[1:]:
            df[c] = pd.to_numeric(df[c])
        df['日期'] = pd.to_datetime(df['日期'])
        return df
    except Exception as e:
        print(f"数据获取失败: {e}")
        return None

def calc_indicators(df):
    # 基础指标
    df['MA5'] = df['收盘'].rolling(5).mean()
    df['MA20'] = df['收盘'].rolling(20).mean()
    df['Trend_OK'] = df['MA5'] > df['MA20']
    
    # ATR
    h_l = df['最高'] - df['最低']
    h_c = (df['最高'] - df['收盘'].shift()).abs()
    l_c = (df['最低'] - df['收盘'].shift()).abs()
    df['ATR'] = pd.concat([h_l, h_c, l_c], axis=1).max(axis=1).rolling(14).mean()
    df['NATR'] = df['ATR'] / df['收盘']
    
    df['Norm_MACD'] = (df['收盘'].ewm(span=12).mean() - df['收盘'].ewm(span=26).mean()).ewm(span=9).mean() * 2 / df['收盘']
    delta = df['收盘'].diff()
    gain = (delta.where(delta>0, 0)).rolling(14).mean()
    loss = (-delta.where(delta<0, 0)).rolling(14).mean()
    df['RSI'] = 100 - (100 / (1 + gain/loss))
    df['Bias'] = (df['收盘'] - df['MA20']) / df['MA20']
    df['Vol_Ratio'] = df['成交量'] / df['成交量'].rolling(5).mean()
    
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

# ==============================================================================
# 2. 机器学习模块
# ==============================================================================

def train_and_predict(df, cfg, train_window):
    feature_cols = ['Norm_MACD', 'RSI', 'Bias', 'Vol_Ratio', 'NATR', 'BB_Pos', 'KDJ_J', 'ROC', 'OBV_Slope']
    
    last_row = df.iloc[[-1]].copy()
    train_df = df.iloc[:-1].copy()
    
    if train_window is not None and isinstance(train_window, int):
        if len(train_df) > train_window:
            train_df = train_df.iloc[-train_window:]
    
    X_train = train_df[feature_cols].iloc[:-5] 
    
    closes = train_df['收盘'].values
    atrs = train_df['ATR'].values
    highs = train_df['最高'].values
    lows = train_df['最低'].values
    
    buy_y, sell_y = [], []
    v_len = len(train_df) - 5
    
    for k in range(v_len):
        c, a = closes[k], atrs[k]
        t_up = c + a * cfg['target_up']
        s_down = c - a * GLOBAL_STOP_DOWN
        risk_trig = c - a * cfg['risk_trig']
        
        is_buy = 0
        if np.max(highs[k+1:k+6]) >= t_up:
            is_buy = 1
        elif np.min(lows[k+1:k+6]) <= s_down:
            is_buy = 0
        buy_y.append(is_buy)
        
        is_risk = 1 if np.min(lows[k+1:k+6]) <= risk_trig else 0
        sell_y.append(is_risk)
        
    m_buy = HistGradientBoostingClassifier(max_depth=4).fit(X_train, buy_y)
    m_sell = HistGradientBoostingClassifier(max_depth=4).fit(X_train, sell_y)
    
    buy_prob = m_buy.predict_proba(last_row[feature_cols])[:, 1][0]
    sell_prob = m_sell.predict_proba(last_row[feature_cols])[:, 1][0]
    
    return buy_prob, sell_prob

# ==============================================================================
# 3. 新闻侦探模块
# ==============================================================================

def get_sentiment_from_news(target_name, keywords, target_date=None):
    if target_date:
        simulated_now_str = f"{target_date} 23:59:59"
        search_query = f"{keywords} {target_date}"
        time_filter = None 
        print(f"🌍 正在回溯搜索【{target_name}】在 {target_date} 附近的新闻...")
    else:
        utc_now = datetime.now(timezone.utc)
        beijing_now = utc_now + timedelta(hours=8)
        simulated_now_str = beijing_now.strftime("%Y-%m-%d %H:%M")
        search_query = keywords
        time_filter = "qdr:d" 
        print(f"🌍 正在连接 Google News 搜索【{target_name}】最新新闻...")
    
    url = "https://google.serper.dev/news"
    payload_dict = { "q": search_query, "gl": "cn", "hl": "zh-cn", "num": 5 }
    if time_filter: payload_dict["tbs"] = time_filter
    payload = json.dumps(payload_dict)
    headers = { 'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json' }

    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        results = response.json().get('news', [])
        
        if not results:
            print("❌ 未搜索到相关新闻。")
            return None
        
        news_text = ""
        for i, res in enumerate(results):
            time_tag = res.get('date', '未知时间')
            news_text += f"[{i+1}] ({time_tag}) {res['title']}: {res.get('snippet', '')}\n"
            
    except Exception as e:
        print(f"❌ Serper 搜索出错: {e}")
        return None

    prompt = f"""
    你是一位资深量化交易员。
    【重要】：假设现在的时间是【{simulated_now_str}】。
    请忽略所有晚于此时间的知识，只根据提供的新闻摘要分析。
    
    关于【{target_name}】的新闻摘要：
    {news_text}
    
    任务：
    1. 提取每条新闻对该标的的利空/利多逻辑。
    2. 给出该标的短期（未来3-5天）的综合情绪评分。
    
    请严格输出 JSON：
    {{
        "news_details": [
            {{"time": "新闻时间", "summary": "摘要", "impact": "利多/利空/中性"}}
        ],
        "bullish_pct": 0-100 (整数),
        "final_reason": "一句话总结"
    }}
    """
    
    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "Output valid JSON only."},
                      {"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM 调用失败: {e}")
        return None

# ==============================================================================
# 4. 核心决策调度
# ==============================================================================

def make_decision(strategy_key, target_date_str=None, fetch_news=False):
    cfg = STRATEGIES[strategy_key]
    pf = cfg['portfolio']
    
    # --- 1. 获取全量数据 ---
    df_full = get_data(cfg['code'], GLOBAL_LOOKBACK_DAYS)
    if df_full is None: return
    df_full = calc_indicators(df_full)

    if target_date_str:
        target_dt = pd.to_datetime(target_date_str)
        df = df_full[df_full['日期'] <= target_dt].copy()
        if df.empty:
            print(f"❌ 错误：在 {target_date_str} 之前没有数据。")
            return
        real_last_date = df['日期'].iloc[-1]
        if real_last_date != target_dt:
            print(f"⚠️ 注意：目标日期 {target_date_str} 休市，使用前一交易日: {real_last_date.date()}")
    else:
        df = df_full.copy()
    
    last_row = df.iloc[-1]
    
    print("="*60)
    print(f"🚀 标的: {cfg['name']} ({cfg['code']}) | 日期: {last_row['日期'].date()}")
    
    # --- 2. 持仓状态分析 (使用 portfolio 参数) ---
    is_holding = pf['hold_shares'] > 0
    hold_info_str = "未持仓"
    
    if is_holding:
        curr_price = last_row['收盘']
        cost = pf['avg_cost']
        profit_pct = (curr_price - cost) / cost * 100
        market_val = pf['hold_shares'] * curr_price
        hold_info_str = f"持仓 {pf['hold_shares']}股 | 成本 {cost} | 浮动盈亏 {profit_pct:.2f}% | 仓位 {pf['units_used']}/{GLOBAL_MAX_BULLETS}"
    
    print(f"🎒 账户状态: {hold_info_str}")
    print(f"📊 市场数据: 收盘 {last_row['收盘']:.3f} | ATR {last_row['ATR']:.3f}")

    # --- 3. 检查移动止损 (使用 trailing_atr 参数) ---
    stop_signal = False
    stop_reason_text = ""
    new_highest_advice = 0
    
    if is_holding:
        # 逻辑：如果今天最高价比记录的最高价高，理论止损线上移
        record_high = pf['highest_price']
        today_high = last_row['最高']
        effective_high = max(record_high, today_high)
        
        # 计算移动止损线
        trailing_stop_price = effective_high - (last_row['ATR'] * cfg['trailing_atr'])
        
        print(f"🛡️ 移动止损: 历史最高 {record_high} -> 有效最高 {effective_high} | 止损线 {trailing_stop_price:.3f}")
        
        if last_row['最低'] < trailing_stop_price:
            stop_signal = True
            stop_reason_text = f"触发移动止损 (最低价 {last_row['最低']:.3f} 跌破 {trailing_stop_price:.3f})"
        
        new_highest_advice = effective_high

    # --- 4. 运行 AI 模型 ---
    print("\n🤖 启动 AI 分析...")
    tech_buy_prob, tech_sell_prob = train_and_predict(df, cfg, FIXED_TRAIN_WINDOW)
    
    sentiment = None
    if fetch_news:
        sentiment = get_sentiment_from_news(cfg['name'], cfg.get('news_keywords', cfg['name']), target_date_str)
    else:
        print("🔕 消息面: 已跳过 (未输入 news 参数，默认中性)")
    
    news_bullish = 50 
    news_reason = "无有效新闻数据"
    if sentiment:
        news_bullish = sentiment.get('bullish_pct', 50)
        news_reason = sentiment.get('final_reason', '未知')
        print("\n📰 新闻分析摘要:")
        for item in sentiment.get('news_details', []):
            flag = "🔴" if "利多" in item['impact'] else ("🟢" if "利空" in item['impact'] else "⚪")
            print(f"   {flag} {item.get('summary')}")

    final_score = (tech_buy_prob * 0.6) + ((news_bullish / 100) * 0.4)
    print(f"📈 评分: 买入分 {final_score:.2f} (阈值 {cfg['buy_conf']}) | 风险分 {tech_sell_prob:.2f}")

    print("\n" + "-" * 50)
    print(f"📈 技术面: 买入概率 {tech_buy_prob:.2f} | 风险概率 {tech_sell_prob:.2f} | MA5: {last_row['MA5']:.2f} vs MA20: {last_row['MA20']:.2f}")
    if sentiment:
        print(f"🌍 消息面: 看多程度 {news_bullish}% | 观点: {news_reason}")
    else:
        print(f"🌍 消息面: 50% (默认中性)")
        
    # --- 5. 生成最终建议 (整合所有逻辑) ---
    print("\n" + "-" * 30 + " 📝 决策建议 " + "-" * 30)
    
    # 优先级 1: 止损/风控
    sell_reason = ""
    do_sell = False
    
    if is_holding:
        if stop_signal:
            do_sell = True
            sell_reason = f"🛑 {stop_reason_text}"
        elif (tech_sell_prob > 0.85):
            do_sell = True
            sell_reason = f"⚠️ AI识别高风险 (Risk: {tech_sell_prob:.2f})"
        elif (tech_sell_prob > 0.75 and news_bullish < 30):
            do_sell = True
            sell_reason = f"📉 技术+消息共振利空"
            
    if do_sell:
        print(f"📢 操作: 【卖出 / 清仓】")
        print(f"💡 原因: {sell_reason}")
        return # 卖出后不再建议买入

    # 优先级 2: 买入/加仓 (使用 GLOBAL_MAX_BULLETS)
    do_buy = False
    buy_reason = ""
    
    risk_pass = (tech_sell_prob < 0.85)
    trend_ok = last_row['Trend_OK']
    
    if trend_ok and (final_score > cfg['buy_conf']) and risk_pass:
        if not is_holding:
            do_buy = True
            buy_reason = f"✅ 首仓信号 (分数 {final_score:.2f} > {cfg['buy_conf']})"
        elif pf['units_used'] < GLOBAL_MAX_BULLETS:
            do_buy = True
            buy_reason = f"➕ 加仓信号 (分数 {final_score:.2f} 且 仍有子弹)"
        else:
            print(f"🧘 操作: 【持有】 (信号触发但仓位已满 {GLOBAL_MAX_BULLETS}发)")
            if new_highest_advice > pf['highest_price']:
                print(f"📌 提示: 请更新配置中的 highest_price 为 {new_highest_advice:.3f} 以提高止损线")
            return

    if do_buy:
        print(f"📢 操作: 【{'建仓' if not is_holding else '加仓'}】")
        print(f"💡 原因: {buy_reason}")
        
        # 价格更新提示
        if is_holding and new_highest_advice > pf['highest_price']:
            print(f"📌 提示: 交易后请更新 highest_price 为 {new_highest_advice:.3f}")
        elif not is_holding:
            print(f"📌 提示: 交易后请更新 avg_cost 和 highest_price")
            
    else:
        # 既不卖也不买
        if is_holding:
            print(f"🧘 操作: 【持有】")
            if new_highest_advice > pf['highest_price']:
                print(f"📌 提示: 今日创新高，请更新 highest_price 为 {new_highest_advice:.3f}")
        else:
            print(f"👀 操作: 【观望】 (分数 {final_score:.2f} 未达标)")

# ==============================================================================
# 主程序入口
# ==============================================================================

if __name__ == "__main__":
    # 使用方式: 
    # 1. python strategy.py (不查新闻)
    # 2. python strategy.py news (查新闻)
    # 3. python strategy.py 2024-05-20 (不查新闻)
    # 4. python strategy.py 2024-05-20 news (查新闻)
    
    target_date_arg = None
    fetch_news_arg = False

    args = sys.argv[1:]
    for arg in args:
        if arg.lower() == 'news':
            fetch_news_arg = True
        else:
            try:
                pd.to_datetime(arg)
                target_date_arg = arg
            except:
                pass

    print(f"🖥️ 系统启动 | 目标日期: {target_date_arg if target_date_arg else 'Today'} | 消息面分析: {'✅ 开启' if fetch_news_arg else '❌ 关闭'}")
    
    for strategy_key in STRATEGIES.keys():
        make_decision(strategy_key, target_date_str=target_date_arg, fetch_news=fetch_news_arg)
        print("\n\n")
