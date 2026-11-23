"""分析引擎：技术面/基本面/资金面打分 + 综合建议。"""
import pandas as pd

from indicators import add_indicators, support_resistance

W_TECH, W_FUND, W_CAP = 40, 35, 25


def _find_col(df: pd.DataFrame, keywords: list[str]) -> str | None:
    for col in df.columns:
        if all(k in col for k in keywords):
            return col
    return None


def _latest_valid(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.iloc[0] if len(s) else None


# ---------------- 技术面 ----------------

def score_technical(kline: pd.DataFrame) -> dict:
    df = add_indicators(kline)
    last, prev = df.iloc[-1], df.iloc[-2]
    items = []

    # 1. 趋势（10分）
    s = 0
    if last["close"] > last["ma20"]:
        s += 5
    if last["ma20"] > last["ma60"]:
        s += 5
    trend_desc = ("多头排列，趋势向上" if s == 10 else
                  "趋势修复中" if s == 5 else "趋势偏弱，股价位于中期均线下方")
    items.append(("均线趋势", s, 10, trend_desc))

    # 2. MACD（8分）
    s = 0
    if last["dif"] > last["dea"]:
        s += 4
    if last["dif"] > 0:
        s += 2
    if last["macd"] > prev["macd"]:
        s += 2
    golden = last["dif"] > last["dea"] and prev["dif"] <= prev["dea"]
    items.append(("MACD", s, 8,
                  "刚形成金叉" if golden else
                  ("DIF在DEA上方，动能偏多" if last["dif"] > last["dea"] else "DIF在DEA下方，动能偏空")))

    # 3. RSI（8分）
    rsi = last["rsi"]
    if rsi < 30:
        s, d = 8, f"RSI={rsi:.0f}，超卖区，存在反弹机会"
    elif rsi < 40:
        s, d = 6, f"RSI={rsi:.0f}，偏弱但接近超卖"
    elif rsi <= 60:
        s, d = 4, f"RSI={rsi:.0f}，中性区间"
    elif rsi <= 70:
        s, d = 3, f"RSI={rsi:.0f}，偏强"
    else:
        s, d = 0, f"RSI={rsi:.0f}，超买区，短期追高风险大"
    items.append(("RSI", s, 8, d))

    # 4. KDJ（6分）
    golden_kdj = last["k"] > last["d"] and prev["k"] <= prev["d"]
    if golden_kdj:
        s, d = 6, "KDJ金叉"
    elif last["j"] < 0:
        s, d = 5, f"J={last['j']:.0f}，深度超卖"
    elif last["k"] > last["d"]:
        s, d = 4, "K在D上方，短线偏多"
    elif last["j"] > 100:
        s, d = 0, f"J={last['j']:.0f}，严重超买"
    else:
        s, d = 2, "K在D下方，短线偏空"
    items.append(("KDJ", s, 6, d))

    # 5. 布林带位置（4分）
    c = last["close"]
    if c < last["boll_low"]:
        s, d = 3, "跌破布林下轨，超跌但需防趋势破位"
    elif c < last["boll_mid"]:
        s, d = 4, "位于布林中轨与下轨之间，价格偏低"
    elif c < last["boll_up"]:
        s, d = 2, "位于布林中轨与上轨之间"
    else:
        s, d = 0, "突破布林上轨，短期过热"
    items.append(("布林带", s, 4, d))

    # 6. 量价配合（4分）
    vr, up = last["vol_ratio"], last["pct_chg"] > 0
    if up and vr > 1.2:
        s, d = 4, f"放量上涨（量比{vr:.1f}），资金积极"
    elif up:
        s, d = 2, "缩量上涨"
    elif not up and vr < 0.8:
        s, d = 2, "缩量回调，抛压有限"
    else:
        s, d = 0, f"放量下跌（量比{vr:.1f}），警惕出货"
    items.append(("量价配合", s, 4, d))

    total = sum(i[1] for i in items)
    sr = support_resistance(df)
    return {"score": total, "max": W_TECH, "items": items, "sr": sr, "df": df}


# ---------------- 基本面（分行业估值尺子） ----------------

STYLE_KEYWORDS = {
    "科技成长": ["半导体", "软件", "计算机", "通信", "电子", "互联网", "游戏", "元件",
                "光学", "航天", "航空", "军工", "仪器", "芯片", "消费电子", "传媒", "数字"],
    "金融": ["银行", "保险", "证券", "多元金融", "信托"],
    "周期": ["钢铁", "煤炭", "有色", "化工", "化纤", "化学", "水泥", "玻璃", "航运",
            "港口", "石油", "采掘", "贵金属", "能源金属", "造纸", "工程机械"],
}


def industry_style(industry: str | None) -> str:
    if industry:
        for style, kws in STYLE_KEYWORDS.items():
            if any(k in industry for k in kws):
                return style
    return "通用"


def score_fundamental(valuation: pd.DataFrame | None, financial: pd.DataFrame | None,
                      industry: str | None = None, trend_ok: bool = True) -> dict:
    """trend_ok=False（股价在60日线下方）时估值分位得分打5折——
    回测证明下跌趋势中低估值是价值陷阱（越便宜越跌）。"""
    style = industry_style(industry)
    items = []

    # ---- 提取估值指标及其历史分位 ----
    pe = pb = ps = peg = pe_pct = pb_pct = ps_pct = None
    if valuation is not None and len(valuation) > 60:
        def series(col):
            return pd.to_numeric(valuation[col], errors="coerce").dropna() if col in valuation.columns else pd.Series(dtype=float)

        pe_s, pb_s, ps_s = series("PE(TTM)"), series("市净率"), series("市销率")
        peg_s = series("PEG值")
        pe = pe_s.iloc[-1] if len(pe_s) else None
        pb = pb_s.iloc[-1] if len(pb_s) else None
        ps = ps_s.iloc[-1] if len(ps_s) else None
        peg = peg_s.iloc[-1] if len(peg_s) else None
        if pe is not None and pe > 0:
            pe_pct = (pe_s[pe_s > 0] < pe).mean() * 100
        if pb is not None and pb > 0:
            pb_pct = (pb_s < pb).mean() * 100
        if ps is not None and ps > 0:
            ps_pct = (ps_s[ps_s > 0] < ps).mean() * 100

    def pct_score(pct, full, label):
        if pct is None:
            return 0, f"{label}数据缺失或为负"
        if pct < 20:
            s, d = full, f"{label}处于历史{pct:.0f}%分位，明显低估"
        elif pct < 40:
            s, d = round(full * 0.7), f"{label}处于历史{pct:.0f}%分位，偏低"
        elif pct < 60:
            s, d = round(full * 0.5), f"{label}处于历史{pct:.0f}%分位，中等"
        elif pct < 80:
            s, d = round(full * 0.2), f"{label}处于历史{pct:.0f}%分位，偏高"
        else:
            s, d = 0, f"{label}处于历史{pct:.0f}%分位，明显高估"
        if not trend_ok and s > 0:
            s = round(s * 0.5)
            d += "；⚠️ 但股价处于下跌趋势（60日线下方），低估值可能是价值陷阱，得分打5折"
        return s, d

    # ---- 提取财务指标 ----
    roe = growth_np = growth_rev = None
    if financial is not None and len(financial):
        col = _find_col(financial, ["净资产收益率"])
        if col:
            roe = _latest_valid(financial[col])
        col = _find_col(financial, ["净利润增长率"])
        if col:
            growth_np = _latest_valid(financial[col])
        col = _find_col(financial, ["主营业务收入增长率"]) or _find_col(financial, ["收入增长率"])
        if col:
            growth_rev = _latest_valid(financial[col])

    def roe_item(full):
        if roe is None:
            return ("ROE", 0, full, "数据缺失")
        ladder = [(15, 1.0, "盈利能力优秀"), (10, 0.7, "盈利能力良好"),
                  (5, 0.4, "盈利能力一般"), (0, 0.2, "盈利能力偏弱")]
        for th, k, desc in ladder:
            if roe >= th and roe > 0:
                return (f"ROE（{roe:.1f}%）", round(full * k), full, desc)
        return (f"ROE（{roe:.1f}%）", 0, full, "亏损")

    def growth_item(full, rev_only=False):
        s, parts = 0, []
        pairs = [(growth_rev, "营收")] if rev_only else [(growth_np, "净利润"), (growth_rev, "营收")]
        half = full / len(pairs)
        for g, label in pairs:
            if g is None:
                parts.append(f"{label}增速缺失")
            elif g > 20:
                s += half
                parts.append(f"{label}增速{g:.0f}%（高增长）")
            elif g > 0:
                s += half / 2
                parts.append(f"{label}增速{g:.0f}%（正增长）")
            else:
                parts.append(f"{label}增速{g:.0f}%（负增长）")
        return ("成长性", round(s), full, "；".join(parts))

    def peg_item(full):
        if peg is None or peg <= 0:
            return ("PEG", 0, full, "PEG缺失或为负（利润下滑期，估值参考性弱）")
        if peg < 0.8:
            return (f"PEG（{peg:.2f}）", full, full, "成长性价比高（PEG<0.8）")
        if peg < 1.2:
            return (f"PEG（{peg:.2f}）", round(full * 0.7), full, "估值与成长匹配")
        if peg < 2:
            return (f"PEG（{peg:.2f}）", round(full * 0.3), full, "估值略贵于成长")
        return (f"PEG（{peg:.2f}）", 0, full, "估值明显透支成长")

    # ---- 按行业风格选用不同的尺子 ----
    if style == "科技成长":
        # 科技股 PE 常失真：改用 PS、PEG、营收增速为主
        s, d = pct_score(ps_pct, 8, "PS")
        items.append((f"PS估值（{ps:.1f}）" if ps else "PS估值", s, 8, d))
        items.append(peg_item(7))
        items.append(growth_item(10, rev_only=True))
        items.append(roe_item(5))
        s, d = pct_score(pe_pct, 5, "PE")
        items.append((f"PE参考（TTM={pe:.1f}）" if pe and pe > 0 else "PE参考（亏损/缺失）", s, 5, d))
    elif style == "金融":
        # 金融股：PB + ROE 是核心
        s, d = pct_score(pb_pct, 15, "PB")
        items.append((f"PB估值（{pb:.2f}）" if pb else "PB估值", s, 15, d))
        items.append(roe_item(12))
        items.append(growth_item(8))
    elif style == "周期":
        # 周期股：PB分位为主；PE低位常是周期顶部陷阱
        s, d = pct_score(pb_pct, 18, "PB")
        items.append((f"PB估值（{pb:.2f}）" if pb else "PB估值", s, 18, d))
        items.append(roe_item(9))
        items.append(growth_item(8))
        if pe_pct is not None and pe_pct < 20:
            items.append(("周期提示", 0, 0, f"⚠️ PE处于历史{pe_pct:.0f}%低分位——周期股低PE常对应盈利顶部，勿当成低估"))
    else:
        s, d = pct_score(pe_pct, 10, "PE")
        items.append((f"PE估值（TTM={pe:.1f}）" if pe else "PE估值", s, 10, d))
        s, d = pct_score(pb_pct, 7, "PB")
        items.append((f"PB估值（{pb:.2f}）" if pb else "PB估值", s, 7, d))
        items.append(roe_item(10))
        items.append(growth_item(8))

    return {"score": min(sum(i[1] for i in items), W_FUND), "max": W_FUND, "items": items,
            "pe": pe, "pb": pb, "ps": ps, "peg": peg,
            "pe_pct": pe_pct, "pb_pct": pb_pct, "ps_pct": ps_pct, "style": style}


# ---------------- 资金面 ----------------

def score_capital(fund_flow: pd.DataFrame | None) -> dict:
    items = []
    if fund_flow is None or len(fund_flow) < 20:
        return {"score": 0, "max": W_CAP, "items": [("资金流数据", 0, W_CAP, "数据缺失")]}

    main = pd.to_numeric(fund_flow["主力净流入-净额"], errors="coerce")
    ratio = pd.to_numeric(fund_flow["主力净流入-净占比"], errors="coerce")

    def flow_score(net_sum, days, full):
        yi = net_sum / 1e8
        if yi > 0:
            pos = (main.tail(days) > 0).mean()
            s = full if pos > 0.5 else round(full * 0.7)
            return s, f"近{days}日主力净流入{yi:.2f}亿（{pos*100:.0f}%交易日为净流入）"
        return 0, f"近{days}日主力净流出{abs(yi):.2f}亿"

    s, d = flow_score(main.tail(5).sum(), 5, 10)
    items.append(("短期主力资金（5日）", s, 10, d))
    s, d = flow_score(main.tail(20).sum(), 20, 8)
    items.append(("中期主力资金（20日）", s, 8, d))

    r = ratio.iloc[-1]
    if r > 5:
        s, d = 7, f"最新主力净占比{r:.1f}%，大幅流入"
    elif r > 0:
        s, d = 4, f"最新主力净占比{r:.1f}%，小幅流入"
    elif r > -5:
        s, d = 2, f"最新主力净占比{r:.1f}%，小幅流出"
    else:
        s, d = 0, f"最新主力净占比{r:.1f}%，大幅流出"
    items.append(("最新主力动向", s, 7, d))

    return {"score": sum(i[1] for i in items), "max": W_CAP, "items": items}


# ---------------- 综合 ----------------

def make_recommendation(tech: dict, fund: dict, cap: dict, kline: pd.DataFrame) -> dict:
    total = tech["score"] + fund["score"] + cap["score"]
    close = kline["close"].iloc[-1]
    sr = tech["sr"]
    ma20 = tech["df"]["ma20"].iloc[-1]

    if total >= 70:
        verdict, action = "建议买入", "多维度共振偏多，可考虑分批建仓"
    elif total >= 58:
        verdict, action = "谨慎买入", "信号偏多但不充分，建议小仓位试探、分批买入"
    elif total >= 42:
        verdict, action = "观望", "多空信号混杂，建议等待更明确的买点"
    else:
        verdict, action = "回避", "多维度偏空，不建议现在买入"

    buy_low = min(sr["support"], ma20)
    buy_high = max(sr["support"], ma20)
    return {
        "total": total, "verdict": verdict, "action": action,
        "close": close,
        "buy_zone": (round(buy_low, 2), round(buy_high, 2)),
        "stop_loss": round(sr["support"] * 0.95, 2),
        "resistance": sr["resistance"],
    }


def analyze(data: dict, industry: str | None = None) -> dict:
    tech = score_technical(data["kline"])
    tdf = tech["df"]
    trend_ok = bool(tdf["close"].iloc[-1] > tdf["ma60"].iloc[-1])
    fund = score_fundamental(data.get("valuation"), data.get("financial"), industry,
                             trend_ok=trend_ok)
    cap = score_capital(data.get("fund_flow"))
    rec = make_recommendation(tech, fund, cap, data["kline"])
    return {"tech": tech, "fund": fund, "cap": cap, "rec": rec}
