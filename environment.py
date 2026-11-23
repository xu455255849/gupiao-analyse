"""环境层：大盘红绿灯、行业相对强度与资金流、财务红旗一票否决。"""
from datetime import datetime, timedelta

import akshare as ak
import pandas as pd
import streamlit as st

import data  # noqa: F401  确保代理环境变量已处理


# ================= 大盘红绿灯 =================

@st.cache_data(ttl=1800, show_spinner=False)
def _hs300_daily() -> pd.DataFrame:
    df = ak.index_zh_daily_sina(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"])
    return df.tail(320).reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def _market_activity() -> dict:
    df = ak.stock_market_activity_legu()
    return dict(zip(df["item"], df["value"]))


def get_market_light() -> dict:
    """大盘环境评估：绿灯（做多）/ 黄灯（谨慎）/ 红灯（防守）。"""
    items, score = [], 0

    try:
        idx = _hs300_daily()
        close = idx["close"].iloc[-1]
        ma250 = idx["close"].rolling(250).mean().iloc[-1]
        ma60 = idx["close"].rolling(60).mean().iloc[-1]
        if close > ma250:
            score += 2
            items.append(("✅", f"沪深300（{close:.0f}）在年线（{ma250:.0f}）上方，牛市格局未破"))
        else:
            items.append(("❌", f"沪深300（{close:.0f}）在年线（{ma250:.0f}）下方，熊市/弱势环境"))
        if close > ma60:
            score += 1
            items.append(("✅", f"指数在60日线（{ma60:.0f}）上方，中期趋势偏多"))
        else:
            items.append(("❌", f"指数在60日线（{ma60:.0f}）下方，中期趋势转弱"))

        vol20 = idx["volume"].rolling(20).mean().iloc[-1]
        vol60 = idx["volume"].rolling(60).mean().iloc[-1]
        if vol20 > vol60:
            score += 1
            items.append(("✅", "近20日量能高于60日均量，市场流动性充裕"))
        else:
            items.append(("⚠️", "量能萎缩（20日均量低于60日），增量资金不足"))
    except Exception as e:
        items.append(("⚠️", f"指数数据获取失败：{str(e)[:60]}"))

    try:
        act = _market_activity()
        up, down = float(act.get("上涨", 0)), float(act.get("下跌", 0))
        limit_down = float(act.get("真实跌停", 0))
        if up > down:
            score += 1
            items.append(("✅", f"市场赚钱效应正向（上涨{up:.0f}家 / 下跌{down:.0f}家）"))
        else:
            items.append(("❌", f"市场亏钱效应（上涨{up:.0f}家 / 下跌{down:.0f}家，真实跌停{limit_down:.0f}家）"))
    except Exception as e:
        items.append(("⚠️", f"市场情绪数据获取失败：{str(e)[:60]}"))

    if score >= 4:
        light, desc = "绿灯", "大盘环境健康，可正常执行买入信号"
    elif score >= 2:
        light, desc = "黄灯", "大盘环境中性偏弱，买入信号降一级执行，控制仓位"
    else:
        light, desc = "红灯", "大盘系统性风险高，再好的个股评分也建议只看不买"
    return {"light": light, "score": score, "max": 5, "desc": desc, "items": items}


# ================= 行业环境层 =================

@st.cache_data(ttl=604800, show_spinner=False)
def get_stock_industry(symbol: str) -> str | None:
    """个股所属东财行业。"""
    df = ak.stock_individual_info_em(symbol=symbol)
    info = dict(zip(df["item"], df["value"]))
    return info.get("行业")


@st.cache_data(ttl=3600, show_spinner=False)
def _industry_hist(board_name: str) -> pd.DataFrame:
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=180)).strftime("%Y%m%d")
    df = ak.stock_board_industry_hist_em(symbol=board_name, start_date=start,
                                         end_date=end, period="日k", adjust="")
    df["日期"] = pd.to_datetime(df["日期"])
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def _sector_fund_flow(indicator: str = "5日") -> pd.DataFrame:
    return ak.stock_fund_flow_sector_rank_em(indicator=indicator, sector_type="行业资金流")


def get_industry_env(symbol: str) -> dict:
    """行业相对强度（RS）+ 行业资金流 → 环境调整分（-10 ~ +10）。"""
    out = {"industry": None, "adj": 0, "items": [], "available": False}
    try:
        industry = get_stock_industry(symbol)
    except Exception:
        out["items"].append(("⚠️", "所属行业获取失败（接口限流），行业层暂不参与评分"))
        return out
    if not industry:
        out["items"].append(("⚠️", "未识别所属行业"))
        return out
    out["industry"] = industry
    adj = 0

    # 行业相对强度：行业指数 vs 沪深300 的 20/60 日超额收益
    try:
        ind = _industry_hist(industry)
        hs = _hs300_daily()
        def ret(s, n):
            return s.iloc[-1] / s.iloc[-1 - n] - 1
        rs20 = ret(ind["收盘"], 20) - ret(hs["close"], 20)
        rs60 = ret(ind["收盘"], 60) - ret(hs["close"], 60)
        if rs20 > 0 and rs60 > 0:
            adj += 6
            out["items"].append(("✅", f"行业20日/60日均跑赢沪深300（+{rs20*100:.1f}% / +{rs60*100:.1f}%），强势行业"))
        elif rs20 > 0 or rs60 > 0:
            adj += 2
            out["items"].append(("➖", f"行业相对强度分化（20日{rs20*100:+.1f}% / 60日{rs60*100:+.1f}%）"))
        else:
            adj -= 6
            out["items"].append(("❌", f"行业20日/60日均跑输沪深300（{rs20*100:.1f}% / {rs60*100:.1f}%），弱势行业"))
        out["rs20"], out["rs60"] = rs20, rs60
    except Exception as e:
        out["items"].append(("⚠️", f"行业指数获取失败：{str(e)[:60]}"))

    # 行业资金流（近5日排名）
    try:
        ff = _sector_fund_flow("5日")
        row = ff[ff["名称"] == industry]
        if len(row):
            rank = int(row.index[0]) + 1
            total = len(ff)
            net = float(row["主力净流入-净额"].iloc[0]) / 1e8
            if rank <= total * 0.3 and net > 0:
                adj += 4
                out["items"].append(("✅", f"行业5日主力净流入{net:.1f}亿，资金流排名 {rank}/{total}（前30%）"))
            elif net > 0:
                adj += 1
                out["items"].append(("➖", f"行业5日主力净流入{net:.1f}亿，排名 {rank}/{total}"))
            else:
                adj -= 4
                out["items"].append(("❌", f"行业5日主力净流出{abs(net):.1f}亿，排名 {rank}/{total}"))
    except Exception as e:
        out["items"].append(("⚠️", f"行业资金流获取失败：{str(e)[:60]}"))

    out["adj"] = max(-10, min(10, adj))
    out["available"] = True
    return out


# ================= 财务红旗（一票否决） =================

@st.cache_data(ttl=86400, show_spinner=False)
def _pledge_table() -> pd.DataFrame | None:
    """全市场质押比例表（数据按周五更新，自动回溯最近4个周五）。"""
    day = datetime.now()
    # 回到最近的周五
    day -= timedelta(days=(day.weekday() - 4) % 7)
    for _ in range(4):
        try:
            df = ak.stock_gpzy_pledge_ratio_em(date=day.strftime("%Y%m%d"))
            if df is not None and len(df):
                df["股票代码"] = df["股票代码"].astype(str).str.zfill(6)
                return df
        except Exception:
            pass
        day -= timedelta(days=7)
    return None


def check_red_flags(symbol: str, financial: pd.DataFrame | None,
                    style: str = "通用") -> dict:
    """财务红旗：命中1条→结论最高只给观望；命中2条以上→直接回避。"""
    flags, notes = [], []

    # 1. 大股东质押比例
    try:
        pt = _pledge_table()
        if pt is not None:
            row = pt[pt["股票代码"] == symbol]
            if len(row):
                ratio = float(row["质押比例"].iloc[0])
                if ratio > 50:
                    flags.append(f"质押比例高达{ratio:.0f}%，大股东资金链风险（历史暴雷股常见特征）")
                elif ratio > 30:
                    notes.append(f"质押比例{ratio:.0f}%，偏高需留意")
    except Exception:
        pass

    if financial is not None and len(financial):
        def col_val(keywords):
            for c in financial.columns:
                if all(k in c for k in keywords):
                    s = pd.to_numeric(financial[c], errors="coerce").dropna()
                    if len(s):
                        return s.iloc[0]
            return None

        # 2. 资产负债率（金融类公司天然高杠杆，豁免）
        if style != "金融":
            debt = col_val(["资产负债率"])
            if debt is not None and debt > 85:
                flags.append(f"资产负债率{debt:.0f}%，杠杆过高")

        # 3. 盈利现金含量：每股经营现金流 / 每股收益
        ocf = col_val(["每股经营性现金流"])
        eps = col_val(["摊薄每股收益"])
        if ocf is not None and eps is not None and eps > 0.05:
            if ocf / eps < 0.3:
                flags.append(f"经营现金流/净利润仅{ocf/eps:.0%}，利润含金量低，警惕纸面富贵")

    return {"flags": flags, "notes": notes}


# ================= 综合结论修正 =================

VERDICT_ORDER = ["回避", "观望", "谨慎买入", "建议买入"]


def finalize(rec: dict, market: dict, ind_env: dict, red: dict) -> dict:
    """核心评分 + 环境层修正 + 红旗否决 → 最终结论。"""
    final = dict(rec)
    adjust_log = []

    # 行业环境调整分
    total = rec["total"] + ind_env.get("adj", 0)
    total = max(0, min(100, total))
    if ind_env.get("adj"):
        adjust_log.append(f"行业环境调整 {ind_env['adj']:+d} 分")
    final["total"] = total

    # 按调整后总分重算结论
    if total >= 70:
        verdict = "建议买入"
    elif total >= 58:
        verdict = "谨慎买入"
    elif total >= 42:
        verdict = "观望"
    else:
        verdict = "回避"

    # 大盘红绿灯封顶
    if market["light"] == "红灯" and VERDICT_ORDER.index(verdict) > VERDICT_ORDER.index("观望"):
        verdict = "观望"
        adjust_log.append("大盘红灯：买入信号冻结，封顶至「观望」")
    elif market["light"] == "黄灯" and verdict == "建议买入":
        verdict = "谨慎买入"
        adjust_log.append("大盘黄灯：「建议买入」降级为「谨慎买入」")

    # 财务红旗一票否决
    if len(red["flags"]) >= 2:
        verdict = "回避"
        adjust_log.append("命中2条以上财务红旗：强制「回避」")
    elif len(red["flags"]) == 1 and VERDICT_ORDER.index(verdict) > VERDICT_ORDER.index("观望"):
        verdict = "观望"
        adjust_log.append("命中财务红旗：封顶至「观望」")

    action_map = {
        "建议买入": "环境与个股信号共振，可分批建仓",
        "谨慎买入": "信号偏多但环境或个股存在瑕疵，小仓位分批试探",
        "观望": "存在环境风险或信号不足，等待更好的买点",
        "回避": "风险信号明确，不建议买入",
    }
    final["verdict"] = verdict
    final["action"] = action_map[verdict]
    final["adjust_log"] = adjust_log
    final["core_total"] = rec["total"]
    return final


def full_analyze(symbol: str, market: dict | None = None) -> dict:
    """个股完整分析（含环境层与红旗），供个股页和排行榜共用。"""
    from analysis import analyze, industry_style
    d = data.load_all(symbol)
    if d["kline"] is None:
        raise RuntimeError(d["errors"].get("kline", "行情获取失败"))
    if market is None:
        market = get_market_light()
    ind_env = get_industry_env(symbol)
    r = analyze(d, industry=ind_env.get("industry"))
    red = check_red_flags(symbol, d.get("financial"),
                          style=industry_style(ind_env.get("industry")))
    r["rec"] = finalize(r["rec"], market, ind_env, red)
    r["market"] = market
    r["ind_env"] = ind_env
    r["red"] = red
    r["data"] = d
    return r
