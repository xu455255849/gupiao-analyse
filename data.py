"""数据获取模块：基于 akshare，带缓存与代理绕过。"""
import os
from datetime import datetime, timedelta

# 东财等国内数据源走本地代理会失败，强制直连
os.environ["NO_PROXY"] = "*"
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(k, None)

import akshare as ak
import pandas as pd
import streamlit as st


@st.cache_data(ttl=86400, show_spinner=False)
def get_stock_name_map() -> dict:
    """A股代码->名称映射（来自交易所官网，不走东财）。"""
    df = ak.stock_info_a_code_name()
    return dict(zip(df["code"], df["name"]))


def _sina_symbol(symbol: str) -> str:
    if symbol.startswith(("6", "9", "5")):
        return f"sh{symbol}"
    if symbol.startswith(("4", "8")):
        return f"bj{symbol}"
    return f"sz{symbol}"


@st.cache_data(ttl=3600, show_spinner=False)
def get_daily_kline(symbol: str, days: int = 500) -> pd.DataFrame:
    """前复权日K线。优先东方财富，被限流时自动切换新浪。"""
    end = datetime.now().strftime("%Y%m%d")
    start = (datetime.now() - timedelta(days=int(days * 1.6))).strftime("%Y%m%d")
    try:
        df = ak.stock_zh_a_hist_em(symbol=symbol, period="daily",
                                   start_date=start, end_date=end, adjust="qfq")
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "收盘": "close", "最高": "high",
            "最低": "low", "成交量": "volume", "成交额": "amount",
            "涨跌幅": "pct_chg", "换手率": "turnover",
        })
    except Exception:
        # 东财限流时的备用源：新浪
        df = ak.stock_zh_a_hist_sina(symbol=_sina_symbol(symbol), period="daily",
                                     start_date=start, end_date=end, adjust="qfq")
        df["pct_chg"] = df["close"].pct_change() * 100
        df["turnover"] = df["turnover"] * 100
    df["date"] = pd.to_datetime(df["date"])
    return df.tail(days).reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def get_valuation(symbol: str) -> pd.DataFrame:
    """估值历史：PE(TTM)、PB、PEG、总市值等。"""
    df = ak.stock_value_em(symbol=symbol)
    df["数据日期"] = pd.to_datetime(df["数据日期"])
    return df


@st.cache_data(ttl=86400, show_spinner=False)
def get_financial_indicators(symbol: str, years: int = 4) -> pd.DataFrame:
    """财务指标（ROE、每股收益、增长率等），按报告期倒序。"""
    start_year = str(datetime.now().year - years)
    df = ak.stock_financial_analysis_indicator(symbol=symbol, start_year=start_year)
    df["日期"] = pd.to_datetime(df["日期"])
    return df.sort_values("日期", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def get_fund_flow(symbol: str) -> pd.DataFrame:
    """个股资金流历史（约120个交易日），主力/超大单/大单净流入。"""
    df = ak.stock_fund_flow_individual_hist_em(symbol=symbol)
    df["日期"] = pd.to_datetime(df["日期"])
    return df.sort_values("日期").reset_index(drop=True)


@st.cache_data(ttl=86400, show_spinner=False)
def get_industry_boards() -> pd.DataFrame:
    """东财行业板块列表（含当日涨跌幅、领涨股等）。"""
    return ak.stock_board_industry_name_em()


@st.cache_data(ttl=3600, show_spinner=False)
def get_board_constituents(board_name: str) -> pd.DataFrame:
    """行业板块成分股快照（最新价、涨跌幅、换手率、市盈率、市净率等）。"""
    return ak.stock_board_industry_cons_em(symbol=board_name)


def load_all(symbol: str) -> dict:
    """一次性拉取全部数据，单项失败不影响整体。"""
    out = {"symbol": symbol, "errors": {}}
    try:
        out["name"] = get_stock_name_map().get(symbol, symbol)
    except Exception as e:
        out["name"] = symbol
        out["errors"]["name"] = str(e)
    for key, fn in [("kline", get_daily_kline), ("valuation", get_valuation),
                    ("financial", get_financial_indicators), ("fund_flow", get_fund_flow)]:
        try:
            out[key] = fn(symbol)
        except Exception as e:
            out[key] = None
            out["errors"][key] = str(e)
    return out
