"""单股历史回测：逐日计算评分（无未来函数），验证评分体系的实际胜率。

可回测维度：技术面（40分，完整复现）+ 估值分位（点时分位，映射到35分）。
资金面历史数据只有120天，不参与回测。总分折算回 0~100（75分制 / 0.75）。
"""
import numpy as np
import pandas as pd

import data
from indicators import add_indicators


def _technical_score_series(df: pd.DataFrame) -> pd.Series:
    """逐日技术面评分（0~40），向量化实现，规则与 analysis.score_technical 一致。"""
    s = pd.Series(0.0, index=df.index)

    # 趋势 10
    s += (df["close"] > df["ma20"]) * 5 + (df["ma20"] > df["ma60"]) * 5
    # MACD 8
    s += (df["dif"] > df["dea"]) * 4 + (df["dif"] > 0) * 2 + (df["macd"] > df["macd"].shift(1)) * 2
    # RSI 8
    rsi = df["rsi"]
    s += np.select([rsi < 30, rsi < 40, rsi <= 60, rsi <= 70], [8, 6, 4, 3], default=0)
    # KDJ 6
    golden = (df["k"] > df["d"]) & (df["k"].shift(1) <= df["d"].shift(1))
    s += np.select([golden, df["j"] < 0, df["k"] > df["d"], df["j"] > 100],
                   [6, 5, 4, 0], default=2)
    # 布林 4
    c = df["close"]
    s += np.select([c < df["boll_low"], c < df["boll_mid"], c < df["boll_up"]],
                   [3, 4, 2], default=0)
    # 量价 4
    up = df["pct_chg"] > 0
    vr = df["vol_ratio"]
    s += np.select([up & (vr > 1.2), up, (~up) & (vr < 0.8)], [4, 2, 2], default=0)
    return s


def _valuation_score_series(kline: pd.DataFrame, valuation: pd.DataFrame | None) -> pd.Series:
    """逐日估值分（0~35）：PE/PB 相对【截至当日】历史的分位（expanding，无未来函数）。"""
    if valuation is None or len(valuation) < 250:
        return pd.Series(np.nan, index=kline.index)

    v = valuation.copy()
    v["pe"] = pd.to_numeric(v["PE(TTM)"], errors="coerce")
    v["pb"] = pd.to_numeric(v["市净率"], errors="coerce")
    v = v.dropna(subset=["pe", "pb"]).reset_index(drop=True)

    def expanding_pct(series, positive_only=True):
        arr = series.to_numpy()
        out = np.full(len(arr), np.nan)
        for i in range(60, len(arr)):  # 至少60日历史才计算分位
            hist = arr[: i + 1]
            if positive_only:
                hist = hist[hist > 0]
            if len(hist) > 30 and arr[i] > 0:
                out[i] = (hist < arr[i]).mean() * 100
        return out

    v["pe_pct"] = expanding_pct(v["pe"])
    v["pb_pct"] = expanding_pct(v["pb"], positive_only=False)

    def pct_to_score(pct, full):
        return np.select([pct < 20, pct < 40, pct < 60, pct < 80],
                         [full, full * 0.7, full * 0.5, full * 0.2], default=0)

    v["val_score"] = pct_to_score(v["pe_pct"], 20) + pct_to_score(v["pb_pct"], 15)
    v.loc[v["pe_pct"].isna(), "val_score"] = np.nan

    val_map = pd.Series(v["val_score"].values, index=pd.to_datetime(v["数据日期"]).values)
    return kline["date"].map(lambda d: val_map.get(d, np.nan)).ffill()


def run_backtest(symbol: str, buy_threshold: int = 65, sell_threshold: int = 45,
                 fwd_windows: tuple = (5, 20, 60), gate: bool = True) -> dict:
    """回测：评分≥buy_threshold 买入持有，跌破 sell_threshold 卖出。"""
    kline = data.get_daily_kline(symbol, days=900)
    try:
        valuation = data.get_valuation(symbol)
    except Exception:
        valuation = None

    df = add_indicators(kline)
    tech = _technical_score_series(df)
    val = _valuation_score_series(df, valuation)
    if gate:
        # 趋势门控：近10日多数时间在60日线下方 → 估值分打5折
        # （下跌趋势中低估值=价值陷阱；用10日平滑避免均线附近评分抖动）
        below = (df["close"] < df["ma60"]).rolling(10, min_periods=1).mean() > 0.5
        val = val.where(~below, val * 0.5)

    has_val = val.notna().mean() > 0.5
    if has_val:
        score = (tech + val) / 75 * 100  # 折算回百分制
        score_mode = "技术面40 + 估值35（折算百分制）"
    else:
        score = tech / 40 * 100
        score_mode = "仅技术面（估值历史不足，折算百分制）"
    df["score"] = score.round(1)

    warm = 60  # 指标预热期
    df = df.iloc[warm:].reset_index(drop=True)
    n = len(df)
    if n < 120:
        raise RuntimeError("有效历史数据不足（<120日），无法回测")

    # ---- 信号日的前瞻收益统计 ----
    stats = []
    for w in fwd_windows:
        fwd = df["close"].shift(-w) / df["close"] - 1
        sig_mask = df["score"] >= buy_threshold
        sig, all_ = fwd[sig_mask].dropna(), fwd.dropna()
        if len(sig) >= 5:
            stats.append({
                "窗口": f"{w}日", "信号日数": len(sig),
                "信号日均收益": f"{sig.mean() * 100:.2f}%",
                "全样本均收益": f"{all_.mean() * 100:.2f}%",
                "超额": f"{(sig.mean() - all_.mean()) * 100:+.2f}%",
                "信号胜率": f"{(sig > 0).mean() * 100:.0f}%",
                "基准胜率": f"{(all_ > 0).mean() * 100:.0f}%",
            })
        else:
            stats.append({"窗口": f"{w}日", "信号日数": len(sig), "信号日均收益": "样本不足",
                          "全样本均收益": "—", "超额": "—", "信号胜率": "—", "基准胜率": "—"})

    # ---- 简单策略净值：评分上穿买入，下穿卖出（次日开盘成交，含0.1%单边成本） ----
    cost = 0.001
    position = np.zeros(n)
    hold = False
    trades = []
    for i in range(n - 1):
        sc = df["score"].iloc[i]
        if not hold and sc >= buy_threshold:
            hold = True
            trades.append({"类型": "买入", "日期": df["date"].iloc[i + 1].date(),
                           "价格": df["open"].iloc[i + 1], "评分": sc})
        elif hold and sc < sell_threshold:
            hold = False
            trades.append({"类型": "卖出", "日期": df["date"].iloc[i + 1].date(),
                           "价格": df["open"].iloc[i + 1], "评分": sc})
        position[i + 1] = 1.0 if hold else 0.0

    daily_ret = df["close"].pct_change().fillna(0).to_numpy()
    strat_ret = daily_ret * position
    # 扣交易成本
    switches = np.abs(np.diff(position, prepend=0))
    strat_ret -= switches * cost
    df["strategy_nav"] = (1 + pd.Series(strat_ret, index=df.index)).cumprod()
    df["buyhold_nav"] = (1 + pd.Series(daily_ret, index=df.index)).cumprod()

    def mdd(nav):
        return ((nav / nav.cummax()) - 1).min()

    years = n / 243
    summary = {
        "回测区间": f"{df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}（{n}个交易日）",
        "评分口径": score_mode,
        "策略总收益": f"{(df['strategy_nav'].iloc[-1] - 1) * 100:.1f}%",
        "买入持有收益": f"{(df['buyhold_nav'].iloc[-1] - 1) * 100:.1f}%",
        "策略年化": f"{(df['strategy_nav'].iloc[-1] ** (1 / years) - 1) * 100:.1f}%",
        "策略最大回撤": f"{mdd(df['strategy_nav']) * 100:.1f}%",
        "买入持有最大回撤": f"{mdd(df['buyhold_nav']) * 100:.1f}%",
        "持仓时间占比": f"{position.mean() * 100:.0f}%",
        "交易次数": len(trades),
    }
    return {"df": df, "stats": pd.DataFrame(stats), "summary": summary,
            "trades": pd.DataFrame(trades), "score_mode": score_mode}


def run_batch(symbols: list[str], buy_threshold: int = 65, sell_threshold: int = 45,
              fwd_windows: tuple = (5, 20, 60), progress_cb=None, gate: bool = True) -> dict:
    """多股批量回测：汇总信号日前瞻收益（合并样本），避免单股特例误导。"""
    import time

    per_stock, errors = [], []
    pooled = {w: {"sig": [], "all": []} for w in fwd_windows}

    for i, sym in enumerate(symbols):
        if progress_cb:
            progress_cb(i / len(symbols), f"回测中 {sym}（{i + 1}/{len(symbols)}）")
        try:
            r = run_backtest(sym, buy_threshold, sell_threshold, fwd_windows, gate=gate)
        except Exception as e:
            errors.append(f"{sym}: {str(e)[:80]}")
            continue
        df, s = r["df"], r["summary"]
        try:
            name = data.get_stock_name_map().get(sym, sym)
        except Exception:
            name = sym
        per_stock.append({
            "代码": sym, "名称": name,
            "策略收益": s["策略总收益"], "买入持有": s["买入持有收益"],
            "策略回撤": s["策略最大回撤"], "持有回撤": s["买入持有最大回撤"],
            "持仓占比": s["持仓时间占比"], "交易次数": s["交易次数"],
        })
        sig_mask = df["score"] >= buy_threshold
        for w in fwd_windows:
            fwd = (df["close"].shift(-w) / df["close"] - 1).dropna()
            pooled[w]["sig"].extend(fwd[sig_mask.reindex(fwd.index, fill_value=False)].tolist())
            pooled[w]["all"].extend(fwd.tolist())
        time.sleep(0.3)

    pooled_rows = []
    for w in fwd_windows:
        sig = pd.Series(pooled[w]["sig"])
        all_ = pd.Series(pooled[w]["all"])
        if len(sig) >= 30:
            pooled_rows.append({
                "窗口": f"{w}日", "信号样本": len(sig), "全样本": len(all_),
                "信号均收益": f"{sig.mean() * 100:.2f}%",
                "全样本均收益": f"{all_.mean() * 100:.2f}%",
                "超额": f"{(sig.mean() - all_.mean()) * 100:+.2f}%",
                "信号胜率": f"{(sig > 0).mean() * 100:.0f}%",
                "基准胜率": f"{(all_ > 0).mean() * 100:.0f}%",
            })
        else:
            pooled_rows.append({"窗口": f"{w}日", "信号样本": len(sig), "全样本": len(all_),
                                "信号均收益": "样本不足", "全样本均收益": "—", "超额": "—",
                                "信号胜率": "—", "基准胜率": "—"})
    if progress_cb:
        progress_cb(1.0, "批量回测完成")
    return {"per_stock": pd.DataFrame(per_stock), "pooled": pd.DataFrame(pooled_rows),
            "errors": errors}
