"""板块排行榜：成分股初筛 + 三维深度评分排名。"""
import time

import pandas as pd

import data
import environment


def lightweight_rank(cons: pd.DataFrame) -> pd.DataFrame:
    """用板块成分股快照做初筛打分（0~100），不发额外请求。

    规则：板块内相对估值（PE、PB 越低越好）+ 当日量价表现。
    """
    df = cons.copy()
    for col in ["最新价", "涨跌幅", "换手率", "市盈率-动态", "市净率"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    pe = df.get("市盈率-动态")
    pb = df.get("市净率")
    chg = df.get("涨跌幅")
    tr = df.get("换手率")

    score = pd.Series(0.0, index=df.index)
    # PE：仅正值参与，板块内分位越低分越高（0~35）
    if pe is not None:
        valid = pe > 0
        score += valid.astype(float) * (1 - pe.where(valid).rank(pct=True).fillna(1)) * 35
    # PB：正值分位越低分越高（0~25）
    if pb is not None:
        valid = pb > 0
        score += valid.astype(float) * (1 - pb.where(valid).rank(pct=True).fillna(1)) * 25
    # 当日涨跌幅：温和上涨最好，暴涨暴跌都降分（0~20）
    if chg is not None:
        score += (20 - (chg - 2).abs().clip(upper=10) * 2).clip(lower=0)
    # 换手率：适中活跃度最好（1%~8%）（0~20）
    if tr is not None:
        score += ((tr >= 1) & (tr <= 8)).astype(float) * 20 + \
                 ((tr > 0.3) & (tr < 1)).astype(float) * 10

    df["初筛分"] = score.round(1)
    return df.sort_values("初筛分", ascending=False).reset_index(drop=True)


def deep_rank(board_name: str, top_n: int = 5, progress_cb=None) -> tuple[pd.DataFrame, list[str]]:
    """对板块初筛 Top N 的股票做完整三维评分，返回排名表和警告信息。"""
    cons = data.get_board_constituents(board_name)
    ranked = lightweight_rank(cons)
    candidates = ranked.head(top_n)
    market = environment.get_market_light()  # 各股共用，避免重复请求

    rows, warnings = [], []
    for i, (_, row) in enumerate(candidates.iterrows()):
        code = str(row["代码"]).zfill(6)
        if progress_cb:
            progress_cb(i / len(candidates), f"{board_name}：正在分析 {code} {row['名称']} ({i + 1}/{len(candidates)})")
        try:
            r = environment.full_analyze(code, market=market)
            flags = len(r["red"]["flags"])
            rows.append({
                "代码": code, "名称": row["名称"],
                "综合评分": r["rec"]["total"], "结论": r["rec"]["verdict"],
                "技术面": f"{r['tech']['score']}/40",
                "基本面": f"{r['fund']['score']}/35",
                "资金面": f"{r['cap']['score']}/25",
                "行业调整": f"{r['ind_env'].get('adj', 0):+d}",
                "红旗": f"🚩×{flags}" if flags else "—",
                "最新价": r["rec"]["close"],
                "参考买入区间": f"{r['rec']['buy_zone'][0]} ~ {r['rec']['buy_zone'][1]}",
            })
        except Exception as e:
            warnings.append(f"{code} {row['名称']}：{str(e)[:80]}")
        time.sleep(0.3)  # 控制请求频率，避免触发数据源限流

    if progress_cb:
        progress_cb(1.0, f"{board_name}：分析完成")
    result = pd.DataFrame(rows)
    if len(result):
        result = result.sort_values("综合评分", ascending=False).reset_index(drop=True)
        result.index = result.index + 1
    return result, warnings
