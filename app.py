"""A股专业分析工具 - Streamlit 入口。运行: streamlit run app.py"""
import data  # noqa: F401  必须最先导入（处理代理环境变量）

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

st.set_page_config(page_title="A股专业分析", page_icon="📈", layout="wide")

VERDICT_COLOR = {"建议买入": "#d62728", "谨慎买入": "#ff7f0e", "观望": "#7f7f7f", "回避": "#2ca02c"}


def render_kline(df: pd.DataFrame, name: str):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{name} 日K（前复权）", "成交量", "MACD"))
    fig.add_trace(go.Candlestick(
        x=df["date"], open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color="#d62728", decreasing_line_color="#2ca02c", name="K线"), row=1, col=1)
    for n, color in [(5, "#1f77b4"), (10, "#ff7f0e"), (20, "#9467bd"), (60, "#8c564b")]:
        fig.add_trace(go.Scatter(x=df["date"], y=df[f"ma{n}"], name=f"MA{n}",
                                 line=dict(width=1, color=color)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["boll_up"], name="布林上轨",
                             line=dict(width=0.8, dash="dot", color="#aaaaaa")), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["boll_low"], name="布林下轨",
                             line=dict(width=0.8, dash="dot", color="#aaaaaa"),
                             fill="tonexty", fillcolor="rgba(200,200,200,0.1)"), row=1, col=1)

    vol_colors = ["#d62728" if c >= o else "#2ca02c" for c, o in zip(df["close"], df["open"])]
    fig.add_trace(go.Bar(x=df["date"], y=df["volume"], marker_color=vol_colors,
                         name="成交量", showlegend=False), row=2, col=1)

    macd_colors = ["#d62728" if v >= 0 else "#2ca02c" for v in df["macd"]]
    fig.add_trace(go.Bar(x=df["date"], y=df["macd"], marker_color=macd_colors,
                         name="MACD柱", showlegend=False), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["dif"], name="DIF",
                             line=dict(width=1, color="#1f77b4")), row=3, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["dea"], name="DEA",
                             line=dict(width=1, color="#ff7f0e")), row=3, col=1)

    fig.update_layout(height=700, xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.08),
                      margin=dict(t=60, b=20))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, use_container_width=True)


def render_score_detail(title: str, result: dict):
    st.subheader(f"{title}　{result['score']}/{result['max']}分")
    rows = [{"指标": i[0], "得分": f"{i[1]}/{i[2]}", "解读": i[3]} for i in result["items"]]
    st.table(pd.DataFrame(rows))


def render_fund_flow_chart(ff: pd.DataFrame):
    df = ff.tail(60).copy()
    net = pd.to_numeric(df["主力净流入-净额"], errors="coerce") / 1e8
    colors = ["#d62728" if v >= 0 else "#2ca02c" for v in net]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["日期"], y=net, marker_color=colors, name="主力净流入(亿)"))
    fig.add_trace(go.Scatter(x=df["日期"], y=net.rolling(5).mean(),
                             name="5日均值", line=dict(color="#1f77b4", width=1.5)))
    fig.update_layout(title="近60日主力资金净流入（亿元）", height=320, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)


def render_valuation_chart(val: pd.DataFrame):
    df = val.tail(750)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df["数据日期"], y=df["PE(TTM)"], name="PE(TTM)",
                             line=dict(color="#1f77b4", width=1.2)))
    fig.add_trace(go.Scatter(x=df["数据日期"], y=df["市净率"], name="PB",
                             line=dict(color="#ff7f0e", width=1.2)), secondary_y=True)
    fig.update_layout(title="近三年估值走势", height=320, margin=dict(t=40, b=20))
    fig.update_yaxes(title_text="PE(TTM)", secondary_y=False)
    fig.update_yaxes(title_text="PB", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


def render_ranking_tab():
    st.subheader("🏆 行业板块排行榜")
    st.caption("先用板块成分股快照初筛（板块内相对估值+量价），再对初筛靠前的股票做完整三维评分，选出每个板块得分最高的股票。")

    try:
        boards = data.get_industry_boards()
    except Exception as e:
        st.error(f"板块列表获取失败（东财接口可能临时限流，几分钟后重试）：{str(e)[:120]}")
        return

    board_names = boards["板块名称"].tolist() if "板块名称" in boards.columns else boards.iloc[:, 1].tolist()
    c1, c2 = st.columns([3, 1])
    selected = c1.multiselect("选择行业板块（可多选，建议每次不超过3个以免触发数据源限流）",
                              board_names, max_selections=3)
    top_n = c2.slider("每板块选出 Top", 3, 10, 5)

    if st.button("开始排行分析", type="primary"):
        if not selected:
            st.warning("请先选择至少一个板块")
            return
        import ranking
        import environment
        try:
            m = environment.get_market_light()
            icon = {"绿灯": "🟢", "黄灯": "🟡", "红灯": "🔴"}[m["light"]]
            st.markdown(f"{icon} **大盘环境：{m['light']}** — {m['desc']}（结论已按环境修正）")
        except Exception:
            pass
        for board in selected:
            st.markdown(f"### 📌 {board}")
            bar = st.progress(0.0, text=f"{board}：准备中...")

            def cb(p, text):
                bar.progress(min(p, 1.0), text=text)

            try:
                result, warns = ranking.deep_rank(board, top_n=top_n, progress_cb=cb)
            except Exception as e:
                bar.empty()
                st.error(f"{board} 分析失败：{str(e)[:150]}")
                continue
            bar.empty()
            for w in warns:
                st.warning(w)
            if len(result) == 0:
                st.info("该板块暂无有效结果")
                continue

            def color_verdict(v):
                return f"color: {VERDICT_COLOR.get(v, '#000')}; font-weight: bold"

            st.dataframe(result.style.map(color_verdict, subset=["结论"]),
                         use_container_width=True)
        st.caption("提示：在「个股分析」页输入代码可查看任意股票的完整分析明细。")


def main():
    st.title("📈 A股专业分析工具")
    st.caption("技术面 + 基本面 + 资金面 三维分析，综合评分给出操作建议。数据来源：东方财富（akshare）。本工具仅供研究参考，不构成投资建议。")

    tab_stock, tab_rank, tab_bt = st.tabs(["🔍 个股分析", "🏆 板块排行榜", "🧪 策略回测"])
    with tab_rank:
        render_ranking_tab()
    with tab_bt:
        render_backtest_tab()
    with tab_stock:
        render_stock_tab()


def render_backtest_tab():
    st.subheader("🧪 评分体系历史回测")
    st.caption("逐日重算历史评分（技术面 + 点时估值分位，无未来函数），验证「高分买入」在该股历史上的真实胜率，并模拟 评分上穿买入/下穿卖出 的策略净值（含0.1%单边成本）。")

    mode = st.radio("回测模式", ["单股详细回测", "多股批量验证"], horizontal=True, key="bt_mode")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    if mode == "单股详细回测":
        bt_symbol = c1.text_input("股票代码", value="600519", key="bt_symbol")
    else:
        bt_symbols_text = c1.text_input("股票代码列表（空格或逗号分隔）",
                                        value="600519 601318 600036 300750 601899",
                                        key="bt_symbols",
                                        help="建议混合不同风格：消费/金融/科技/周期，5~10只为宜")
    buy_thr = c2.slider("买入阈值", 50, 85, 65, key="bt_buy")
    sell_thr = c3.slider("卖出阈值", 30, 60, 45, key="bt_sell")
    run_bt = c4.button("开始回测", type="primary")
    gate = st.checkbox("估值趋势门控（股价持续处于60日线下方时，估值分打5折，防价值陷阱）",
                       value=True, key="bt_gate")

    if not run_bt:
        return
    if sell_thr >= buy_thr:
        st.error("卖出阈值必须低于买入阈值")
        return

    import backtest

    if mode == "多股批量验证":
        symbols = [s for s in bt_symbols_text.replace("，", " ").replace(",", " ").split() if s.isdigit() and len(s) == 6]
        if not symbols:
            st.error("请输入至少一个6位股票代码")
            return
        bar = st.progress(0.0, text="准备中...")
        r = backtest.run_batch(symbols, buy_threshold=buy_thr, sell_threshold=sell_thr, gate=gate,
                               progress_cb=lambda p, t: bar.progress(min(p, 1.0), text=t))
        bar.empty()
        for e in r["errors"]:
            st.warning(e)
        st.markdown("**🔬 合并样本：高分信号日前瞻收益**（跨股票汇总，统计意义更强）")
        st.table(r["pooled"])
        st.markdown("**各股策略表现**")
        if len(r["per_stock"]):
            st.dataframe(r["per_stock"], use_container_width=True)
        st.caption("关注点：①「超额」是否为正且随窗口放大；②「信号胜率」是否高于「基准胜率」。若不成立，说明当前阈值/权重对这批股票无预测力。")
        return
    with st.spinner("回测中：逐日重算约900个交易日的评分..."):
        try:
            r = backtest.run_backtest(bt_symbol.strip(), buy_threshold=buy_thr, sell_threshold=sell_thr, gate=gate)
        except Exception as e:
            st.error(f"回测失败：{str(e)[:150]}")
            return

    s = r["summary"]
    st.markdown(f"**{s['回测区间']}**　·　{s['评分口径']}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("策略总收益", s["策略总收益"], delta=f"vs 买入持有 {s['买入持有收益']}", delta_color="off")
    m2.metric("策略年化", s["策略年化"])
    m3.metric("策略最大回撤", s["策略最大回撤"], delta=f"vs 持有 {s['买入持有最大回撤']}", delta_color="off")
    m4.metric("持仓占比 / 交易次数", f"{s['持仓时间占比']} / {s['交易次数']}次")

    st.markdown("**📊 高分信号日的前瞻收益**（评分≥阈值当天买入，持有N日的统计）")
    st.table(r["stats"])

    df = r["df"]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.65, 0.35], subplot_titles=("策略净值 vs 买入持有", "每日综合评分"))
    fig.add_trace(go.Scatter(x=df["date"], y=df["strategy_nav"], name="评分策略",
                              line=dict(color="#d62728", width=1.6)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["buyhold_nav"], name="买入持有",
                              line=dict(color="#7f7f7f", width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["date"], y=df["score"], name="评分",
                              line=dict(color="#1f77b4", width=1)), row=2, col=1)
    fig.add_hline(y=buy_thr, line_dash="dot", line_color="#d62728", row=2, col=1)
    fig.add_hline(y=sell_thr, line_dash="dot", line_color="#2ca02c", row=2, col=1)
    fig.update_layout(height=560, margin=dict(t=50, b=20), legend=dict(orientation="h", y=1.07))
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    st.plotly_chart(fig, use_container_width=True)

    if len(r["trades"]):
        with st.expander(f"交易明细（{len(r['trades'])}笔）"):
            st.dataframe(r["trades"], use_container_width=True)
    st.caption("⚠️ 单一个股回测存在幸存者偏差与过拟合风险，结论仅说明该评分在此股历史上的表现，不保证未来有效。")


def render_stock_tab():
    with st.sidebar:
        st.header("股票选择")
        symbol = st.text_input("股票代码（6位数字）", value="600519",
                               help="如 600519 贵州茅台、000858 五粮液、300750 宁德时代")
        run = st.button("开始分析", type="primary", use_container_width=True)
        st.divider()
        st.markdown("**评分权重**\n- 技术面 40分\n- 基本面 35分\n- 资金面 25分")
        st.markdown("**结论阈值**\n- ≥70 建议买入\n- 58~69 谨慎买入\n- 42~57 观望\n- <42 回避")

    if not run and "last_symbol" not in st.session_state:
        st.info("👈 在左侧输入股票代码，点击「开始分析」")
        return
    if run:
        st.session_state["last_symbol"] = symbol.strip()
    symbol = st.session_state["last_symbol"]

    if not (symbol.isdigit() and len(symbol) == 6):
        st.error("请输入6位数字的A股代码")
        return

    import environment

    with st.spinner(f"正在获取 {symbol} 的行情、财务、资金与环境数据..."):
        try:
            result = environment.full_analyze(symbol)
        except Exception as e:
            st.error(f"分析失败：{str(e)[:150]}")
            return

    d = result["data"]
    if d["errors"]:
        for k, v in d["errors"].items():
            st.warning(f"部分数据获取失败（{k}），对应维度得分为0：{str(v)[:120]}")

    rec, tech, fund, cap = result["rec"], result["tech"], result["fund"], result["cap"]
    market, ind_env, red = result["market"], result["ind_env"], result["red"]
    name = d["name"]

    # ---- 大盘红绿灯横幅 ----
    LIGHT_STYLE = {"绿灯": ("🟢", "#e8f5e9"), "黄灯": ("🟡", "#fff8e1"), "红灯": ("🔴", "#ffebee")}
    icon, bg = LIGHT_STYLE[market["light"]]
    st.markdown(
        f"""<div style="background:{bg}; border-radius:8px; padding:10px 16px; margin-bottom:10px;">
        {icon} <b>大盘环境：{market['light']}（{market['score']}/{market['max']}）</b>　{market['desc']}</div>""",
        unsafe_allow_html=True)
    with st.expander("大盘环境明细", expanded=False):
        for mark, text in market["items"]:
            st.markdown(f"{mark} {text}")

    # ---- 结论卡片 ----
    color = VERDICT_COLOR[rec["verdict"]]
    score_text = (f"综合评分 {rec['total']}/100"
                  + (f"（核心{rec['core_total']}{ind_env['adj']:+d}行业）" if ind_env.get("adj") else ""))
    industry_text = f"　所属行业：{ind_env['industry']}" if ind_env.get("industry") else ""
    st.markdown(
        f"""<div style="border:2px solid {color}; border-radius:12px; padding:16px 24px; margin-bottom:12px;">
        <span style="font-size:1.6em; font-weight:bold;">{name}（{symbol}）</span>
        <span style="font-size:2em; font-weight:bold; color:{color}; margin-left:24px;">{rec['verdict']}</span>
        <span style="font-size:1.4em; margin-left:24px;">{score_text}</span>
        <div style="margin-top:8px; color:#555;">{rec['action']}{industry_text}</div></div>""",
        unsafe_allow_html=True)

    # ---- 红旗与结论修正过程 ----
    for f in red["flags"]:
        st.error(f"🚩 财务红旗：{f}")
    for n in red["notes"]:
        st.warning(f"⚠️ {n}")
    for log in rec.get("adjust_log", []):
        st.info(f"⚖️ {log}")
    if ind_env["items"]:
        with st.expander(f"行业环境明细（调整 {ind_env.get('adj', 0):+d} 分）", expanded=False):
            for mark, text in ind_env["items"]:
                st.markdown(f"{mark} {text}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("最新收盘", f"{rec['close']:.2f}")
    c2.metric("参考买入区间", f"{rec['buy_zone'][0]} ~ {rec['buy_zone'][1]}")
    c3.metric("支撑位", f"{tech['sr']['support']:.2f}")
    c4.metric("压力位", f"{rec['resistance']:.2f}")
    c5.metric("参考止损", f"{rec['stop_loss']:.2f}")

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("技术面", f"{tech['score']}/{tech['max']}")
    sc2.metric("基本面", f"{fund['score']}/{fund['max']}")
    sc3.metric("资金面", f"{cap['score']}/{cap['max']}")

    # ---- 图表与明细 ----
    tab1, tab2, tab3, tab4 = st.tabs(["📊 K线与技术指标", "💰 基本面", "🏦 资金面", "📋 评分明细"])

    with tab1:
        render_kline(tech["df"], name)
        last = tech["df"].iloc[-1]
        st.caption(f"RSI(14)={last['rsi']:.1f}　KDJ: K={last['k']:.1f} D={last['d']:.1f} J={last['j']:.1f}　"
                   f"近60日价格区间位置：{tech['sr']['pos_in_range']}%")

    with tab2:
        style_desc = {"科技成长": "PS/PEG/营收增速为主（科技股PE常失真）",
                      "金融": "PB+ROE为主（金融股天然高杠杆）",
                      "周期": "PB分位为主（周期股低PE常是顶部陷阱）",
                      "通用": "PE/PB分位 + ROE + 成长"}
        st.info(f"📐 估值尺子：**{fund.get('style', '通用')}** — {style_desc[fund.get('style', '通用')]}")
        if d["valuation"] is not None:
            if fund.get("ps") or fund.get("peg"):
                e1, e2 = st.columns(2)
                e1.metric("PS（市销率）", f"{fund['ps']:.2f}" if fund.get("ps") else "—",
                          delta=f"历史分位 {fund['ps_pct']:.0f}%" if fund.get("ps_pct") is not None else None,
                          delta_color="off")
                e2.metric("PEG", f"{fund['peg']:.2f}" if fund.get("peg") else "—")
            v1, v2, v3, v4 = st.columns(4)
            v1.metric("PE(TTM)", f"{fund['pe']:.1f}" if fund["pe"] else "—")
            v2.metric("PE历史分位", f"{fund['pe_pct']:.0f}%" if fund["pe_pct"] is not None else "—")
            v3.metric("PB", f"{fund['pb']:.2f}" if fund["pb"] else "—")
            v4.metric("PB历史分位", f"{fund['pb_pct']:.0f}%" if fund["pb_pct"] is not None else "—")
            render_valuation_chart(d["valuation"])
        if d["financial"] is not None:
            st.markdown("**近期财务指标（按报告期）**")
            cols = [c for c in d["financial"].columns
                    if any(k in c for k in ["日期", "净资产收益率", "每股收益", "净利润增长率", "主营业务收入增长率", "资产负债率"])]
            st.dataframe(d["financial"][cols].head(8), use_container_width=True)

    with tab3:
        if d["fund_flow"] is not None:
            render_fund_flow_chart(d["fund_flow"])
            st.dataframe(d["fund_flow"].sort_values("日期", ascending=False).head(10),
                         use_container_width=True)
        else:
            st.info("资金流数据暂不可用")

    with tab4:
        render_score_detail("技术面", tech)
        render_score_detail("基本面", fund)
        render_score_detail("资金面", cap)

    st.divider()
    st.caption("⚠️ 免责声明：本工具基于公开数据的量化规则生成分析结果，仅供学习研究使用，不构成任何投资建议。股市有风险，投资需谨慎。")


if __name__ == "__main__":
    main()
