"""
VibeFinance - Multi-Agent Investment Intelligence for Indian Retail Investors
Offline multi-agent architecture with visible reasoning, citation grounding,
risk guardrails, behavioral nudges, and backtest evaluation.
"""

import os
import json
import asyncio
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

# Streamlit Page Config
st.set_page_config(
    page_title="VibeFinance | Explainable Multi-Agent Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling (Dark Theme & Glassmorphism)
st.markdown("""
<style>
    /* Global Typography & Palette */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    code, pre {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .stApp {
        background-color: #0b0f17;
        color: #e2e8f0;
    }
    
    /* Metrics and Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.3);
        margin-bottom: 12px;
    }
    
    .agent-pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    
    .pill-green { background-color: rgba(16, 185, 129, 0.18); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .pill-red { background-color: rgba(239, 68, 68, 0.18); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .pill-yellow { background-color: rgba(245, 158, 11, 0.18); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .pill-blue { background-color: rgba(59, 130, 246, 0.18); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    .pill-gray { background-color: rgba(148, 163, 184, 0.15); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.25); }
    
    .citation-box {
        background-color: rgba(15, 23, 42, 0.6);
        border-left: 3px solid #38bdf8;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        font-size: 0.86rem;
    }
    
    .guardrail-alert {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.15), rgba(15, 23, 42, 0.7));
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fca5a5;
        font-weight: 500;
        margin: 10px 0;
    }
    
    .nudge-alert {
        background: linear-gradient(90deg, rgba(245, 158, 11, 0.15), rgba(15, 23, 42, 0.7));
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fde68a;
        font-weight: 500;
        margin: 10px 0;
    }
    
    /* Footer */
    .disclaimer-footer {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding: 24px 0 12px 0;
        margin-top: 40px;
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Imports from core modules
from data import get_market_store, TICKER_METADATA
from users import load_user_profile, get_all_user_ids, calculate_portfolio_metrics
from agents import run_agents_parallel, run_agents_sequential
from synthesizer import synthesize_recommendation, Recommendation
from evalmetrics import get_backtest_results, get_session_performance_metrics
from db import log_analysis_result, init_db

# Initialize database
init_db()
store = get_market_store()

# --- SIDEBAR CONFIGURATION & FAILURE LAB ---
st.sidebar.markdown("""
<div style="margin-bottom: 4px;">
  <svg width="56" height="56" viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="56" height="56" rx="12" fill="url(#grad)"/>
    <defs>
      <linearGradient id="grad" x1="0" y1="0" x2="56" y2="56" gradientUnits="userSpaceOnUse">
        <stop offset="0%" stop-color="#f97316"/>
        <stop offset="100%" stop-color="#ea580c"/>
      </linearGradient>
    </defs>
    <!-- Candlestick chart -->
    <rect x="11" y="30" width="6" height="14" rx="1.5" fill="white" fill-opacity="0.9"/>
    <rect x="13.5" y="24" width="1" height="6" fill="white" fill-opacity="0.7"/>
    <rect x="13.5" y="44" width="1" height="3" fill="white" fill-opacity="0.7"/>
    <rect x="24" y="18" width="6" height="18" rx="1.5" fill="white" fill-opacity="0.9"/>
    <rect x="26.5" y="13" width="1" height="5" fill="white" fill-opacity="0.7"/>
    <rect x="26.5" y="36" width="1" height="4" fill="white" fill-opacity="0.7"/>
    <rect x="37" y="24" width="6" height="12" rx="1.5" fill="white" fill-opacity="0.9"/>
    <rect x="39.5" y="18" width="1" height="6" fill="white" fill-opacity="0.7"/>
    <rect x="39.5" y="36" width="1" height="3" fill="white" fill-opacity="0.7"/>
    <!-- Trend line -->
    <polyline points="14,37 27,27 40,30" stroke="white" stroke-width="1.5" stroke-opacity="0.5" stroke-dasharray="2,2" fill="none"/>
  </svg>
</div>
""", unsafe_allow_html=True)
st.sidebar.title("⚡ VibeFinance")
st.sidebar.caption("Explainable Multi-Agent Intelligence for NSE Investors")

st.sidebar.markdown("---")
st.sidebar.subheader("👤 Select Investor Profile")
user_list = get_all_user_ids()
user_options = {u["name"]: u["user_id"] for u in user_list}
selected_user_name = st.sidebar.selectbox("Active Account", list(user_options.keys()), index=0)
active_user_id = user_options[selected_user_name]
active_user = load_user_profile(active_user_id)

st.sidebar.markdown(f"""
**Profile:** `{active_user.profile_type}`  
**Risk Tolerance:** `{active_user.risk_score:.2f}`  
**Max Stock Cap:** `{active_user.max_stock_weight*100:.0f}%` | **Sector Cap:** `{active_user.sector_cap*100:.0f}%`  
**Cash:** `₹{active_user.cash_balance:,.0f}`
""")

if active_user.overtrading_flag:
    st.sidebar.warning("⚠️ Behavioral Flag: Overtrading Detected")
if active_user.panic_sell_flag:
    st.sidebar.warning("⚠️ Behavioral Flag: Panic Sell History")

st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Failure Mode Lab")
st.sidebar.caption("Verify graceful degradation across 4 synthetic anomaly scenarios:")
sim_stale = st.sidebar.checkbox("1. Price Feed Stale / Down", value=False, help="Uses 5-day cached data with degraded banner & discounted confidence")
st.sidebar.caption("👉 **2. Missing Filings:** Select ticker `ADANIENT` or `BHARTIARTL`.")
st.sidebar.caption("👉 **3. Agent Signal Conflict:** Select ticker `INFY`.")
sim_timeout = st.sidebar.checkbox("4. Agent Timeout Injection", value=False, help="Forces MomentumAgent to exceed execution timeout")

# 1-Click Scripted Demo Scenario
st.sidebar.markdown("---")
st.sidebar.subheader("🎬 Scripted 2-Min Demo")
st.sidebar.caption("Volume breakout with hidden supply chain filing risk across all 3 profiles:")
run_demo_button = st.sidebar.button("▶️ Run Demo Scenario (TATAMOTORS)", use_container_width=True, type="primary")

# Top Banner for Stale Data Simulation
if sim_stale:
    st.warning("⚠️ **SYSTEM ALERT: Price Feed Degraded** — Primary NSE feed unreachable. Running on cached 5-day snapshot. Agent confidence penalties applied.")

# Tabs
tab1, tab2, tab3 = st.tabs([
    "📊 1. Signals & Multi-Agent Analysis",
    "💼 2. Portfolio & Compare Profiles",
    "🔬 3. Evaluation Lab & Metrics",
])


# ==============================================================================
# TAB 1: SIGNALS & ANALYSIS
# ==============================================================================
with tab1:
    st.markdown("### NSE Universe Market Pulse")
    st.caption("2-year regime-switching GBM with live indicator handoffs and synthetic sentiment.")

    # Universe Summary Table
    summary_df = store.get_universe_summary()
    st.dataframe(
        summary_df,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "1D Change (%)": st.column_config.NumberColumn("1D Return", format="%.2f%%"),
            "20D Return (%)": st.column_config.NumberColumn("20D Momentum", format="%.2f%%"),
            "RSI (14)": st.column_config.ProgressColumn("RSI-14", min_value=0, max_value=100, format="%.1f"),
            "Vol Z-Score": st.column_config.NumberColumn("Vol Z-Score", format="%.2f"),
        },
        hide_index=True,
        width="stretch",
    )

    st.markdown("---")
    col_sel, col_alloc, col_btn = st.columns([3, 2, 2])
    with col_sel:
        target_ticker = st.selectbox(
            "Select Ticker for Multi-Agent Synthesis:",
            store.get_all_tickers(),
            index=8 if run_demo_button else 0,  # Default TATAMOTORS on demo
        )
    with col_alloc:
        target_alloc = st.slider("Target Allocation (%)", min_value=2.0, max_value=25.0, value=5.0, step=1.0) / 100.0
    with col_btn:
        st.write("")
        st.write("")
        analyze_clicked = st.button("🚀 Run Synthesis", use_container_width=True, type="primary")

    if analyze_clicked or run_demo_button:
        st.markdown(f"### 🔍 Deep Dive Multi-Agent Audit: `{target_ticker}` for `{active_user.name}`")
        
        with st.spinner("Dispatching parallel agents (Momentum, Volume, Sentiment, FilingsRAG, Risk)..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            agents_out, par_latency = loop.run_until_complete(
                run_agents_parallel(
                    ticker=target_ticker,
                    user=active_user,
                    simulate_stale=sim_stale,
                    simulate_agent_timeout=sim_timeout,
                    agent_timeout=1.5,
                )
            )
            # Run sequential for latency comparison
            _, seq_latency = loop.run_until_complete(
                run_agents_sequential(ticker=target_ticker, user=active_user, simulate_stale=sim_stale)
            )
            loop.close()

            # Synthesize
            rec = synthesize_recommendation(
                ticker=target_ticker,
                user=active_user,
                agents=agents_out,
                suggested_allocation_pct=target_alloc,
            )

            # Log to DB
            log_analysis_result({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": target_ticker,
                "user_id": active_user.user_id,
                "action": rec.action,
                "position_size_pct": rec.position_size_pct,
                "overall_confidence": rec.overall_confidence,
                "score": rec.score,
                "latency_ms": par_latency,
                "parallel_latency_ms": par_latency,
                "sequential_latency_ms": seq_latency,
                "citation_coverage": rec.citation_coverage,
                "hhi_before": rec.impact_analysis.get("hhi_before", 0),
                "hhi_after": rec.impact_analysis.get("hhi_after", 0),
                "max_sector_weight_before": rec.impact_analysis.get("max_sector_weight_before", 0),
                "max_sector_weight_after": rec.impact_analysis.get("max_sector_weight_after", 0),
                "disagreements_count": len(rec.disagreements),
                "guardrail_triggered": rec.guardrail_triggered or "",
                "nudge_triggered": rec.nudge_triggered or "",
                "details_json": json.dumps({"reasoning": rec.reasoning_chain}),
            })

        # Agent Latency & Status Header
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Parallel Execution", f"{par_latency:.1f} ms", delta=f"{seq_latency:.1f} ms seq", delta_color="inverse")
        c2.metric("Overall Confidence", f"{rec.overall_confidence:.0%}")
        c3.metric("Composite Score", f"{rec.score:+.2f}")
        c4.metric("Citation Coverage", f"{rec.citation_coverage:.1f}%")

        # Action Badge Banner
        pill_class = (
            "pill-green" if rec.action == "Accumulate"
            else ("pill-yellow" if rec.action in ["Watch", "Hold"]
            else "pill-red")
        )
        st.markdown(f"""
        <div class="metric-card" style="border-left: 6px solid {'#10b981' if rec.action == 'Accumulate' else ('#f59e0b' if rec.action in ['Watch', 'Hold'] else '#ef4444')};">
            <h2 style="margin: 0; display: inline-block;">
                <span class="agent-pill {pill_class}" style="font-size: 1.2rem; padding: 6px 16px;">{rec.action.upper()}</span>
                &nbsp;&nbsp;Recommended Allocation: {rec.position_size_pct:.1f}%
            </h2>
            <p style="margin-top: 10px; color: #cbd5e1; font-size: 1.05rem;">{rec.rationale.splitlines()[0]}</p>
        </div>
        """, unsafe_allow_html=True)

        # Show Guardrails & Nudges if triggered
        if rec.guardrail_triggered:
            st.markdown(f"""
            <div class="guardrail-alert">
                🛡️ <b>Guardrail Violation Intercept:</b> {rec.guardrail_triggered}
            </div>
            """, unsafe_allow_html=True)

        if rec.nudge_triggered:
            st.markdown(f"""
            <div class="nudge-alert">
                💡 {rec.nudge_triggered}
            </div>
            """, unsafe_allow_html=True)

        # Specialist Agents Finishing Cards
        st.markdown("#### 🤖 Specialist Agent Consensus")
        agent_cols = st.columns(5)
        for idx, a in enumerate(agents_out):
            with agent_cols[idx]:
                a_color = "pill-green" if a.score > 0.2 else ("pill-red" if a.score < -0.2 else "pill-yellow")
                if a.status == "failed":
                    a_color = "pill-gray"
                st.markdown(f"""
                <div class="metric-card" style="min-height: 220px;">
                    <div style="font-size: 0.82rem; color: #94a3b8; font-weight: 600;">{a.agent}</div>
                    <div style="margin: 6px 0;"><span class="agent-pill {a_color}">{a.label}</span></div>
                    <div style="font-size: 1.1rem; font-weight: 700;">{a.score:+.2f} <span style="font-size: 0.75rem; font-weight: 400; color: #94a3b8;">({a.confidence:.0%})</span></div>
                    <div style="font-size: 0.75rem; color: #64748b; margin-top: 4px;">Latency: {a.latency_ms:.1f}ms | {a.status}</div>
                    <hr style="margin: 8px 0; border-color: rgba(255,255,255,0.06);"/>
                    <div style="font-size: 0.75rem; color: #cbd5e1; line-height: 1.3;">{a.reasoning[0] if a.reasoning else 'No signal'}</div>
                </div>
                """, unsafe_allow_html=True)

        # Disagreements & Conflict Arbitration
        if rec.disagreements:
            st.markdown("#### ⚡ Signal Conflict Arbitration")
            for c in rec.disagreements:
                st.info(f"⚖️ **{c.resolution}**\n\n*(Agent A: {c.agent_a} [{c.score_a:+.2f}] vs Agent B: {c.agent_b} [{c.score_b:+.2f}])*")

        # Full Reasoning Chain & Attribution
        col_reason, col_cites = st.columns([1, 1])
        with col_reason:
            st.markdown("#### 🧠 Transparent Reasoning Chain")
            for step in rec.reasoning_chain:
                st.markdown(f"- {step}")

            with st.expander("Pairwise Agent Alignment Matrix"):
                matrix_df = pd.DataFrame(rec.agent_agreement_matrix).round(2)
                st.dataframe(matrix_df, width="stretch")

        with col_cites:
            st.markdown("#### 📑 Grounded Citations & Evidence Ledger")
            if rec.cited_sources:
                for c in rec.cited_sources:
                    st.markdown(f"""
                    <div class="citation-box">
                        <b>[{c['agent']}]</b> <span style="color: #38bdf8;">{c['source_id']}</span><br/>
                        <span style="color: #94a3b8;">{c['snippet']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No citations available. Claim labelled: uncited / low confidence.")


# ==============================================================================
# TAB 2: PORTFOLIO & COMPARE PROFILES
# ==============================================================================
with tab2:
    st.markdown(f"### Portfolio Risk Structure: `{active_user.name}` ({active_user.profile_type})")
    
    current_prices = {t: store.get_ticker_df(t).iloc[-1]["Close"] for t in store.get_all_tickers()}
    stock_weights, hhi, sector_weights = calculate_portfolio_metrics(active_user, current_prices)

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Total Equity & Cash", f"₹{active_user.total_capital:,.0f}")
    p2.metric("Portfolio HHI", f"{hhi:.0f}", help="<1500: Diversified | 1500-2500: Moderate | >2500: Concentrated")
    p3.metric("Holdings Count", f"{len(active_user.holdings)} stocks")
    p4.metric("Cash Reserve", f"₹{active_user.cash_balance:,.0f} ({active_user.cash_balance/active_user.total_capital*100:.0f}%)")

    col_hold, col_sec = st.columns([3, 2])
    with col_hold:
        st.markdown("#### Current Holdings")
        if active_user.holdings:
            h_data = []
            for h in active_user.holdings:
                curr_p = current_prices.get(h.ticker, h.avg_buy_price)
                mkt_val = h.shares * curr_p
                pnl = ((curr_p - h.avg_buy_price) / h.avg_buy_price) * 100.0
                wt = stock_weights.get(h.ticker, 0.0) * 100.0
                h_data.append({
                    "Ticker": h.ticker,
                    "Sector": h.sector,
                    "Shares": h.shares,
                    "Buy Price (₹)": h.avg_buy_price,
                    "Current (₹)": round(curr_p, 2),
                    "Market Value (₹)": round(mkt_val, 0),
                    "PnL (%)": round(pnl, 2),
                    "Weight (%)": round(wt, 1),
                })
            st.dataframe(pd.DataFrame(h_data), hide_index=True, width="stretch")
        else:
            st.info("No active equity holdings.")

    with col_sec:
        st.markdown("#### Sector Exposures vs Cap")
        sec_df = pd.DataFrame([
            {"Sector": s, "Exposure (%)": round(w * 100, 1), "Cap (%)": round(active_user.sector_cap * 100, 1)}
            for s, w in sector_weights.items()
        ])
        if not sec_df.empty:
            st.bar_chart(sec_df.set_index("Sector")[["Exposure (%)", "Cap (%)"]])

    st.markdown("---")
    st.markdown("### 👥 Compare Profiles: Run SAME Ticker Across All 3 Investors")
    st.caption("Demonstrates how personalized risk profiles, weight tables, and guardrails produce custom decisions for the identical market event.")

    col_cmp_tick, col_cmp_btn = st.columns([3, 1])
    with col_cmp_tick:
        cmp_ticker = st.selectbox("Select Benchmark Ticker to Compare:", store.get_all_tickers(), index=8)  # TATAMOTORS default
    with col_cmp_btn:
        st.write("")
        st.write("")
        cmp_clicked = st.button("🔄 Run 3-Way Comparison", use_container_width=True, type="primary")

    if cmp_clicked or run_demo_button:
        users = [load_user_profile(u["user_id"]) for u in get_all_user_ids()]
        cols = st.columns(3)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for idx, u in enumerate(users):
            u_agents, _ = loop.run_until_complete(run_agents_parallel(cmp_ticker, u))
            u_rec = synthesize_recommendation(cmp_ticker, u, u_agents)

            with cols[idx]:
                p_badge = "pill-green" if u_rec.action == "Accumulate" else ("pill-yellow" if u_rec.action in ["Watch", "Hold"] else "pill-red")
                st.markdown(f"""
                <div class="metric-card">
                    <h4>{u.name}</h4>
                    <div style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px;">{u.profile_type}</div>
                    <div style="margin-bottom: 12px;">
                        <span class="agent-pill {p_badge}" style="font-size: 1rem; padding: 4px 12px;">{u_rec.action.upper()}</span>
                        &nbsp;<b>{u_rec.position_size_pct:.1f}%</b>
                    </div>
                    <div style="font-size: 0.82rem; margin-bottom: 4px;">Score: <b>{u_rec.score:+.2f}</b> | Conf: <b>{u_rec.overall_confidence:.0%}</b></div>
                    <div style="font-size: 0.82rem; margin-bottom: 4px;">Single Stock Cap: <b>{u.max_stock_weight*100:.0f}%</b></div>
                    <div style="font-size: 0.82rem; margin-bottom: 8px;">Sector Cap: <b>{u.sector_cap*100:.0f}%</b></div>
                    <hr style="margin: 8px 0; border-color: rgba(255,255,255,0.08);"/>
                    <div style="font-size: 0.8rem; color: #cbd5e1; line-height: 1.4;">
                        <b>Key Driver:</b> {u_rec.reasoning_chain[1]}
                    </div>
                    {f'<div style="margin-top: 8px; color: #f87171; font-size: 0.78rem;">🛡️ {u_rec.guardrail_triggered}</div>' if u_rec.guardrail_triggered else ''}
                    {f'<div style="margin-top: 8px; color: #fbbf24; font-size: 0.78rem;">⚡ {u_rec.nudge_triggered}</div>' if u_rec.nudge_triggered else ''}
                </div>
                """, unsafe_allow_html=True)
        loop.close()


# ==============================================================================
# TAB 3: EVALUATION & FAILURE LAB
# ==============================================================================
with tab3:
    st.markdown("### 🔬 System Evaluation & Backtest Analytics")
    st.caption("Quantitative audit of directional accuracy, latency speedup, citation coverage, and failure recovery.")

    backtest = get_backtest_results()
    session = get_session_performance_metrics()

    # Section 1: Session Performance
    st.markdown("#### ⏱️ Real-Time Session Execution Benchmarks")
    if session["has_session_data"]:
        st.caption(f"Measured from **{session['total_analyses']} live session analyses** ({session['successful_analyses']} healthy, {session['degraded_analyses']} degraded, {session['failed_analyses']} failed):")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Parallel Execution", f"{session['avg_parallel_latency_ms']} ms", f"Speedup: {session['parallel_speedup']}x vs sequential")
        m2.metric("Sequential Baseline", f"{session['avg_sequential_latency_ms']} ms")
        m3.metric("Citation Coverage", f"{session['avg_citation_coverage_pct']}%", help="Percentage of synthesized claims grounded in citations")
        m4.metric("Guardrails & Conflicts", f"{session['guardrail_triggers']} Intercepts", f"{session['disagreements_resolved']} conflicts arbitrated")
    else:
        st.info("ℹ️ **No live session measurements recorded yet.** Run an analysis in Tab 1 or click 'Run Demo Scenario' to capture real-time measurements in SQLite.")
        with st.expander("📊 View Precomputed Offline Calibration Baseline (N = 14 benchmark runs)", expanded=True):
            b = session["demo_baseline"]
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Parallel Speedup", f"{b['parallel_speedup']}x", f"{b['avg_parallel_latency_ms']}ms par vs {b['avg_sequential_latency_ms']}ms seq")
            b2.metric("Citation Coverage", f"{b['avg_citation_coverage_pct']}%")
            b3.metric("Avg HHI Shift", f"{b['avg_hhi_change']:.0f}")
            b4.metric("Precomputed Runs", f"{b['samples']} analyses")
            st.caption("⚠️ *Note: These figures reflect offline calibration on simulated scenarios, not live session measurements.*")

    st.markdown("---")
    st.markdown("#### 📈 Walk-Forward Backtest (Zero Lookahead on 2-Year Synthetic History)")
    st.caption(
        f"**Dataset:** {backtest['evaluation_dataset']} | "
        f"**Valid Observations:** {backtest['valid_observations']} ({backtest['correct_predictions']} correct, {backtest['incorrect_predictions']} incorrect, {backtest['excluded_samples']} neutral excluded)"
    )

    bt_c1, bt_c2, bt_c3, bt_c4 = st.columns(4)
    bt_c1.metric(
        "30-Day Forward Hit Rate",
        f"{backtest['overall_hit_rate']}%",
        f"{backtest['outperformance_vs_baseline_pct']:+0.1f}% vs Majority Baseline",
    )
    bt_c2.metric("Majority Direction Baseline", f"{backtest['majority_baseline_hit_rate_pct']}%", help="Hit rate from naive majority-class directional prediction")
    bt_c3.metric("Directional Edge", f"{backtest['outperformance_vs_baseline_pct']:+0.1f}%", help="Ensemble hit rate minus naive majority baseline")
    bt_c4.metric("Sample Checkpoints", f"{backtest['valid_observations']} observations", f"{backtest['total_evaluated_checkpoints']} tested")

    col_bt1, col_bt2 = st.columns([1, 1])

    with col_bt1:
        st.markdown("##### Agent-by-Agent Directional Accuracy")
        st.caption("Hit rate vs 30-day forward return across historical checkpoints:")
        acc_data = []
        for ag, d in backtest["agent_accuracy"].items():
            acc_data.append({
                "Component / Agent": ag,
                "Total Calls": d["total_calls"],
                "Directional Hits": d["correct_calls"],
                "Hit Rate (%)": d["hit_rate_pct"],
            })
        st.dataframe(pd.DataFrame(acc_data), hide_index=True, width="stretch")

    with col_bt2:
        st.markdown("##### Ticker Hit Rate Breakdown")
        st.dataframe(
            pd.DataFrame(backtest["ticker_breakdown"]),
            column_config={
                "hit_rate_pct": st.column_config.ProgressColumn("Hit Rate", min_value=0, max_value=100, format="%.1f%%")
            },
            hide_index=True,
            width="stretch",
        )

    st.caption("⚖️ *Methodology Disclosure: Backtest uses rolling indicators strictly recomputed on historical slices with non-overlapping forward return targets. Evaluated on simulated data for research prototyping.*")
    st.markdown("---")

    st.markdown("#### 🧪 Failure Mode Resilience Verification")
    st.markdown("""
    | Failure Scenario | Built-in Mechanism | Status |
    | :--- | :--- | :--- |
    | **1. Price Feed Down** | Falls back to cached snapshot, marks status=`degraded`, renders top banner, penalizes confidence by 25%. | ✅ Handled |
    | **2. Missing Filings** | FilingsRAGAgent outputs `degraded`, cites lack of filings, synthesis proceeds labelled "uncited / low confidence". | ✅ Handled (`ADANIENT`, `BHARTIARTL`) |
    | **3. Agent Signal Conflicts** | Detected via opposing signs with confidence >0.6; arbitrated via user risk weight table. | ✅ Handled (`INFY`) |
    | **4. Agent Timeout / Exception** | Caught via `asyncio.wait_for` timeout wrappers; pipeline finishes with remaining agents. | ✅ Handled (1.5s per-agent timeout) |
    """)

