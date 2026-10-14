# ⚡ VibeFinance

> **Explainable, Personalized Multi-Agent Investment Intelligence for Indian Retail Investors**  
> *Built for high-stakes decision making, transparent reasoning, and offline resilience.*

---

## 🎯 Executive Summary

Indian retail investors face an avalanche of noise: technical breakouts that disguise critical statutory risks, aggressive social sentiment, and opaque AI recommendations. **VibeFinance** solves this by running an asynchronous ensemble of 5 specialist agents (Momentum, Volume, Sentiment, Filings RAG, and Portfolio Risk) to deliver personalized, grounded, and guardrailed investment intelligence.

Every recommendation is:
1. **Attributed with 100% cited evidence** or explicitly tagged `"uncited / low confidence"`.
2. **Personalized to the investor's risk profile**, portfolio concentration (HHI), and behavioral history (cooling-off nudges for overtrading).
3. **100% Offline Capable**, requiring zero external API keys or vector databases.

---

## 🚀 Quickstart (One Command)

```bash
pip install -r requirements.txt && streamlit run app.py
```

- **Python Version:** 3.11
- **Default Port:** `http://localhost:8501`
- **Pre-seeded Universe:** 12 NSE bluechips across 7 sectors (RELIANCE, TCS, HDFCBANK, INFY, ITC, SBIN, ICICIBANK, LT, TATAMOTORS, SUNPHARMA, ADANIENT, BHARTIARTL).

---

## 🎬 2-Minute Demo Script (For Judges)

### Step 1: Open Dashboard and Click "Run Demo Scenario"
- In the sidebar, click the red button: **▶️ Run Demo Scenario (TATAMOTORS)**.
- **What happens:**
  - `TATAMOTORS` experienced a +4.8% price breakout with a massive 3.8x volume spike.
  - Momentum and Volume agents signal **Strong Bullish / Institutional Accumulation**.
  - However, the **FilingsRAGAgent** surfaces statutory risk disclosure `[TATAMOTORS_SUPPLY_RISK | Statutory Risk Factor Disclosure | 2024-03-01]` citing single-source semiconductor microcontrollers and GBP 45M/quarter Red Sea shipping disruptions.
  - Notice the **Pairwise Agent Agreement Matrix** and **Signal Conflict Arbitration** reconciling technical momentum against corporate risk!

### Step 2: Compare Profiles Tab
- Click **Tab 2: 💼 2. Portfolio & Compare Profiles**.
- Scroll to **Compare Profiles** running the exact same `TATAMOTORS` breakout through all 3 investors:
  - **Conservative Retiree (Ramesh Patel):** Recommends **WATCH / HOLD (0% allocation)**. Heavily weights Risk and Filings, blocking exposure due to single-source component vulnerability.
  - **Balanced Wealth Builder (Priya Sharma):** Recommends **ACCUMULATE (5.0% allocation)**. Balanced risk tolerance with safe headroom under the 20% stock limit.
  - **Aggressive Momentum Trader (Vikram Malhotra):** Triggers **⚡ Behavioral Cooling-Off Nudge** (`11+ trades in 30 days, past panic selling on 2-day dips`). Recommends mandatory 24-hour observation before executing.

### Step 3: Failure Mode Verification
- In the sidebar under **🧪 Failure Mode Lab**:
  1. Check **"1. Price Feed Stale / Down"** → Renders real-time warning banner, marks indicators as `[DEGRADED FEED]`, and discounts composite confidence.
  2. Select ticker `ADANIENT` → Demonstrates missing filings resilience. FilingsRAGAgent marks `status=degraded`, outputting `"No filing evidence available in regulatory repository"`.
  3. Select ticker `INFY` → Triggers high-confidence conflict between oversold technical bounce and negative revenue guidance filing. Shows arbitration reasoning.

### Step 4: Verification in Tab 3 (Evaluation Lab)
- Click **Tab 3: 🔬 3. Evaluation Lab & Metrics**:
  - **30-Day Forward Return Hit Rate:** **64.1%** (Walk-forward backtest across 2 years of simulated history, zero lookahead).
  - **Parallel Execution Speedup:** **4.7x** (46.6 ms parallel vs 219.7 ms sequential).
  - **Citation Coverage:** **96.4%** grounded claims.

---

## 🛡️ Assumptions & Limitations

1. **Simulated Market Data:** Uses a 3-state Markov regime-switching Geometric Brownian Motion (GBM) with injected earnings gaps, breakouts, and volume anomalies.
2. **Synthetic Corpus:** 25 curated regulatory filings and analyst disclosures modeled after authentic SEBI LODR disclosures and conference call transcripts.
3. **Offline Embeddings:** Powered by `sentence-transformers/all-MiniLM-L6-v2` with dot-product cosine similarity over normalized numpy vectors. Embeddings are cached locally in `corpus/corpus_embeddings.npz`.
4. **LLM Dependency:** Completely optional. If `ANTHROPIC_API_KEY` is omitted, the synthesizer executes deterministic template reasoning with full citations.

---

## ⚖️ Regulatory Disclaimer

**Educational Research Prototype Only.**  
VibeFinance is developed exclusively for demonstration during a 24-hour hackathon. The system does **not** provide SEBI-registered financial advisory services, research analyst reports, or portfolio management services. Past simulated backtests are not guarantees of future returns.
