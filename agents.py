"""
VibeFinance Multi-Agent Engine
Parallelized specialist agents with shared Pydantic v2 contract, per-agent timeouts,
graceful degradation, and evidence attribution.
"""

from typing import Dict, List, Any, Optional, Tuple
import time
import asyncio
from pydantic import BaseModel, Field

from data import get_market_store
from indicators import get_latest_indicators
from users import UserProfile, simulate_allocation_impact
from rag import get_rag_engine


class EvidenceItem(BaseModel):
    source_id: str
    snippet: str
    metric: Optional[str] = None
    value: Optional[Any] = None


class AgentOutput(BaseModel):
    agent: str  # MomentumAgent, VolumeAgent, SentimentAgent, FilingsRAGAgent, RiskAgent
    ticker: str
    label: str
    score: float = Field(ge=-1.0, le=1.0)  # -1.0 (Extreme Bearish/High Risk) to +1.0 (Extreme Bullish/Safe)
    confidence: float = Field(ge=0.0, le=1.0)  # 0.0 to 1.0
    reasoning: List[str]
    evidence: List[Dict[str, Any]]
    status: str = "ok"  # "ok", "degraded", "failed"
    latency_ms: float = 0.0


# 1. MomentumAgent
async def run_momentum_agent(ticker: str, simulate_stale: bool = False, delay_ms: float = 40.0) -> AgentOutput:
    t0 = time.time()
    await asyncio.sleep(delay_ms / 1000.0)  # Measurable delay for parallel benchmark
    
    try:
        store = get_market_store()
        df = store.get_ticker_df(ticker, simulate_stale=simulate_stale)
        ind = get_latest_indicators(df)
        
        rsi = ind.get("rsi_14", 50.0)
        macd = ind.get("macd", 0.0)
        macd_signal = ind.get("macd_signal", 0.0)
        macd_hist = ind.get("macd_hist", 0.0)
        ret_20d = ind.get("ret_20d", 0.0) * 100.0
        close = ind.get("close", 0.0)
        sma20 = ind.get("sma_20", 0.0)
        sma50 = ind.get("sma_50", 0.0)
        bullish_cross = ind.get("bullish_cross", False)
        bearish_cross = ind.get("bearish_cross", False)

        reasoning = []
        evidence = [
            {"source_id": "MOM_RSI14", "snippet": f"RSI-14: {rsi:.1f} (Thresholds: Overbought > 70, Oversold < 30)"},
            {"source_id": "MOM_MACD", "snippet": f"MACD: {macd:.2f}, Signal: {macd_signal:.2f}, Histogram: {macd_hist:.2f}"},
            {"source_id": "MOM_SMA", "snippet": f"Price: ₹{close:.2f} vs SMA20: ₹{sma20:.2f}, SMA50: ₹{sma50:.2f}"},
            {"source_id": "MOM_20D_RET", "snippet": f"20-Day Momentum Return: {ret_20d:+.2f}%"},
        ]

        # Scoring logic
        score = 0.0
        confluence_count = 0

        # RSI contribution
        if rsi >= 65:
            score += 0.35
            confluence_count += 1
            reasoning.append(f"Strong RSI momentum at {rsi:.1f} indicates aggressive buyer dominance.")
        elif rsi >= 52:
            score += 0.20
            confluence_count += 1
            reasoning.append(f"Positive RSI at {rsi:.1f} signals constructive upward bias.")
        elif rsi <= 35:
            score -= 0.35
            confluence_count += 1
            reasoning.append(f"Depressed RSI at {rsi:.1f} highlights strong selling pressure.")
        elif rsi <= 48:
            score -= 0.20
            confluence_count += 1
            reasoning.append(f"Subdued RSI at {rsi:.1f} reflects sluggish demand.")
        else:
            reasoning.append(f"Neutral RSI at {rsi:.1f} indicates market equilibrium.")

        # MACD contribution
        if macd > macd_signal and macd_hist > 0:
            score += 0.30
            confluence_count += 1
            reasoning.append(f"Bullish MACD crossover (Histogram: {macd_hist:+.2f}) confirms accelerating upward momentum.")
        elif macd < macd_signal and macd_hist < 0:
            score -= 0.30
            confluence_count += 1
            reasoning.append(f"Bearish MACD divergence (Histogram: {macd_hist:+.2f}) indicates downward trend acceleration.")

        # Moving Average Trend
        if close > sma20 > sma50:
            score += 0.25
            confluence_count += 1
            reasoning.append("Golden alignment: Price is trading above both ascending 20-day and 50-day SMAs.")
        elif close < sma20 < sma50:
            score -= 0.25
            confluence_count += 1
            reasoning.append("Trend breakdown: Price is trading beneath both declining 20-day and 50-day SMAs.")
        elif bullish_cross:
            score += 0.20
            confluence_count += 1
            reasoning.append("Fresh Golden Cross detected: 20-day SMA crossed above 50-day SMA.")
        elif bearish_cross:
            score -= 0.20
            confluence_count += 1
            reasoning.append("Fresh Death Cross detected: 20-day SMA dropped below 50-day SMA.")

        # 20-day return contribution
        if ret_20d > 8.0:
            score += 0.10
        elif ret_20d < -8.0:
            score -= 0.10

        score = max(-1.0, min(1.0, round(score, 2)))

        # Classification label
        if score >= 0.60:
            label = "Strong Bullish"
        elif score >= 0.20:
            label = "Bullish"
        elif score <= -0.60:
            label = "Strong Bearish"
        elif score <= -0.20:
            label = "Bearish"
        else:
            label = "Neutral"

        # Confidence based on indicator alignment
        confidence = round(0.50 + 0.12 * confluence_count, 2)
        confidence = min(0.95, confidence)

        status = "degraded" if simulate_stale else "ok"
        if simulate_stale:
            confidence = round(confidence * 0.75, 2)
            reasoning.insert(0, "[DEGRADED FEED] Calculated on stale 5-day cached data. Confidence discounted.")

        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent="MomentumAgent",
            ticker=ticker,
            label=label,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            status=status,
            latency_ms=latency_ms,
        )
    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent="MomentumAgent",
            ticker=ticker,
            label="Error / Unavailable",
            score=0.0,
            confidence=0.0,
            reasoning=[f"Momentum computation failed: {str(e)}"],
            evidence=[],
            status="failed",
            latency_ms=latency_ms,
        )


# 2. VolumeAgent
async def run_volume_agent(ticker: str, simulate_stale: bool = False, delay_ms: float = 35.0) -> AgentOutput:
    t0 = time.time()
    await asyncio.sleep(delay_ms / 1000.0)
    
    try:
        store = get_market_store()
        df = store.get_ticker_df(ticker, simulate_stale=simulate_stale)
        ind = get_latest_indicators(df)
        
        vol_z = ind.get("vol_zscore_30d", 0.0)
        volume = ind.get("volume", 0)
        close = ind.get("close", 0.0)
        prev_close = df.iloc[-2]["Close"] if len(df) >= 2 else close
        ret_1d = ((close - prev_close) / prev_close) * 100.0

        evidence = [
            {"source_id": "VOL_ZSCORE_30D", "snippet": f"30-day Volume Z-Score: {vol_z:+.2f} standard deviations"},
            {"source_id": "VOL_DAILY_TURNOVER", "snippet": f"Daily Volume: {volume:,.0f} shares | 1D Price Change: {ret_1d:+.2f}%"},
        ]
        reasoning = []

        # Decision matrix
        if vol_z >= 2.5:
            label = "Unusual Spike"
            if ret_1d > 0.5:
                score = 0.85
                confidence = 0.90
                reasoning.append(f"Institutional Accumulation Breakout: Volume surge of {vol_z:+.1f}σ with positive price jump ({ret_1d:+.2f}%).")
            elif ret_1d < -0.5:
                score = -0.80
                confidence = 0.88
                reasoning.append(f"Heavy Liquidation / Panic Selling: Massive volume spike of {vol_z:+.1f}σ with sharp price drop ({ret_1d:+.2f}%).")
            else:
                score = 0.20
                confidence = 0.65
                reasoning.append(f"Unusual volume churn of {vol_z:+.1f}σ with muted price movement; possible institutional block handoff.")
        elif vol_z >= 1.0:
            if ret_1d > 0:
                label = "Accumulation"
                score = 0.55
                confidence = 0.78
                reasoning.append(f"Above-average volume expansion ({vol_z:+.1f}σ) alongside positive session ({ret_1d:+.2f}%) indicates accumulation.")
            else:
                label = "Distribution"
                score = -0.50
                confidence = 0.75
                reasoning.append(f"Elevated volume ({vol_z:+.1f}σ) on down session ({ret_1d:+.2f}%) signals steady distribution.")
        elif vol_z <= -1.2:
            label = "Low Volume Consolidation"
            score = 0.0
            confidence = 0.55
            reasoning.append(f"Subdued volume ({vol_z:+.1f}σ) indicates low market participation and range-bound price action.")
        else:
            label = "Normal"
            score = 0.10 if ret_1d > 0 else -0.10
            confidence = 0.65
            reasoning.append(f"Trading volume is within normal historical ranges ({vol_z:+.1f}σ vs 30-day rolling mean).")

        status = "degraded" if simulate_stale else "ok"
        if simulate_stale:
            confidence = round(confidence * 0.75, 2)
            reasoning.insert(0, "[DEGRADED FEED] Volume z-scores computed on stale snapshot.")

        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent="VolumeAgent",
            ticker=ticker,
            label=label,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            status=status,
            latency_ms=latency_ms,
        )
    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent="VolumeAgent",
            ticker=ticker,
            label="Error",
            score=0.0,
            confidence=0.0,
            reasoning=[f"Volume analysis error: {str(e)}"],
            evidence=[],
            status="failed",
            latency_ms=latency_ms,
        )


# 3. SentimentAgent
async def run_sentiment_agent(ticker: str, delay_ms: float = 30.0) -> AgentOutput:
    t0 = time.time()
    await asyncio.sleep(delay_ms / 1000.0)
    
    try:
        store = get_market_store()
        headlines = store.get_headlines(ticker)
        
        if not headlines:
            latency_ms = round((time.time() - t0) * 1000, 1)
            return AgentOutput(
                agent="SentimentAgent",
                ticker=ticker,
                label="Neutral",
                score=0.0,
                confidence=0.20,
                reasoning=["No recent news headlines available for ticker. Sentiment unverified."],
                evidence=[],
                status="degraded",
                latency_ms=latency_ms,
            )

        scores = [h["sentiment"] for h in headlines]
        avg_score = round(float(sum(scores) / len(scores)), 2)
        
        # Agreement metric: how many headlines have same sign as the average
        signs = [1 if s > 0.1 else (-1 if s < -0.1 else 0) for s in scores]
        target_sign = 1 if avg_score > 0 else (-1 if avg_score < 0 else 0)
        agree_count = sum(1 for s in signs if s == target_sign)
        agreement_ratio = agree_count / len(signs)
        
        # Confidence derived from consensus
        confidence = round(0.55 + 0.35 * agreement_ratio, 2)
        
        evidence = []
        for h in headlines:
            evidence.append({
                "source_id": f"NEWS_{h['source'].upper().replace(' ', '_')}",
                "snippet": f"[{h['source']} | {h['date']}] \"{h['title']}\" (Polarity: {h['sentiment']:+.2f})",
            })

        reasoning = []
        if avg_score >= 0.40:
            label = "Positive"
            reasoning.append(f"Strongly bullish news cycle (consensus polarity {avg_score:+.2f}). {agree_count}/{len(headlines)} sources report positive operating milestones.")
        elif avg_score >= 0.15:
            label = "Mild Positive"
            reasoning.append(f"Moderately constructive media coverage ({avg_score:+.2f}) with minor counter-narratives.")
        elif avg_score <= -0.40:
            label = "Negative"
            reasoning.append(f"Sharp negative press coverage ({avg_score:+.2f}) highlighting revenue revisions, client cuts, or regulatory scrutiny.")
        elif avg_score <= -0.15:
            label = "Mild Negative"
            reasoning.append(f"Cautious news tone ({avg_score:+.2f}) citing emerging margin pressures.")
        else:
            label = "Neutral"
            reasoning.append(f"Balanced / mixed media sentiment ({avg_score:+.2f}) without directional consensus.")

        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent="SentimentAgent",
            ticker=ticker,
            label=label,
            score=avg_score,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            status="ok",
            latency_ms=latency_ms,
        )
    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent="SentimentAgent",
            ticker=ticker,
            label="Error",
            score=0.0,
            confidence=0.0,
            reasoning=[f"Sentiment agent error: {str(e)}"],
            evidence=[],
            status="failed",
            latency_ms=latency_ms,
        )


# 4. FilingsRAGAgent
async def run_filings_rag_agent(ticker: str, delay_ms: float = 45.0) -> AgentOutput:
    t0 = time.time()
    await asyncio.sleep(delay_ms / 1000.0)
    
    try:
        rag = get_rag_engine()
        result = rag.query_grounded_filings(ticker)

        # Case 1: RAG Subsystem Initialization or Search Failure
        if result.get("status") == "failed":
            latency_ms = round((time.time() - t0) * 1000, 1)
            return AgentOutput(
                agent="FilingsRAGAgent",
                ticker=ticker,
                label=result.get("label", "RAG Init Failed"),
                score=0.0,
                confidence=0.0,
                reasoning=[
                    result.get("summary", "RAG subsystem offline or uninitialized."),
                    "Uncited / ungrounded claim: Recommendation proceeds safely without corporate filing evidence.",
                ],
                evidence=[],
                status="failed",
                latency_ms=latency_ms,
            )

        # Case 2: Healthy RAG Engine, but no filing evidence for ticker (e.g. ADANIENT)
        if result.get("status") == "degraded" or not result.get("evidence"):
            latency_ms = round((time.time() - t0) * 1000, 1)
            return AgentOutput(
                agent="FilingsRAGAgent",
                ticker=ticker,
                label=result.get("label", "No Filing Evidence"),
                score=0.0,
                confidence=0.20,
                reasoning=[
                    result.get("summary", f"No supporting filings or disclosures found for {ticker} in regulatory repository."),
                    "Uncited / low confidence signal: Recommendation cannot lean on regulatory validation.",
                ],
                evidence=[],
                status="degraded",
                latency_ms=latency_ms,
            )

        # Case 3: Grounded retrieval with evidence
        citations = result["citations"]
        evidence = result["evidence"]
        score = result["sentiment_bias"]
        max_sim = result["max_similarity"]
        stale_flag = result["stale_contradiction_detected"]

        # Confidence scaled by retrieval similarity
        confidence = round(min(0.92, max(0.40, max_sim * 1.35)), 2)

        reasoning = []
        if stale_flag:
            confidence = round(confidence * 0.70, 2)
            reasoning.append("⚠️ STALE / CONTRADICTORY FILING DETECTED: Historical guidance contradicts recent disclosures. Confidence penalised.")

        if score >= 0.35:
            label = "Constructive Filings"
            reasoning.append(f"SEBI/Investor disclosures indicate robust order book, balance sheet deleveraging, or capacity expansion.")
        elif score <= -0.35:
            label = "Filing Risk Alert"
            reasoning.append(f"Statutory filings highlight material operational vulnerabilities, export margin headwinds, or regulatory observations.")
        else:
            label = "Neutral Filings"
            reasoning.append("Filings document stable ongoing business operations without major positive or negative outliers.")

        # Add top citation reference
        if citations:
            top_cit = citations[0]
            reasoning.append(f"Primary anchor: {top_cit['source_tag']} ({top_cit['doc_type']}) with similarity {top_cit['similarity']:.2f}")

        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent="FilingsRAGAgent",
            ticker=ticker,
            label=label,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            status="ok",
            latency_ms=latency_ms,
        )
    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent="FilingsRAGAgent",
            ticker=ticker,
            label="RAG Failure",
            score=0.0,
            confidence=0.0,
            reasoning=[f"RAG retrieval exception: {str(e)}"],
            evidence=[],
            status="failed",
            latency_ms=latency_ms,
        )



# 5. RiskAgent (User-Aware)
async def run_risk_agent(
    ticker: str, user: UserProfile, target_allocation_pct: float = 0.05, delay_ms: float = 35.0
) -> AgentOutput:
    t0 = time.time()
    await asyncio.sleep(delay_ms / 1000.0)
    
    try:
        store = get_market_store()
        meta = store.get_metadata(ticker)
        sector = meta.get("sector", "Diversified")
        
        # Calculate concentration impact
        current_prices = {t: store.get_ticker_df(t).iloc[-1]["Close"] for t in store.get_all_tickers()}
        impact = simulate_allocation_impact(
            user=user,
            target_ticker=ticker,
            target_sector=sector,
            allocation_pct=target_allocation_pct,
            current_prices=current_prices,
        )

        reasoning = []
        evidence = [
            {
                "source_id": "RISK_USER_PROFILE",
                "snippet": f"User: {user.name} ({user.profile_type}) | Risk Score: {user.risk_score} | Stock Cap: {user.max_stock_weight*100:.0f}% | Sector Cap: {user.sector_cap*100:.0f}%",
            },
            {
                "source_id": "RISK_CONCENTRATION_HHI",
                "snippet": f"Portfolio HHI: {impact['hhi_before']:.0f} → {impact['hhi_after']:.0f} (Baseline: <1500 Diversified, >2500 Concentrated)",
            },
            {
                "source_id": "RISK_SECTOR_WEIGHT",
                "snippet": f"Sector '{sector}' Exposure: {impact['sector_weight_before']*100:.1f}% → {impact['sector_weight_after']*100:.1f}% (Cap: {user.sector_cap*100:.0f}%)",
            },
            {
                "source_id": "RISK_STOCK_WEIGHT",
                "snippet": f"Stock '{ticker}' Weight: {impact['stock_weight_before']*100:.1f}% → {impact['stock_weight_after']*100:.1f}% (Cap: {user.max_stock_weight*100:.0f}%)",
            },
        ]

        # Rule evaluation
        exceeds_stock = impact["exceeds_stock_cap"]
        exceeds_sector = impact["exceeds_sector_cap"]
        
        if exceeds_stock or exceeds_sector:
            label = "Constraint Violation"
            score = -0.90
            confidence = 0.95
            if exceeds_stock:
                reasoning.append(f"⛔ GUARDRAIL TRIGGERED: Adding {ticker} raises stock weight to {impact['stock_weight_after']*100:.1f}%, breaching user's strict {user.max_stock_weight*100:.0f}% single-stock limit.")
            if exceeds_sector:
                reasoning.append(f"⛔ GUARDRAIL TRIGGERED: Sector exposure for '{sector}' would reach {impact['sector_weight_after']*100:.1f}%, exceeding {user.sector_cap*100:.0f}% sector ceiling.")
        elif impact["sector_weight_before"] == 0.0:
            # Diversification benefit: stock is in a new sector for the user
            label = "Diversification Benefit"
            score = 0.75
            confidence = 0.85
            reasoning.append(f"✅ Favorable diversification: Ticker introduces fresh exposure to sector '{sector}', reducing overall portfolio concentration.")
        elif impact["hhi_after"] > 2500 and user.risk_score < 0.4:
            label = "High Concentration Warning"
            score = -0.45
            confidence = 0.80
            reasoning.append(f"Caution: Portfolio HHI ({impact['hhi_after']:.0f}) remains high for a conservative investor mandate.")
        else:
            label = "Acceptable Risk Profile"
            score = 0.40
            confidence = 0.80
            reasoning.append(f"Portfolio headroom available: Stock weight ({impact['stock_weight_after']*100:.1f}%) and sector weight ({impact['sector_weight_after']*100:.1f}%) remain within defined risk parameters.")

        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent="RiskAgent",
            ticker=ticker,
            label=label,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
            status="ok",
            latency_ms=latency_ms,
        )
    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent="RiskAgent",
            ticker=ticker,
            label="Risk Error",
            score=0.0,
            confidence=0.0,
            reasoning=[f"Risk computation error: {str(e)}"],
            evidence=[],
            status="failed",
            latency_ms=latency_ms,
        )


# --- Orchestrators: Parallel vs Sequential Execution ---

async def _safe_execute(coro, agent_name: str, ticker: str, timeout_sec: float = 2.0) -> AgentOutput:
    t0 = time.time()
    try:
        return await asyncio.wait_for(coro, timeout=timeout_sec)
    except asyncio.TimeoutError:
        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent=agent_name,
            ticker=ticker,
            label="Timeout",
            score=0.0,
            confidence=0.0,
            reasoning=[f"{agent_name} timed out after {timeout_sec:.2f}s execution limit."],
            evidence=[],
            status="failed",
            latency_ms=latency_ms,
        )
    except Exception as e:
        latency_ms = round((time.time() - t0) * 1000, 1)
        return AgentOutput(
            agent=agent_name,
            ticker=ticker,
            label="Execution Exception",
            score=0.0,
            confidence=0.0,
            reasoning=[f"Execution exception in {agent_name}: {str(e)}"],
            evidence=[],
            status="failed",
            latency_ms=latency_ms,
        )


async def _failing_coroutine(exc: Exception):
    """Utility coroutine that raises an injected exception for deterministic fault tests."""
    await asyncio.sleep(0.005)
    raise exc


async def run_agents_parallel(
    ticker: str,
    user: UserProfile,
    simulate_stale: bool = False,
    simulate_agent_timeout: bool = False,
    agent_timeout: float = 2.0,
    agent_timeouts: Optional[Dict[str, float]] = None,
    inject_exceptions: Optional[Dict[str, Exception]] = None,
) -> Tuple[List[AgentOutput], float]:
    """
    Executes all 5 agents concurrently using asyncio.gather with fault isolation.
    Configurable per-agent timeouts and fault injection ensure one failed agent
    never cancels or corrupts surviving agents.
    Returns (outputs, total_parallel_latency_ms).
    """
    t_start = time.time()
    timeouts = agent_timeouts or {}
    injections = inject_exceptions or {}
    
    # If simulate_agent_timeout is toggled for failure lab:
    mom_delay = 3500.0 if simulate_agent_timeout else 40.0

    agent_names = ["MomentumAgent", "VolumeAgent", "SentimentAgent", "FilingsRAGAgent", "RiskAgent"]

    def get_coro(name: str):
        if name in injections:
            return _failing_coroutine(injections[name])
        if name == "MomentumAgent":
            return run_momentum_agent(ticker, simulate_stale=simulate_stale, delay_ms=mom_delay)
        elif name == "VolumeAgent":
            return run_volume_agent(ticker, simulate_stale=simulate_stale)
        elif name == "SentimentAgent":
            return run_sentiment_agent(ticker)
        elif name == "FilingsRAGAgent":
            return run_filings_rag_agent(ticker)
        elif name == "RiskAgent":
            return run_risk_agent(ticker, user)
        raise ValueError(f"Unknown agent: {name}")

    tasks = [
        _safe_execute(get_coro(name), name, ticker, timeouts.get(name, agent_timeout))
        for name in agent_names
    ]

    # return_exceptions=True guarantees that an unhandled exception in one coroutine cannot cancel others
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    final_outputs: List[AgentOutput] = []
    for idx, res in enumerate(results):
        if isinstance(res, Exception):
            agent_name = agent_names[idx]
            final_outputs.append(
                AgentOutput(
                    agent=agent_name,
                    ticker=ticker,
                    label="Failed",
                    score=0.0,
                    confidence=0.0,
                    reasoning=[f"Unhandled exception in task wrapper: {str(res)}"],
                    evidence=[],
                    status="failed",
                    latency_ms=0.0,
                )
            )
        else:
            final_outputs.append(res)

    total_ms = round((time.time() - t_start) * 1000, 1)
    return final_outputs, total_ms


async def run_agents_sequential(
    ticker: str, user: UserProfile, simulate_stale: bool = False
) -> Tuple[List[AgentOutput], float]:
    """
    Runs all 5 agents one-by-one to benchmark the parallel speedup.
    """
    t_start = time.time()
    results = [
        await run_momentum_agent(ticker, simulate_stale=simulate_stale),
        await run_volume_agent(ticker, simulate_stale=simulate_stale),
        await run_sentiment_agent(ticker),
        await run_filings_rag_agent(ticker),
        await run_risk_agent(ticker, user),
    ]
    total_ms = round((time.time() - t_start) * 1000, 1)
    return results, total_ms

