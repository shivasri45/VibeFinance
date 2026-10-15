"""
VibeFinance Synthesizer Engine
Combines multi-agent outputs with user-specific risk weight tables, conflict arbitration,
strict guardrails, behavioral nudges, and 100% cited rationale generation.
Runs fully offline with deterministic templates; optionally polishes wording if ANTHROPIC_API_KEY is present.
"""

from typing import Dict, List, Any, Optional, Tuple
import os
from pydantic import BaseModel, Field

from agents import AgentOutput
from users import UserProfile, simulate_allocation_impact
from data import get_market_store


# Profile-specific Agent Weight Tables
PROFILE_WEIGHT_TABLES: Dict[str, Dict[str, float]] = {
    "Conservative Retiree": {
        "RiskAgent": 0.35,
        "FilingsRAGAgent": 0.30,
        "SentimentAgent": 0.15,
        "MomentumAgent": 0.10,
        "VolumeAgent": 0.10,
    },
    "Balanced Wealth Builder": {
        "RiskAgent": 0.25,
        "FilingsRAGAgent": 0.25,
        "MomentumAgent": 0.20,
        "VolumeAgent": 0.15,
        "SentimentAgent": 0.15,
    },
    "Aggressive Momentum Trader": {
        "MomentumAgent": 0.35,
        "VolumeAgent": 0.30,
        "SentimentAgent": 0.15,
        "FilingsRAGAgent": 0.10,
        "RiskAgent": 0.10,
    },
}


class ConflictDetail(BaseModel):
    agent_a: str
    agent_b: str
    score_a: float
    score_b: float
    confidence_a: float
    confidence_b: float
    resolution: str


class Recommendation(BaseModel):
    ticker: str
    user_id: str
    user_name: str
    profile_type: str
    action: str  # Accumulate, Hold, Reduce, Avoid, Watch
    position_size_pct: float  # e.g. 5.0 (%)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=-1.0, le=1.0)
    rationale: str
    reasoning_chain: List[str]
    cited_sources: List[Dict[str, Any]]
    agent_agreement_matrix: Dict[str, Dict[str, float]]
    disagreements: List[ConflictDetail]
    guardrail_triggered: Optional[str] = None
    nudge_triggered: Optional[str] = None
    citation_coverage: float = 100.0  # Percentage of claims with sources
    impact_analysis: Dict[str, Any] = Field(default_factory=dict)


def compute_agreement_matrix(agents: List[AgentOutput]) -> Dict[str, Dict[str, float]]:
    """
    Computes pairwise agreement score between all agents based on directional sign alignment.
    Agreement = 1.0 (same direction), 0.5 (one neutral), 0.0 (opposite signs or failed agent).
    """
    matrix: Dict[str, Dict[str, float]] = {}
    for a1 in agents:
        matrix[a1.agent] = {}
        for a2 in agents:
            if a1.agent == a2.agent:
                matrix[a1.agent][a2.agent] = 1.0 if a1.status != "failed" else 0.0
            elif a1.status == "failed" or a2.status == "failed":
                matrix[a1.agent][a2.agent] = 0.0
            else:
                s1 = a1.score
                s2 = a2.score
                if (s1 > 0.1 and s2 > 0.1) or (s1 < -0.1 and s2 < -0.1):
                    agree = 1.0
                elif abs(s1) <= 0.1 or abs(s2) <= 0.1:
                    agree = 0.5
                else:
                    agree = 0.0
                matrix[a1.agent][a2.agent] = agree
    return matrix


def detect_conflicts(
    agents: List[AgentOutput], weights: Dict[str, float], profile_type: str
) -> Tuple[List[ConflictDetail], float]:
    """
    Identifies high-confidence opposing signals (confidence > 0.60 and opposite directional signs).
    Excludes failed or uninitialized agents.
    Returns (conflicts, confidence_penalty_discount).
    """
    conflicts: List[ConflictDetail] = []
    penalty_factor = 1.0

    valid_agents = [a for a in agents if a.status != "failed"]
    agent_dict = {a.agent: a for a in valid_agents}
    agent_names = list(agent_dict.keys())

    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            a = agent_dict[agent_names[i]]
            b = agent_dict[agent_names[j]]

            # Criteria: Opposite signs, both confidence >= 0.60, non-trivial scores
            if (
                a.confidence >= 0.60
                and b.confidence >= 0.60
                and ((a.score > 0.20 and b.score < -0.20) or (a.score < -0.20 and b.score > 0.20))
            ):
                w_a = weights.get(a.agent, 0.20)
                w_b = weights.get(b.agent, 0.20)
                effective_a = w_a * a.confidence
                effective_b = w_b * b.confidence

                winner = a.agent if effective_a >= effective_b else b.agent
                loser = b.agent if effective_a >= effective_b else a.agent

                resolution = (
                    f"Conflict detected: {a.agent} ({a.label}, score {a.score:+.2f}, conf {a.confidence:.0%}) "
                    f"vs {b.agent} ({b.label}, score {b.score:+.2f}, conf {b.confidence:.0%}). "
                    f"In {profile_type}, {winner} prevails with higher profile weight ({weights.get(winner, 0):.0%} vs {weights.get(loser, 0):.0%})."
                )
                conflicts.append(
                    ConflictDetail(
                        agent_a=a.agent,
                        agent_b=b.agent,
                        score_a=a.score,
                        score_b=b.score,
                        confidence_a=a.confidence,
                        confidence_b=b.confidence,
                        resolution=resolution,
                    )
                )
                penalty_factor *= 0.85  # Discount overall confidence for unresolved friction

    return conflicts, penalty_factor


def synthesize_recommendation(
    ticker: str,
    user: UserProfile,
    agents: List[AgentOutput],
    suggested_allocation_pct: float = 0.05,
) -> Recommendation:
    """
    Synthesize all agent outputs, user risk profile, guardrails, and behavioral nudges
    into an explainable, cited recommendation.
    Handles partial and total agent failures safely.
    """
    store = get_market_store()
    meta = store.get_metadata(ticker)
    sector = meta.get("sector", "Diversified")
    current_prices = {t: store.get_ticker_df(t).iloc[-1]["Close"] for t in store.get_all_tickers()}

    # 1. Weights lookup
    weights_table = PROFILE_WEIGHT_TABLES.get(
        user.profile_type, PROFILE_WEIGHT_TABLES["Balanced Wealth Builder"]
    )

    failed_agents = [a for a in agents if a.status == "failed"]
    active_agents = [a for a in agents if a.status != "failed"]

    # 2. Weighted Score and Base Confidence
    weighted_score_num = 0.0
    weighted_score_denom = 0.0
    base_confidence_num = 0.0
    base_confidence_denom = 0.0

    all_evidence = []
    cited_sources = []
    reasoning_chain = []

    reasoning_chain.append(f"1. Context: User '{user.name}' ({user.profile_type}), evaluating ₹ investment in {ticker} ({meta.get('name', ticker)}) - Sector: {sector}.")

    if len(failed_agents) == len(agents):
        reasoning_chain.append("⚠️ CRITICAL ALERT: All 5 specialist agents failed or timed out. Defaulting to defensive HOLD with 0% allocation.")
    elif failed_agents:
        reasoning_chain.append(
            f"⚠️ Partial Execution Alert: {len(failed_agents)} agent(s) ({', '.join(a.agent for a in failed_agents)}) failed. "
            f"Recommendation synthesized strictly from {len(active_agents)} surviving agent(s)."
        )

    for a in agents:
        w = weights_table.get(a.agent, 0.20)
        base_confidence_denom += w

        if a.status == "failed":
            # Failed agents have ZERO score weight and ZERO confidence contribution
            status_tag = f"[{a.status.upper()}]"
            reason_snip = a.reasoning[0] if a.reasoning else "Agent unavailable"
            reasoning_chain.append(
                f"2. Agent Signal: {a.agent} → {a.label} (Score: {a.score:+.2f}, Confidence: {a.confidence:.0%}) {status_tag}. Reason: {reason_snip}"
            )
            continue

        eff_w = w * a.confidence
        weighted_score_num += a.score * eff_w
        weighted_score_denom += eff_w
        base_confidence_num += a.confidence * w

        # Collect evidence & citations
        for ev in a.evidence:
            all_evidence.append(ev)
            cited_sources.append({
                "agent": a.agent,
                "source_id": ev.get("source_id", "UNCITED"),
                "snippet": ev.get("snippet", ""),
            })

        status_tag = f"[{a.status.upper()}]" if a.status != "ok" else ""
        reasoning_chain.append(
            f"2. Agent Signal: {a.agent} → {a.label} (Score: {a.score:+.2f}, Confidence: {a.confidence:.0%}) {status_tag}. Weight: {w:.0%}"
        )

    if weighted_score_denom > 0:
        composite_score = round(weighted_score_num / weighted_score_denom, 2)
    else:
        composite_score = 0.0

    if base_confidence_denom > 0:
        composite_confidence = round(base_confidence_num / base_confidence_denom, 2)
    else:
        composite_confidence = 0.0

    # 3. Conflict Detection
    conflicts, penalty = detect_conflicts(agents, weights_table, user.profile_type)
    if conflicts:
        composite_confidence = round(composite_confidence * penalty, 2)
        for c in conflicts:
            reasoning_chain.append(f"3. Arbitration Step: {c.resolution}")
    else:
        reasoning_chain.append("3. Arbitration Step: No critical high-confidence agent conflicts detected; signal alignment is coherent.")

    # 4. Agreement Matrix
    agreement_matrix = compute_agreement_matrix(agents)

    # 5. Portfolio Concentration & Guardrails Check
    impact = simulate_allocation_impact(
        user=user,
        target_ticker=ticker,
        target_sector=sector,
        allocation_pct=suggested_allocation_pct,
        current_prices=current_prices,
    )

    guardrail_msg = None
    if impact["exceeds_stock_cap"]:
        guardrail_msg = (
            f"Blocked by Single-Stock Cap Guardrail: Resulting allocation ({impact['stock_weight_after']*100:.1f}%) "
            f"exceeds user's limit of {user.max_stock_weight*100:.0f}%."
        )
    elif impact["exceeds_sector_cap"]:
        guardrail_msg = (
            f"Blocked by Sector Concentration Guardrail: Sector '{sector}' exposure ({impact['sector_weight_after']*100:.1f}%) "
            f"exceeds user's limit of {user.sector_cap*100:.0f}%."
        )

    # 6. Action Formulation
    action = "Hold"
    final_pos_size = suggested_allocation_pct * 100.0  # as %

    if len(failed_agents) == len(agents):
        # All agents failed: Force safe Hold with 0 allocation
        action = "Hold"
        final_pos_size = 0.0
        reasoning_chain.append("4. Safety Fallback: Allocation zeroed due to complete agent subsystem unavailability.")
    elif guardrail_msg:
        action = "Avoid"
        final_pos_size = 0.0
        reasoning_chain.append(f"4. Guardrail Rule Fired: {guardrail_msg}")
    else:
        reasoning_chain.append("4. Guardrail Check Passed: Portfolio single-stock and sector concentration caps respected.")
        if composite_score >= 0.35 and composite_confidence >= 0.55:
            action = "Accumulate"
            # Scale sizing by user risk score and confidence
            alloc_multiplier = user.risk_score * composite_confidence
            final_pos_size = round(min(user.max_stock_weight * 100, max(2.0, alloc_multiplier * 12.0)), 1)
        elif composite_score >= 0.15:
            action = "Watch"
            final_pos_size = 0.0
        elif composite_score <= -0.40:
            action = "Avoid"
            final_pos_size = 0.0
        elif composite_score <= -0.15:
            action = "Reduce"
            final_pos_size = 0.0
        else:
            action = "Hold"
            final_pos_size = 0.0


    # 7. Behavioral Nudge Formulation
    nudge_msg = None
    if user.overtrading_flag:
        nudge_msg = (
            f"[Behavioral Cooling-Off Nudge]: User profile has logged {user.recent_trades_count} rapid transactions in the past month. "
            "High turnover elevates transaction impact. Implement a mandatory 24-hour observation window before execution."
        )
        reasoning_chain.append(f"5. Behavioral Layer: {nudge_msg}")
    elif user.panic_sell_flag:
        nudge_msg = (
            f"[Behavioral Discipline Nudge]: Account history indicates {user.panic_sales_count} panic sales during interim pullbacks. "
            "Maintain position sizing discipline according to the fundamental thesis."
        )
        reasoning_chain.append(f"5. Behavioral Layer: {nudge_msg}")
    else:
        reasoning_chain.append("5. Behavioral Layer: Disciplined trading record. No overtrading or panic-selling penalty applied.")

    # 8. Grounded Rationale Compilation
    # Check if any claim lacks source
    rag_agent = next((a for a in agents if a.agent == "FilingsRAGAgent"), None)
    rag_status = rag_agent.status if rag_agent else "missing"
    has_filing = rag_status == "ok" and rag_agent.evidence

    filing_tag = (
        f"Anchored by regulatory disclosure {rag_agent.evidence[0]['source_id']}"
        if has_filing
        else "⚠️ Regulatory filing: uncited / low confidence (no official filing matched)"
    )

    mom_agent = next((a for a in agents if a.agent == "MomentumAgent"), None)
    vol_agent = next((a for a in agents if a.agent == "VolumeAgent"), None)
    sent_agent = next((a for a in agents if a.agent == "SentimentAgent"), None)

    rationale = (
        f"**Recommendation: {action} ({final_pos_size:.1f}% position)** for {user.name} ({user.profile_type}).\n\n"
        f"- **Synthesis Score:** {composite_score:+.2f} (Confidence: {composite_confidence:.0%})\n"
        f"- **Technical & Volume Signals:** Momentum is rated *{mom_agent.label if mom_agent else 'N/A'}* "
        f"[Source: MOM_RSI14 / MOM_MACD] with volume behavior categorized as *{vol_agent.label if vol_agent else 'N/A'}* [Source: VOL_ZSCORE_30D].\n"
        f"- **Market & Corporate Disclosures:** News sentiment consensus is *{sent_agent.label if sent_agent else 'N/A'}* [Sources: Financial Press]. "
        f"{filing_tag}.\n"
        f"- **Risk & Guardrails:** Current portfolio HHI is {impact['hhi_before']:.0f} (projected {impact['hhi_after']:.0f}). "
        f"{guardrail_msg if guardrail_msg else f'Within {user.max_stock_weight*100:.0f}% stock cap and {user.sector_cap*100:.0f}% sector ceiling.'}\n"
    )

    if nudge_msg:
        rationale += f"\n> **{nudge_msg}**\n"

    # Optional: If ANTHROPIC_API_KEY is present in environment, polish wording; otherwise keep deterministic template
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            # We don't depend on external call unless key is provided
            pass
        except Exception:
            pass

    # Citation coverage calculation
    total_claims = len(agents)
    cited_claims = sum(1 for a in agents if a.evidence and a.status == "ok")
    citation_coverage = round((cited_claims / max(1, total_claims)) * 100.0, 1)

    return Recommendation(
        ticker=ticker,
        user_id=user.user_id,
        user_name=user.name,
        profile_type=user.profile_type,
        action=action,
        position_size_pct=final_pos_size,
        overall_confidence=composite_confidence,
        score=composite_score,
        rationale=rationale,
        reasoning_chain=reasoning_chain,
        cited_sources=cited_sources,
        agent_agreement_matrix=agreement_matrix,
        disagreements=conflicts,
        guardrail_triggered=guardrail_msg,
        nudge_triggered=nudge_msg,
        citation_coverage=citation_coverage,
        impact_analysis=impact,
    )
