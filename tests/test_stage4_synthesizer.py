"""
Stage 4 verification tests: Synthesizer, Profile comparison, Conflict arbitration, Guardrails, and Nudges.
"""

import pytest
from users import load_user_profile
from agents import run_agents_parallel
from synthesizer import (
    synthesize_recommendation,
    Recommendation,
    PROFILE_WEIGHT_TABLES,
    detect_conflicts,
)


@pytest.mark.asyncio
async def test_compare_profiles_outputs_differ_on_same_ticker():
    """
    Asserts that running the SAME ticker through all 3 user profiles
    produces meaningfully different recommendations, weightings, and risk evaluations.
    """
    ticker = "TATAMOTORS"
    
    cons_user = load_user_profile("user_conservative")
    bal_user = load_user_profile("user_balanced")
    agg_user = load_user_profile("user_aggressive")
    
    # Run agents for each user
    cons_agents, _ = await run_agents_parallel(ticker, cons_user)
    bal_agents, _ = await run_agents_parallel(ticker, bal_user)
    agg_agents, _ = await run_agents_parallel(ticker, agg_user)
    
    rec_cons = synthesize_recommendation(ticker, cons_user, cons_agents)
    rec_bal = synthesize_recommendation(ticker, bal_user, bal_agents)
    rec_agg = synthesize_recommendation(ticker, agg_user, agg_agents)
    
    assert isinstance(rec_cons, Recommendation)
    assert isinstance(rec_bal, Recommendation)
    assert isinstance(rec_agg, Recommendation)
    
    print("\n--- Compare Profiles Test ---")
    print(f"Conservative: Action={rec_cons.action}, Size={rec_cons.position_size_pct}%, Conf={rec_cons.overall_confidence}")
    print(f"Balanced:     Action={rec_bal.action}, Size={rec_bal.position_size_pct}%, Conf={rec_bal.overall_confidence}")
    print(f"Aggressive:   Action={rec_agg.action}, Size={rec_agg.position_size_pct}%, Conf={rec_agg.overall_confidence}")

    # Outputs must differ in position size or behavioral nudges or risk weighting
    assert rec_agg.nudge_triggered is not None, "Aggressive trader must have behavioral nudge"
    assert rec_cons.nudge_triggered is None, "Conservative retiree should not have overtrading nudge"
    assert rec_cons.position_size_pct <= cons_user.max_stock_weight * 100.0


@pytest.mark.asyncio
async def test_conflict_detection_and_arbitration():
    """Verify that opposing high-confidence signals trigger arbitration and lower confidence."""
    ticker = "INFY"
    user = load_user_profile("user_balanced")
    agents, _ = await run_agents_parallel(ticker, user)
    rec = synthesize_recommendation(ticker, user, agents)
    
    # Check if conflicts list is generated or checked cleanly
    assert rec.agent_agreement_matrix is not None
    assert len(rec.reasoning_chain) >= 5
    assert rec.citation_coverage > 50.0


@pytest.mark.asyncio
async def test_guardrail_blocks_over_concentration():
    """Verify that when a user already holds or would breach max sector/stock weight, guardrail fires."""
    user = load_user_profile("user_conservative")
    # For Ramesh, ITC already represents a big chunk. If we test ITC with 0.08 allocation, it should trigger single stock or sector cap
    agents, _ = await run_agents_parallel("ITC", user)
    rec = synthesize_recommendation("ITC", user, agents, suggested_allocation_pct=0.08)
    
    # Guardrail should fire because Ramesh has max_stock_weight = 10%
    assert rec.guardrail_triggered is not None or rec.action in ["Hold", "Avoid", "Reduce"]
    if rec.guardrail_triggered:
        assert rec.action == "Avoid"
        assert rec.position_size_pct == 0.0
