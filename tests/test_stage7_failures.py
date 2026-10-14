"""
Stage 7 verification tests: 4 Failure Modes and Degraded Resilience.
"""

import pytest
from users import load_user_profile
from agents import run_agents_parallel, run_momentum_agent, run_filings_rag_agent
from synthesizer import synthesize_recommendation


@pytest.mark.asyncio
async def test_failure_mode_1_price_feed_down():
    """Verify that when price feed is down/stale, agents mark degraded and confidence is discounted."""
    user = load_user_profile("user_balanced")
    outputs, _ = await run_agents_parallel("RELIANCE", user, simulate_stale=True)
    
    mom = next(o for o in outputs if o.agent == "MomentumAgent")
    vol = next(o for o in outputs if o.agent == "VolumeAgent")
    
    assert mom.status == "degraded"
    assert vol.status == "degraded"
    assert "[DEGRADED FEED]" in mom.reasoning[0]
    
    rec = synthesize_recommendation("RELIANCE", user, outputs)
    assert rec.overall_confidence < 0.90


@pytest.mark.asyncio
async def test_failure_mode_2_missing_filing():
    """Verify that tickers with NO filing (ADANIENT) degrade gracefully without crashing."""
    user = load_user_profile("user_conservative")
    outputs, _ = await run_agents_parallel("ADANIENT", user)
    
    rag = next(o for o in outputs if o.agent == "FilingsRAGAgent")
    assert rag.status == "degraded"
    assert rag.confidence <= 0.30
    
    rec = synthesize_recommendation("ADANIENT", user, outputs)
    assert "uncited / low confidence" in rec.rationale or rag.label == "No Filing Evidence"


@pytest.mark.asyncio
async def test_failure_mode_3_conflicting_agents():
    """Verify that engineered conflict ticker (INFY) triggers conflict arbitration."""
    user = load_user_profile("user_balanced")
    outputs, _ = await run_agents_parallel("INFY", user)
    rec = synthesize_recommendation("INFY", user, outputs)
    
    # Assert arbitration step is documented in reasoning chain
    has_arbitration = any("Arbitration Step" in step for step in rec.reasoning_chain)
    assert has_arbitration


@pytest.mark.asyncio
async def test_failure_mode_4_agent_timeout_and_recovery():
    """Verify that when an agent times out, the pipeline finishes with remaining agents."""
    user = load_user_profile("user_aggressive")
    outputs, par_ms = await run_agents_parallel("TATAMOTORS", user, simulate_agent_timeout=True, agent_timeout=0.20)
    
    mom = next(o for o in outputs if o.agent == "MomentumAgent")
    assert mom.status == "failed"
    
    # 4 remaining agents should still be ok
    ok_agents = [o for o in outputs if o.status == "ok"]
    assert len(ok_agents) >= 4
    
    # Synthesis must still succeed!
    rec = synthesize_recommendation("TATAMOTORS", user, outputs)
    assert rec.action is not None
    assert rec.citation_coverage > 0
