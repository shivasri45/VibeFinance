"""
Stage 7 verification tests: 8 Failure Modes, Anomaly Handling, and Graceful Degradation.
"""

import pytest
from pathlib import Path
from users import load_user_profile
from agents import (
    run_agents_parallel,
    run_momentum_agent,
    run_filings_rag_agent,
    AgentOutput,
)
from rag import FilingsRAGEngine, reset_rag_engine
from synthesizer import synthesize_recommendation, Recommendation


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
    """Verify that tickers with NO filing (ADANIENT) degrade gracefully with zero fabricated citations."""
    user = load_user_profile("user_conservative")
    outputs, _ = await run_agents_parallel("ADANIENT", user)
    
    rag = next(o for o in outputs if o.agent == "FilingsRAGAgent")
    assert rag.status == "degraded"
    assert rag.label == "No Filing Evidence"
    assert rag.confidence <= 0.30
    assert len(rag.evidence) == 0, "Missing filing must produce ZERO fabricated evidence items"
    
    rec = synthesize_recommendation("ADANIENT", user, outputs)
    assert "uncited / low confidence" in rec.rationale or rag.label == "No Filing Evidence"
    assert rec.action is not None


@pytest.mark.asyncio
async def test_failure_mode_3_conflicting_agents():
    """Verify that engineered conflict ticker (INFY) triggers conflict arbitration and penalizes confidence."""
    user = load_user_profile("user_balanced")
    outputs, _ = await run_agents_parallel("INFY", user)
    rec = synthesize_recommendation("INFY", user, outputs)
    
    # Assert arbitration step is documented in reasoning chain
    has_arbitration = any("Arbitration Step" in step for step in rec.reasoning_chain)
    assert has_arbitration
    assert rec.disagreements is not None


@pytest.mark.asyncio
async def test_failure_mode_4_agent_timeout_and_recovery():
    """Verify that when an agent times out, surviving agents complete and synthesis succeeds."""
    user = load_user_profile("user_aggressive")
    outputs, par_ms = await run_agents_parallel(
        "TATAMOTORS",
        user,
        simulate_agent_timeout=True,
        agent_timeouts={"MomentumAgent": 0.15},
    )
    
    mom = next(o for o in outputs if o.agent == "MomentumAgent")
    assert mom.status == "failed"
    assert mom.label == "Timeout"
    assert mom.confidence == 0.0
    assert len(mom.evidence) == 0
    
    # 4 remaining agents should still be ok
    ok_agents = [o for o in outputs if o.status == "ok"]
    assert len(ok_agents) == 4
    
    # Synthesis must still succeed and document partial execution
    rec = synthesize_recommendation("TATAMOTORS", user, outputs)
    assert rec.action is not None
    assert any("Partial Execution Alert" in step for step in rec.reasoning_chain)


@pytest.mark.asyncio
async def test_failure_mode_5_unexpected_agent_exception():
    """Verify that an unexpected exception in one agent does not cancel or corrupt surviving agents."""
    user = load_user_profile("user_balanced")
    injected_err = RuntimeError("Unexpected order flow memory corruption")
    
    outputs, _ = await run_agents_parallel(
        "TATAMOTORS",
        user,
        inject_exceptions={"VolumeAgent": injected_err},
    )
    
    vol = next(o for o in outputs if o.agent == "VolumeAgent")
    assert vol.status == "failed"
    assert vol.confidence == 0.0
    assert "Unexpected order flow" in vol.reasoning[0]
    
    # Check that surviving agents succeeded
    mom = next(o for o in outputs if o.agent == "MomentumAgent")
    rag = next(o for o in outputs if o.agent == "FilingsRAGAgent")
    assert mom.status == "ok"
    assert rag.status == "ok"
    
    rec = synthesize_recommendation("TATAMOTORS", user, outputs)
    assert rec.action in ["Accumulate", "Watch", "Hold", "Reduce", "Avoid"]


@pytest.mark.asyncio
async def test_failure_mode_6_rag_initialization_failure():
    """Verify clean fallback when RAG subsystem cannot initialize (missing or corrupt corpus)."""
    # Create an uninitialized engine pointing to a non-existent path
    broken_engine = FilingsRAGEngine(corpus_path=Path("non_existent_documents_xyz.json"))
    assert broken_engine.is_initialized is False
    assert broken_engine.init_error is not None

    query_res = broken_engine.query_grounded_filings("TATAMOTORS")
    assert query_res["status"] == "failed"
    assert query_res["label"] == "RAG Init Failed"
    assert query_res["citations"] == []
    assert len(query_res["evidence"]) == 0

    # Inject into agents module
    reset_rag_engine(broken_engine)
    try:
        rag_out = await run_filings_rag_agent("TATAMOTORS")
        assert rag_out.status == "failed"
        assert rag_out.confidence == 0.0
        assert len(rag_out.evidence) == 0
        assert "offline" in rag_out.reasoning[0].lower()

        # Ensure synthesis still produces a safe recommendation
        user = load_user_profile("user_conservative")
        rec = synthesize_recommendation("TATAMOTORS", user, [rag_out])
        assert rec.action is not None
    finally:
        # Restore default engine
        reset_rag_engine(None)


@pytest.mark.asyncio
async def test_failure_mode_7_multiple_agents_failing_simultaneously():
    """Verify system stability when multiple independent agents fail together."""
    user = load_user_profile("user_balanced")
    outputs, _ = await run_agents_parallel(
        "TATAMOTORS",
        user,
        inject_exceptions={
            "MomentumAgent": TimeoutError("Simulated momentum timeout"),
            "VolumeAgent": ValueError("Corrupted tick data"),
        },
    )
    
    failed_agents = [o for o in outputs if o.status == "failed"]
    ok_agents = [o for o in outputs if o.status == "ok"]
    assert len(failed_agents) == 2
    assert len(ok_agents) == 3

    rec = synthesize_recommendation("TATAMOTORS", user, outputs)
    assert rec.action is not None
    assert any("2 agent(s)" in step for step in rec.reasoning_chain)


@pytest.mark.asyncio
async def test_failure_mode_8_all_agents_failing_safe_fallback():
    """Verify that when ALL agents fail, system safely defaults to HOLD with 0% allocation and clear warnings."""
    user = load_user_profile("user_conservative")
    outputs, _ = await run_agents_parallel(
        "TATAMOTORS",
        user,
        inject_exceptions={
            "MomentumAgent": RuntimeError("Feed down"),
            "VolumeAgent": RuntimeError("Feed down"),
            "SentimentAgent": RuntimeError("Feed down"),
            "FilingsRAGAgent": RuntimeError("Feed down"),
            "RiskAgent": RuntimeError("Feed down"),
        },
    )

    assert all(o.status == "failed" for o in outputs)

    rec = synthesize_recommendation("TATAMOTORS", user, outputs)
    assert rec.action == "Hold"
    assert rec.position_size_pct == 0.0
    assert rec.overall_confidence == 0.0
    assert rec.citation_coverage == 0.0
    assert any("CRITICAL ALERT: All 5 specialist agents failed" in step for step in rec.reasoning_chain)
