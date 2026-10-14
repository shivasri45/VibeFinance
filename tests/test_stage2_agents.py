"""
Stage 2 verification tests: 5 specialist agents, parallel execution, speedup, and error recovery.
"""

import pytest
import asyncio
from agents import (
    run_momentum_agent,
    run_volume_agent,
    run_sentiment_agent,
    run_filings_rag_agent,
    run_risk_agent,
    run_agents_parallel,
    run_agents_sequential,
    AgentOutput,
)
from users import load_user_profile


@pytest.mark.asyncio
async def test_all_five_agents_output_contracts():
    user = load_user_profile("user_balanced")
    
    # 1. Momentum
    mom = await run_momentum_agent("TATAMOTORS")
    assert isinstance(mom, AgentOutput)
    assert mom.agent == "MomentumAgent"
    assert -1.0 <= mom.score <= 1.0
    assert 0.0 <= mom.confidence <= 1.0
    assert len(mom.reasoning) > 0
    assert len(mom.evidence) > 0
    assert mom.status == "ok"
    
    # 2. Volume
    vol = await run_volume_agent("TATAMOTORS")
    assert isinstance(vol, AgentOutput)
    assert vol.agent == "VolumeAgent"
    assert vol.label in ["Unusual Spike", "Accumulation", "Distribution", "Normal", "Low Volume Consolidation"]
    
    # 3. Sentiment
    sent = await run_sentiment_agent("TATAMOTORS")
    assert isinstance(sent, AgentOutput)
    assert sent.agent == "SentimentAgent"
    assert sent.label in ["Positive", "Mild Positive", "Neutral", "Mild Negative", "Negative"]
    assert len(sent.evidence) == 3  # 3 synthetic headlines
    
    # 4. Filings RAG
    rag_out = await run_filings_rag_agent("TATAMOTORS")
    assert isinstance(rag_out, AgentOutput)
    assert rag_out.agent == "FilingsRAGAgent"
    assert rag_out.status == "ok"
    assert len(rag_out.evidence) >= 1
    
    # 5. Risk Agent
    risk = await run_risk_agent("TATAMOTORS", user)
    assert isinstance(risk, AgentOutput)
    assert risk.agent == "RiskAgent"
    assert len(risk.evidence) >= 3


@pytest.mark.asyncio
async def test_missing_filing_degraded_behavior():
    rag_out = await run_filings_rag_agent("ADANIENT")
    assert rag_out.status == "degraded"
    assert "No filing evidence" in rag_out.label or "degraded" in rag_out.status
    assert rag_out.confidence <= 0.30


@pytest.mark.asyncio
async def test_parallel_vs_sequential_speedup():
    user = load_user_profile("user_conservative")
    
    # Sequential run
    seq_outputs, seq_ms = await run_agents_sequential("RELIANCE", user)
    assert len(seq_outputs) == 5
    
    # Parallel run
    par_outputs, par_ms = await run_agents_parallel("RELIANCE", user)
    assert len(par_outputs) == 5
    
    print(f"\nSequential latency: {seq_ms:.1f}ms | Parallel latency: {par_ms:.1f}ms")
    # Parallel should be significantly faster than sequential because each agent has a 30-45ms delay
    assert par_ms < seq_ms, f"Parallel ({par_ms}ms) should be faster than sequential ({seq_ms}ms)"


@pytest.mark.asyncio
async def test_agent_timeout_resilience():
    user = load_user_profile("user_aggressive")
    outputs, par_ms = await run_agents_parallel(
        "TATAMOTORS", user, simulate_agent_timeout=True, agent_timeout=0.25
    )
    assert len(outputs) == 5
    # MomentumAgent should be marked as failed / timed out
    mom = next(o for o in outputs if o.agent == "MomentumAgent")
    assert mom.status == "failed"
    assert "timed out" in mom.reasoning[0].lower()
    
    # Other agents should still have completed successfully!
    vol = next(o for o in outputs if o.agent == "VolumeAgent")
    assert vol.status == "ok"
