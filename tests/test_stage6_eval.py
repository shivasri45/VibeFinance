"""
Stage 6 verification tests: Walk-forward backtest, Directional hit rate, and Session metrics.
"""

from evalmetrics import run_walk_forward_backtest, get_session_performance_metrics


def test_walk_forward_backtest_execution():
    results = run_walk_forward_backtest(window_days=30)
    assert "overall_hit_rate" in results
    assert "agent_accuracy" in results
    assert len(results["ticker_breakdown"]) == 12
    
    overall = results["overall_hit_rate"]
    print(f"\nWalk-forward Directional Hit Rate: {overall}%")
    assert 40.0 <= overall <= 95.0, f"Hit rate {overall}% out of realistic range"
    
    # Check individual agent stats
    agent_acc = results["agent_accuracy"]
    assert "Momentum" in agent_acc
    assert "Volume" in agent_acc
    assert "Sentiment" in agent_acc
    assert "Filings" in agent_acc


def test_session_performance_metrics():
    metrics = get_session_performance_metrics()
    assert "avg_parallel_latency_ms" in metrics
    assert "parallel_speedup" in metrics
    assert "avg_citation_coverage_pct" in metrics
    assert metrics["parallel_speedup"] > 1.0
