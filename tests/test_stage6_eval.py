"""
Stage 6 verification tests: Walk-forward backtest rigor, Baseline comparison, and Verified session metrics.
"""

import pytest
import pandas as pd
from evalmetrics import run_walk_forward_backtest, get_session_performance_metrics, clear_backtest_cache
from data import get_market_store
from db import init_db, log_analysis_result, get_all_logs


def test_walk_forward_backtest_execution():
    clear_backtest_cache()
    results = run_walk_forward_backtest(window_days=30)
    
    assert "overall_hit_rate" in results
    assert "agent_accuracy" in results
    assert len(results["ticker_breakdown"]) == 12

    # Verification of exact sample accounting and mathematical consistency
    valid_obs = results["valid_observations"]
    directional_calls = results["directional_calls"]
    correct_preds = results["correct_predictions"]
    incorrect_preds = results["incorrect_predictions"]
    neutral_preds = results["neutral_predictions"]
    excluded = results["excluded_samples"]

    assert valid_obs > 0, "Backtest must evaluate a non-zero count of valid observations"
    assert directional_calls > 0, "Must evaluate a non-zero count of directional calls"
    assert valid_obs == directional_calls + neutral_preds, (
        f"Valid observations ({valid_obs}) must equal directional calls ({directional_calls}) + neutral ({neutral_preds})"
    )
    assert directional_calls == correct_preds + incorrect_preds, (
        f"Directional calls ({directional_calls}) must equal correct ({correct_preds}) + incorrect ({incorrect_preds})"
    )

    # Verification that overall hit rate is computed from actual counts, NOT hardcoded
    calculated_rate = round((correct_preds / directional_calls) * 100.0, 1)
    assert results["overall_hit_rate"] == calculated_rate, (
        f"Reported hit rate {results['overall_hit_rate']}% must equal calculated {calculated_rate}%"
    )

    # Baseline comparison verification
    assert "majority_baseline_hit_rate_pct" in results
    assert "outperformance_vs_baseline_pct" in results
    majority_rate = results["majority_baseline_hit_rate_pct"]
    assert 0.0 < majority_rate <= 100.0, f"Majority baseline {majority_rate}% must be between 0 and 100%"
    assert results["outperformance_vs_baseline_pct"] == round(calculated_rate - majority_rate, 1)


    # Check individual agent stats
    agent_acc = results["agent_accuracy"]
    for agent_key in ["Momentum", "Volume", "Sentiment", "Filings"]:
        assert agent_key in agent_acc
        d = agent_acc[agent_key]
        assert d["total_calls"] > 0
        assert 0.0 <= d["hit_rate_pct"] <= 100.0


def test_walk_forward_backtest_edge_cases():
    """Verify handling of empty store or edge windows without crashing."""
    # Test with very large window days that exceeds available data
    results_overflow = run_walk_forward_backtest(window_days=9999)
    assert results_overflow["valid_observations"] == 0
    assert results_overflow["overall_hit_rate"] == 0.0
    assert results_overflow["excluded_samples"] > 0


def test_session_performance_metrics_differentiation():
    """Verify that session metrics distinguish between empty state and real live logs."""
    # 1. Test when no logs exist
    import sqlite3
    from db import get_connection
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM analysis_logs")
    conn.commit()
    conn.close()

    empty_metrics = get_session_performance_metrics()
    assert empty_metrics["has_session_data"] is False
    assert empty_metrics["total_analyses"] == 0
    assert empty_metrics["avg_parallel_latency_ms"] is None
    assert empty_metrics["parallel_speedup"] is None
    assert empty_metrics["avg_citation_coverage_pct"] is None
    assert "demo_baseline" in empty_metrics
    assert empty_metrics["demo_baseline"]["samples"] == 14

    # 2. Test after logging a real analysis
    log_analysis_result({
        "timestamp": "2024-03-28 10:00:00",
        "ticker": "TATAMOTORS",
        "user_id": "user_balanced",
        "action": "Accumulate",
        "position_size_pct": 5.0,
        "overall_confidence": 0.82,
        "score": 0.65,
        "latency_ms": 45.2,
        "parallel_latency_ms": 45.2,
        "sequential_latency_ms": 210.0,
        "citation_coverage": 100.0,
        "hhi_before": 1800,
        "hhi_after": 1720,
        "max_sector_weight_before": 0.20,
        "max_sector_weight_after": 0.25,
        "disagreements_count": 0,
        "guardrail_triggered": "",
        "nudge_triggered": "",
        "details_json": "{}",
    })

    live_metrics = get_session_performance_metrics()
    assert live_metrics["has_session_data"] is True
    assert live_metrics["total_analyses"] == 1
    assert live_metrics["avg_parallel_latency_ms"] == 45.2
    assert live_metrics["avg_sequential_latency_ms"] == 210.0
    assert live_metrics["parallel_speedup"] == round(210.0 / 45.2, 2)
    assert live_metrics["avg_citation_coverage_pct"] == 100.0
