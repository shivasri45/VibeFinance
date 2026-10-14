"""
Stage 1 verification tests: Data generation, Indicators, Corpus, DB, and User profiles.
"""

import json
from pathlib import Path
from data import get_market_store, TICKER_METADATA
from indicators import compute_indicators, get_latest_indicators
from users import load_user_profile, get_all_user_ids, calculate_portfolio_metrics, simulate_allocation_impact
from db import init_db


def test_market_data():
    store = get_market_store()
    tickers = store.get_all_tickers()
    assert len(tickers) == 12, f"Expected 12 tickers, got {len(tickers)}"
    
    for t in tickers:
        df = store.get_ticker_df(t)
        assert len(df) >= 500, f"Ticker {t} should have >= 500 trading days"
        assert "rsi_14" in df.columns
        assert "vol_zscore_30d" in df.columns
        assert "macd" in df.columns
        assert "sma_20" in df.columns
        assert "sma_50" in df.columns
        assert "ret_20d" in df.columns
        
        latest = get_latest_indicators(df)
        assert 0 <= latest["rsi_14"] <= 100
        assert latest["close"] > 0


def test_corpus():
    corpus_file = Path("corpus/documents.json")
    assert corpus_file.exists(), "corpus/documents.json missing"
    with open(corpus_file, "r") as f:
        docs = json.load(f)
    assert len(docs) == 25, f"Expected 25 documents, got {len(docs)}"
    
    # Check that ADANIENT and BHARTIARTL have NO filings
    companies_in_corpus = {d["company"] for d in docs}
    assert "ADANIENT" not in companies_in_corpus
    assert "BHARTIARTL" not in companies_in_corpus
    assert "TATAMOTORS" in companies_in_corpus
    
    # Check stale/contradictory docs
    stale_docs = [d for d in docs if "STALE" in d["content"]]
    assert len(stale_docs) >= 2, f"Expected at least 2 stale docs, got {len(stale_docs)}"


def test_users_and_db():
    init_db(force_reseed=True)
    users = get_all_user_ids()
    assert len(users) == 3, f"Expected 3 users, got {len(users)}"
    
    profiles = [load_user_profile(u["user_id"]) for u in users]
    
    # Ramesh (Conservative Retiree)
    ramesh = next(p for p in profiles if p.user_id == "user_conservative")
    assert ramesh.risk_score == 0.20
    assert ramesh.max_stock_weight == 0.10
    assert ramesh.sector_cap == 0.25
    assert not ramesh.overtrading_flag
    
    # Vikram (Aggressive)
    vikram = next(p for p in profiles if p.user_id == "user_aggressive")
    assert vikram.risk_score == 0.85
    assert vikram.overtrading_flag
    assert vikram.panic_sell_flag
    assert vikram.recent_trades_count > 5

    # Check portfolio metrics
    for p in profiles:
        weights, hhi, sectors = calculate_portfolio_metrics(p, {})
        assert hhi > 0
        assert len(weights) == len(p.holdings)


if __name__ == "__main__":
    test_market_data()
    test_corpus()
    test_users_and_db()
    print("All Stage 1 tests passed successfully!")
