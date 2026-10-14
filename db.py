"""
VibeFinance Database Layer (SQLite)
Manages user profiles, portfolio holdings, decision history, and analysis evaluation logs.
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Any, Optional

DB_PATH = Path(__file__).parent / "vibefinance.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(force_reseed: bool = False):
    """Initialize tables and seed default users if empty."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        profile_type TEXT NOT NULL,
        risk_score REAL NOT NULL,
        max_stock_weight REAL NOT NULL,
        sector_cap REAL NOT NULL,
        total_capital REAL NOT NULL,
        cash_balance REAL NOT NULL,
        overtrading_flag INTEGER DEFAULT 0,
        panic_sell_flag INTEGER DEFAULT 0
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_holdings (
        user_id TEXT NOT NULL,
        ticker TEXT NOT NULL,
        shares INTEGER NOT NULL,
        avg_buy_price REAL NOT NULL,
        sector TEXT NOT NULL,
        PRIMARY KEY (user_id, ticker),
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_watchlist (
        user_id TEXT NOT NULL,
        ticker TEXT NOT NULL,
        PRIMARY KEY (user_id, ticker),
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        ticker TEXT NOT NULL,
        action TEXT NOT NULL,
        amount REAL NOT NULL,
        holding_period_days INTEGER NOT NULL,
        flag_panic_sell INTEGER DEFAULT 0,
        flag_overtrading INTEGER DEFAULT 0,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS analysis_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        ticker TEXT NOT NULL,
        user_id TEXT NOT NULL,
        action TEXT NOT NULL,
        position_size_pct REAL NOT NULL,
        overall_confidence REAL NOT NULL,
        score REAL NOT NULL,
        latency_ms REAL NOT NULL,
        parallel_latency_ms REAL NOT NULL,
        sequential_latency_ms REAL NOT NULL,
        citation_coverage REAL NOT NULL,
        hhi_before REAL NOT NULL,
        hhi_after REAL NOT NULL,
        max_sector_weight_before REAL NOT NULL,
        max_sector_weight_after REAL NOT NULL,
        disagreements_count INTEGER NOT NULL,
        guardrail_triggered TEXT,
        nudge_triggered TEXT,
        details_json TEXT
    )
    """)

    conn.commit()

    # Check if users already seeded
    cur.execute("SELECT COUNT(*) as cnt FROM users")
    count = cur.fetchone()["cnt"]

    if count == 0 or force_reseed:
        if force_reseed:
            cur.execute("DELETE FROM user_holdings")
            cur.execute("DELETE FROM user_watchlist")
            cur.execute("DELETE FROM user_decisions")
            cur.execute("DELETE FROM users")
            conn.commit()

        _seed_default_data(conn)

    conn.close()


def _seed_default_data(conn: sqlite3.Connection):
    cur = conn.cursor()

    users_data = [
        (
            "user_conservative",
            "Ramesh Patel",
            "Conservative Retiree",
            0.20,
            0.10,  # Max 10% in single stock
            0.25,  # Max 25% in one sector
            5_000_000.0,  # ₹50 Lakhs
            1_250_000.0,  # 25% Cash
            0,
            0,
        ),
        (
            "user_balanced",
            "Priya Sharma",
            "Balanced Wealth Builder",
            0.50,
            0.20,  # Max 20% in single stock
            0.35,  # Max 35% in one sector
            2_000_000.0,  # ₹20 Lakhs
            500_000.0,    # 25% Cash
            0,
            0,
        ),
        (
            "user_aggressive",
            "Vikram Malhotra",
            "Aggressive Momentum Trader",
            0.85,
            0.35,  # Max 35% in single stock
            0.50,  # Max 50% in one sector
            800_000.0,    # ₹8 Lakhs
            160_000.0,    # 20% Cash
            1,  # Overtrading flag
            1,  # Panic selling history flag
        ),
    ]

    cur.executemany("""
    INSERT INTO users (
        user_id, name, profile_type, risk_score, max_stock_weight,
        sector_cap, total_capital, cash_balance, overtrading_flag, panic_sell_flag
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, users_data)

    # Seed holdings
    # 1. Conservative Retiree holdings: Heavy FMCG, Large Financials, IT bluechip
    conservative_holdings = [
        ("user_conservative", "ITC", 2500, 410.0, "FMCG"),
        ("user_conservative", "HDFCBANK", 600, 1530.0, "Financial Services"),
        ("user_conservative", "TCS", 250, 3450.0, "Information Technology"),
        ("user_conservative", "ICICIBANK", 900, 950.0, "Financial Services"),
    ]

    # 2. Balanced holdings: Diversified across Energy, IT, PSU Bank, Infra, Pharma
    balanced_holdings = [
        ("user_balanced", "RELIANCE", 140, 2420.0, "Energy & Retail"),
        ("user_balanced", "INFY", 220, 1420.0, "Information Technology"),
        ("user_balanced", "SBIN", 450, 605.0, "Financial Services"),
        ("user_balanced", "LT", 80, 3050.0, "Capital Goods & Infra"),
        ("user_balanced", "SUNPHARMA", 160, 1140.0, "Healthcare & Pharma"),
    ]

    # 3. Aggressive holdings: High beta, auto breakout, metals, telecom
    aggressive_holdings = [
        ("user_aggressive", "TATAMOTORS", 360, 640.0, "Automobile"),
        ("user_aggressive", "ADANIENT", 70, 2750.0, "Diversified & Metals"),
        ("user_aggressive", "BHARTIARTL", 150, 910.0, "Telecommunications"),
    ]

    all_holdings = conservative_holdings + balanced_holdings + aggressive_holdings
    cur.executemany("""
    INSERT INTO user_holdings (user_id, ticker, shares, avg_buy_price, sector)
    VALUES (?, ?, ?, ?, ?)
    """, all_holdings)

    # Watchlists
    watchlists = [
        ("user_conservative", "RELIANCE"),
        ("user_conservative", "SUNPHARMA"),
        ("user_balanced", "TATAMOTORS"),
        ("user_balanced", "TCS"),
        ("user_aggressive", "INFY"),
        ("user_aggressive", "LT"),
        ("user_aggressive", "RELIANCE"),
    ]
    cur.executemany("INSERT INTO user_watchlist (user_id, ticker) VALUES (?, ?)", watchlists)

    # Decision History (User 3 has overtrading and panic sells)
    decisions = [
        # Conservative: disciplined buys, multi-month holds
        ("user_conservative", "2023-04-10", "ITC", "BUY", 200000.0, 360, 0, 0),
        ("user_conservative", "2023-08-15", "TCS", "BUY", 300000.0, 240, 0, 0),
        ("user_conservative", "2023-11-20", "HDFCBANK", "BUY", 250000.0, 140, 0, 0),

        # Balanced: periodic rebalance
        ("user_balanced", "2023-06-12", "RELIANCE", "BUY", 180000.0, 300, 0, 0),
        ("user_balanced", "2023-09-05", "SBIN", "BUY", 150000.0, 210, 0, 0),
        ("user_balanced", "2023-12-18", "INFY", "BUY", 160000.0, 110, 0, 0),

        # Aggressive Vikram: 14 rapid trades in past 30 days, rapid flips, 2 panic sells on dips!
        ("user_aggressive", "2024-03-01", "TATAMOTORS", "BUY", 150000.0, 4, 0, 1),
        ("user_aggressive", "2024-03-03", "TATAMOTORS", "SELL", 145000.0, 2, 1, 1),  # Panic sell
        ("user_aggressive", "2024-03-05", "ADANIENT", "BUY", 200000.0, 3, 0, 1),
        ("user_aggressive", "2024-03-07", "ADANIENT", "SELL", 192000.0, 2, 1, 1),   # Panic sell
        ("user_aggressive", "2024-03-10", "BHARTIARTL", "BUY", 120000.0, 5, 0, 1),
        ("user_aggressive", "2024-03-12", "TATAMOTORS", "BUY", 220000.0, 6, 0, 1),
        ("user_aggressive", "2024-03-15", "INFY", "BUY", 100000.0, 1, 0, 1),
        ("user_aggressive", "2024-03-16", "INFY", "SELL", 97000.0, 1, 1, 1),        # Quick chop
        ("user_aggressive", "2024-03-19", "SBIN", "BUY", 80000.0, 2, 0, 1),
        ("user_aggressive", "2024-03-21", "SBIN", "SELL", 81000.0, 2, 0, 1),
        ("user_aggressive", "2024-03-24", "ADANIENT", "BUY", 180000.0, 3, 0, 1),
    ]

    cur.executemany("""
    INSERT INTO user_decisions (
        user_id, timestamp, ticker, action, amount,
        holding_period_days, flag_panic_sell, flag_overtrading
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, decisions)

    conn.commit()


def log_analysis_result(entry: Dict[str, Any]):
    """Store synthesis and evaluation metrics in SQLite."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO analysis_logs (
        timestamp, ticker, user_id, action, position_size_pct,
        overall_confidence, score, latency_ms, parallel_latency_ms,
        sequential_latency_ms, citation_coverage, hhi_before, hhi_after,
        max_sector_weight_before, max_sector_weight_after,
        disagreements_count, guardrail_triggered, nudge_triggered, details_json
    ) VALUES (
        :timestamp, :ticker, :user_id, :action, :position_size_pct,
        :overall_confidence, :score, :latency_ms, :parallel_latency_ms,
        :sequential_latency_ms, :citation_coverage, :hhi_before, :hhi_after,
        :max_sector_weight_before, :max_sector_weight_after,
        :disagreements_count, :guardrail_triggered, :nudge_triggered, :details_json
    )
    """, entry)
    conn.commit()
    conn.close()


def get_all_logs() -> List[Dict[str, Any]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM analysis_logs ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
