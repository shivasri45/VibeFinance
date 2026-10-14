"""
VibeFinance User Personalization & Portfolio Analytics
Pydantic v2 contracts for user profiles, portfolio risk metrics, and behavioral history.
"""

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
from db import get_connection, init_db


class Holding(BaseModel):
    ticker: str
    shares: int
    avg_buy_price: float
    sector: str
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    weight: Optional[float] = None  # 0.0 to 1.0


class UserProfile(BaseModel):
    user_id: str
    name: str
    profile_type: str  # 'Conservative Retiree', 'Balanced Wealth Builder', 'Aggressive Momentum Trader'
    risk_score: float  # 0.0 to 1.0
    max_stock_weight: float  # e.g. 0.10, 0.20, 0.35
    sector_cap: float  # e.g. 0.25, 0.35, 0.50
    total_capital: float
    cash_balance: float
    overtrading_flag: bool = False
    panic_sell_flag: bool = False
    holdings: List[Holding] = Field(default_factory=list)
    watchlist: List[str] = Field(default_factory=list)
    recent_trades_count: int = 0
    panic_sales_count: int = 0


def calculate_portfolio_metrics(
    user: UserProfile, current_prices: Dict[str, float]
) -> Tuple[Dict[str, float], float, Dict[str, float]]:
    """
    Computes:
      - stock_weights: Dict[ticker, float]
      - hhi: Herfindahl-Hirschman Index (0 to 10,000)
      - sector_weights: Dict[sector, float]
    """
    total_val = user.cash_balance
    stock_values: Dict[str, float] = {}
    sector_values: Dict[str, float] = {}

    for h in user.holdings:
        price = current_prices.get(h.ticker, h.avg_buy_price)
        val = h.shares * price
        stock_values[h.ticker] = val
        sector_values[h.sector] = sector_values.get(h.sector, 0.0) + val
        total_val += val

    if total_val <= 0:
        return {}, 0.0, {}

    stock_weights = {t: v / total_val for t, v in stock_values.items()}
    # Include cash as a stabilizing asset or purely equity weights
    sector_weights = {s: v / total_val for s, v in sector_values.items()}
    
    # HHI: sum of (weight_pct)^2 across all stocks (excluding pure cash)
    hhi = sum((w * 100.0) ** 2 for w in stock_weights.values())

    return stock_weights, round(hhi, 1), sector_weights


def simulate_allocation_impact(
    user: UserProfile,
    target_ticker: str,
    target_sector: str,
    allocation_pct: float,  # e.g. 0.05 for 5%
    current_prices: Dict[str, float],
) -> Dict[str, Any]:
    """
    Calculates portfolio concentration (HHI and sector weight) before and after adding allocation_pct.
    """
    stock_weights, hhi_before, sector_weights = calculate_portfolio_metrics(user, current_prices)
    
    current_ticker_weight = stock_weights.get(target_ticker, 0.0)
    current_sector_weight = sector_weights.get(target_sector, 0.0)

    # After simulated addition:
    # We assume user deploys cash into the target ticker:
    # new stock weight approx = (current_stock_val + allocation_amount) / total_val
    new_stock_weight = current_ticker_weight + allocation_pct
    new_sector_weight = current_sector_weight + allocation_pct

    # Recompute simulated HHI
    sim_weights = dict(stock_weights)
    sim_weights[target_ticker] = new_stock_weight
    hhi_after = sum((w * 100.0) ** 2 for w in sim_weights.values())

    max_sector_weight_before = max(sector_weights.values()) if sector_weights else 0.0
    sim_sector_weights = dict(sector_weights)
    sim_sector_weights[target_sector] = new_sector_weight
    max_sector_weight_after = max(sim_sector_weights.values()) if sim_sector_weights else 0.0

    return {
        "stock_weight_before": round(current_ticker_weight, 4),
        "stock_weight_after": round(new_stock_weight, 4),
        "sector_weight_before": round(current_sector_weight, 4),
        "sector_weight_after": round(new_sector_weight, 4),
        "hhi_before": round(hhi_before, 1),
        "hhi_after": round(hhi_after, 1),
        "max_sector_weight_before": round(max_sector_weight_before, 4),
        "max_sector_weight_after": round(max_sector_weight_after, 4),
        "exceeds_stock_cap": new_stock_weight > user.max_stock_weight,
        "exceeds_sector_cap": new_sector_weight > user.sector_cap,
    }


def load_user_profile(user_id: str, current_prices: Optional[Dict[str, float]] = None) -> UserProfile:
    """Fetch complete UserProfile from SQLite."""
    init_db()
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    u_row = cur.fetchone()
    if not u_row:
        conn.close()
        raise ValueError(f"User {user_id} not found in database.")

    cur.execute("SELECT * FROM user_holdings WHERE user_id = ?", (user_id,))
    h_rows = cur.fetchall()

    cur.execute("SELECT ticker FROM user_watchlist WHERE user_id = ?", (user_id,))
    w_rows = cur.fetchall()

    cur.execute("SELECT COUNT(*) as total, SUM(flag_panic_sell) as panics FROM user_decisions WHERE user_id = ?", (user_id,))
    d_row = cur.fetchone()
    trades_count = d_row["total"] if d_row else 0
    panics_count = d_row["panics"] if d_row and d_row["panics"] else 0

    holdings = []
    prices = current_prices or {}
    total_val = u_row["cash_balance"]

    for hr in h_rows:
        t = hr["ticker"]
        p = prices.get(t, hr["avg_buy_price"])
        val = hr["shares"] * p
        total_val += val
        holdings.append(
            Holding(
                ticker=t,
                shares=hr["shares"],
                avg_buy_price=hr["avg_buy_price"],
                sector=hr["sector"],
                current_price=p,
                market_value=val,
            )
        )

    # Populate relative weights
    if total_val > 0:
        for h in holdings:
            if h.market_value:
                h.weight = round(h.market_value / total_val, 4)

    profile = UserProfile(
        user_id=u_row["user_id"],
        name=u_row["name"],
        profile_type=u_row["profile_type"],
        risk_score=float(u_row["risk_score"]),
        max_stock_weight=float(u_row["max_stock_weight"]),
        sector_cap=float(u_row["sector_cap"]),
        total_capital=float(total_val),
        cash_balance=float(u_row["cash_balance"]),
        overtrading_flag=bool(u_row["overtrading_flag"]),
        panic_sell_flag=bool(u_row["panic_sell_flag"]),
        holdings=holdings,
        watchlist=[w["ticker"] for w in w_rows],
        recent_trades_count=trades_count,
        panic_sales_count=panics_count,
    )
    conn.close()
    return profile


def get_all_user_ids() -> List[Dict[str, str]]:
    """Return user list for UI selection."""
    init_db()
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_id, name, profile_type FROM users")
    rows = [{"user_id": r["user_id"], "name": r["name"], "profile_type": r["profile_type"]} for r in cur.fetchall()]
    conn.close()
    return rows
