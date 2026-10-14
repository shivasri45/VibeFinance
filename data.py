"""
VibeFinance Market Data Generator
Generates 2 years of daily OHLCV for 12 NSE tickers using seeded regime-switching GBM
with realistic Indian market dynamics, injected events, sector mapping, and synthetic news headlines.
"""

from typing import Dict, List, Any
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from indicators import compute_indicators

RANDOM_SEED = 42

TICKER_METADATA: Dict[str, Dict[str, Any]] = {
    "RELIANCE": {
        "name": "Reliance Industries Ltd.",
        "sector": "Energy & Retail",
        "base_price": 2450.0,
        "base_volume": 6_500_000,
        "beta": 1.05,
    },
    "TCS": {
        "name": "Tata Consultancy Services Ltd.",
        "sector": "Information Technology",
        "base_price": 3500.0,
        "base_volume": 2_200_000,
        "beta": 0.85,
    },
    "HDFCBANK": {
        "name": "HDFC Bank Ltd.",
        "sector": "Financial Services",
        "base_price": 1580.0,
        "base_volume": 14_000_000,
        "beta": 1.10,
    },
    "INFY": {
        "name": "Infosys Ltd.",
        "sector": "Information Technology",
        "base_price": 1450.0,
        "base_volume": 7_000_000,
        "beta": 0.95,
        "conflict_engineered": True,  # Engineered conflict: Technical bounce vs negative filing/news
    },
    "ITC": {
        "name": "ITC Ltd.",
        "sector": "FMCG",
        "base_price": 420.0,
        "base_volume": 11_000_000,
        "beta": 0.65,
    },
    "SBIN": {
        "name": "State Bank of India",
        "sector": "Financial Services",
        "base_price": 620.0,
        "base_volume": 18_000_000,
        "beta": 1.25,
    },
    "ICICIBANK": {
        "name": "ICICI Bank Ltd.",
        "sector": "Financial Services",
        "base_price": 980.0,
        "base_volume": 12_500_000,
        "beta": 1.15,
    },
    "LT": {
        "name": "Larsen & Toubro Ltd.",
        "sector": "Capital Goods & Infra",
        "base_price": 3100.0,
        "base_volume": 2_800_000,
        "beta": 1.15,
    },
    "TATAMOTORS": {
        "name": "Tata Motors Passenger Vehicles Ltd.",
        "sector": "Automobile",
        "base_price": 680.0,
        "base_volume": 15_000_000,
        "beta": 1.35,
        "breakout_engineered": True,  # Demo scenario: Volume spike + Breakout + Filing risk
    },
    "SUNPHARMA": {
        "name": "Sun Pharmaceutical Industries Ltd.",
        "sector": "Healthcare & Pharma",
        "base_price": 1180.0,
        "base_volume": 3_500_000,
        "beta": 0.70,
    },
    "ADANIENT": {
        "name": "Adani Enterprises Ltd.",
        "sector": "Diversified & Metals",
        "base_price": 2800.0,
        "base_volume": 4_800_000,
        "beta": 1.60,
        "no_filing": True,  # Engineered failure mode: Missing filing
    },
    "BHARTIARTL": {
        "name": "Bharti Airtel Ltd.",
        "sector": "Telecommunications",
        "base_price": 920.0,
        "base_volume": 8_500_000,
        "beta": 0.90,
        "no_filing": True,  # Engineered failure mode: Missing filing
    },
}

SYNTHETIC_HEADLINES: Dict[str, List[Dict[str, Any]]] = {
    "RELIANCE": [
        {"title": "Reliance Retail plans further expansion into Tier-2 grocery hubs", "source": "Mint", "sentiment": 0.65, "date": "2024-03-12"},
        {"title": "Jio adds 3.8M broadband subscribers; ARPU rises to Rs 181.7", "source": "Economic Times", "sentiment": 0.75, "date": "2024-03-18"},
        {"title": "Oil-to-chemicals margin remains steady despite Red Sea freight volatility", "source": "Business Standard", "sentiment": 0.20, "date": "2024-03-24"},
    ],
    "TCS": [
        {"title": "TCS signs $1.2B mega-deal with UK pensions provider Nest", "source": "Financial Express", "sentiment": 0.80, "date": "2024-03-14"},
        {"title": "Management highlights cautious discretionary IT spending across BFSI", "source": "Mint", "sentiment": -0.30, "date": "2024-03-20"},
        {"title": "TCS expands generative AI deployment across 100,000 workforce", "source": "Moneycontrol", "sentiment": 0.60, "date": "2024-03-25"},
    ],
    "HDFCBANK": [
        {"title": "HDFC Bank advances deposit accretion campaign with special CD rates", "source": "Business Standard", "sentiment": 0.35, "date": "2024-03-10"},
        {"title": "Post-merger loan-to-deposit ratio improves by 120 bps in Q4", "source": "Mint", "sentiment": 0.70, "date": "2024-03-19"},
        {"title": "Net interest margins stable at 3.4% amidst intense retail deposit competition", "source": "Economic Times", "sentiment": 0.25, "date": "2024-03-26"},
    ],
    "INFY": [
        {"title": "Infosys slashes annual revenue growth guidance to 1.5-2.0% amid client budget freezes", "source": "Mint", "sentiment": -0.85, "date": "2024-03-11"},
        {"title": "Senior executive departures spark retention concerns in European sales division", "source": "Economic Times", "sentiment": -0.70, "date": "2024-03-18"},
        {"title": "Infosys secures cloud migration partnership with German manufacturer", "source": "Moneycontrol", "sentiment": 0.40, "date": "2024-03-22"},
    ],
    "ITC": [
        {"title": "ITC hotel business demerger receives key NCLT regulatory clearances", "source": "Business Standard", "sentiment": 0.60, "date": "2024-03-15"},
        {"title": "Cigarette volumes grow 4% YoY; rural FMCG demand signals steady uptick", "source": "Mint", "sentiment": 0.55, "date": "2024-03-22"},
        {"title": "Agri-business exports affected slightly by domestic wheat export curbs", "source": "Financial Express", "sentiment": -0.15, "date": "2024-03-27"},
    ],
    "SBIN": [
        {"title": "SBI gross NPA drops below 2.4%, lowest credit cost in a decade", "source": "Economic Times", "sentiment": 0.85, "date": "2024-03-12"},
        {"title": "State Bank of India raises Rs 5,000 crore via long-term infrastructure bonds", "source": "Mint", "sentiment": 0.50, "date": "2024-03-19"},
        {"title": "Retail loan book crosses Rs 13 lakh crore with strong unsecured asset quality", "source": "Business Standard", "sentiment": 0.70, "date": "2024-03-25"},
    ],
    "ICICIBANK": [
        {"title": "ICICI Bank delivers industry-leading RoA of 2.3% for third consecutive quarter", "source": "Mint", "sentiment": 0.80, "date": "2024-03-14"},
        {"title": "Domestic loan portfolio grows 18.8% YoY driven by business banking and SME", "source": "Economic Times", "sentiment": 0.75, "date": "2024-03-20"},
        {"title": "Core operating profit surges 14% with strong fee income growth", "source": "Moneycontrol", "sentiment": 0.65, "date": "2024-03-26"},
    ],
    "LT": [
        {"title": "L&T order inflow tops Rs 72,000 crore on domestic hydrocarbon and rail orders", "source": "Business Standard", "sentiment": 0.85, "date": "2024-03-12"},
        {"title": "Middle East execution pace accelerates, easing working capital cycle", "source": "Mint", "sentiment": 0.60, "date": "2024-03-17"},
        {"title": "Supply chain steel price uptick poses mild margin headwind for fixed-price contracts", "source": "Financial Express", "sentiment": -0.30, "date": "2024-03-23"},
    ],
    "TATAMOTORS": [
        {"title": "Tata Motors reports breakout EV sales with Nexon.ev and Punch.ev surging 42% YoY", "source": "Economic Times", "sentiment": 0.85, "date": "2024-03-15"},
        {"title": "JLR order book holds firm at 148,000 units with Defender and Range Rover leading", "source": "Mint", "sentiment": 0.75, "date": "2024-03-21"},
        {"title": "Board approves formal demerger of commercial vehicle and passenger vehicle arms", "source": "Business Standard", "sentiment": 0.65, "date": "2024-03-26"},
    ],
    "SUNPHARMA": [
        {"title": "Sun Pharma specialty portfolio revenue expands 19% led by Ilumya and Cequa", "source": "Moneycontrol", "sentiment": 0.75, "date": "2024-03-13"},
        {"title": "US FDA issues inspection observation on Halol manufacturing facility", "source": "Mint", "sentiment": -0.55, "date": "2024-03-18"},
        {"title": "Domestic formulations business grows 11% outpacing broader IPM market", "source": "Business Standard", "sentiment": 0.60, "date": "2024-03-24"},
    ],
    "ADANIENT": [
        {"title": "Adani Enterprises commissions first phase of Khavda solar project", "source": "Economic Times", "sentiment": 0.60, "date": "2024-03-11"},
        {"title": "Airports division EBITDA jumps 34% on surging passenger traffic and non-aero revenue", "source": "Mint", "sentiment": 0.65, "date": "2024-03-19"},
        {"title": "Rating agencies affirm stable outlook following deleveraging milestones", "source": "Business Standard", "sentiment": 0.45, "date": "2024-03-27"},
    ],
    "BHARTIARTL": [
        {"title": "Bharti Airtel 5G subscriber base reaches 65 million with pan-India rollout", "source": "Mint", "sentiment": 0.75, "date": "2024-03-14"},
        {"title": "Africa business reports steady constant-currency revenue despite currency devaluations", "source": "Economic Times", "sentiment": 0.30, "date": "2024-03-20"},
        {"title": "Homes and enterprise businesses maintain healthy double-digit growth trajectory", "source": "Moneycontrol", "sentiment": 0.65, "date": "2024-03-25"},
    ],
}


def generate_regime_switching_gbm(
    n_days: int = 504,  # Approx 2 trading years
    base_price: float = 1000.0,
    base_vol: float = 1_000_000,
    seed: int = 42,
    beta: float = 1.0,
    ticker: str = "TICKER",
) -> pd.DataFrame:
    """
    Simulate OHLCV series using a 3-state Markov regime-switching Geometric Brownian Motion:
      State 0: Bull (drift = +18% ann, vol = 16% ann)
      State 1: Sideways/Volatile (drift = +2% ann, vol = 22% ann)
      State 2: Bear (drift = -16% ann, vol = 28% ann)
    """
    rng = np.random.RandomState(seed)

    # Daily parameters (252 trading days/year)
    regime_params = [
        {"mu": 0.18 / 252, "sigma": 0.16 / np.sqrt(252)},  # Bull
        {"mu": 0.02 / 252, "sigma": 0.22 / np.sqrt(252)},  # Sideways
        {"mu": -0.16 / 252, "sigma": 0.28 / np.sqrt(252)},  # Bear
    ]

    # Markov transition matrix (sticky regimes)
    # [ [P00, P01, P02], [P10, P11, P12], [P20, P21, P22] ]
    P = np.array([
        [0.96, 0.03, 0.01],
        [0.04, 0.93, 0.03],
        [0.02, 0.05, 0.93],
    ])

    current_state = rng.choice([0, 1, 2], p=[0.5, 0.3, 0.2])
    states = [current_state]

    for _ in range(1, n_days):
        current_state = rng.choice([0, 1, 2], p=P[current_state])
        states.append(current_state)

    # Generate log returns
    log_returns = np.zeros(n_days)
    for i in range(n_days):
        st = states[i]
        mu = regime_params[st]["mu"] * beta
        sigma = regime_params[st]["sigma"] * (0.8 + 0.4 * beta)
        log_returns[i] = rng.normal(mu, sigma)

    # Specific event injections for hackathon demos:
    if ticker == "TATAMOTORS":
        # Breakout scenario: Last 15 days strong rally + huge volume spike at t = -3
        log_returns[-12:] += rng.uniform(0.008, 0.025, size=12)
        log_returns[-3] += 0.048  # +4.8% earnings/breakout surge

    elif ticker == "INFY":
        # Conflict scenario: Last 5 days technical bounce (+3%) but recent gap-down prior
        log_returns[-20] = -0.052  # earnings warning gap
        log_returns[-19:-5] += rng.uniform(-0.005, 0.002, size=14)
        log_returns[-4:] += rng.uniform(0.015, 0.03, size=4)  # rapid oversold bounce

    elif ticker == "RELIANCE":
        # Steady consolidation followed by upward breakout
        log_returns[-8:] += rng.uniform(0.005, 0.018, size=8)

    # Cumulative price path
    price_multipliers = np.exp(np.cumsum(log_returns))
    close_prices = base_price * price_multipliers

    # Calendar dates up to current simulated date
    end_date = datetime(2024, 3, 28)
    dates = []
    curr = end_date
    while len(dates) < n_days:
        if curr.weekday() < 5:  # Monday to Friday
            dates.append(curr)
        curr -= timedelta(days=1)
    dates.reverse()

    # Generate realistic Open, High, Low, Volume
    data = []
    for i in range(n_days):
        c = close_prices[i]
        prev_c = close_prices[i - 1] if i > 0 else c
        o = prev_c * (1 + rng.normal(0, 0.003))
        h = max(o, c) * (1 + abs(rng.normal(0.004, 0.003)))
        l = min(o, c) * (1 - abs(rng.normal(0.004, 0.003)))

        # Volume dynamics based on return magnitude + regime
        abs_ret = abs(log_returns[i])
        vol_mult = 1.0 + 12.0 * abs_ret + (0.3 if states[i] == 0 else 0.0)
        vol = base_vol * vol_mult * rng.uniform(0.75, 1.35)

        # Injected volume spike for demo ticker
        if ticker == "TATAMOTORS" and i >= n_days - 3:
            vol *= 3.8  # Unusual volume surge

        data.append({
            "Date": dates[i].strftime("%Y-%m-%d"),
            "Open": round(float(o), 2),
            "High": round(float(h), 2),
            "Low": round(float(l), 2),
            "Close": round(float(c), 2),
            "Volume": int(vol),
        })

    df = pd.DataFrame(data)
    df = compute_indicators(df)
    return df


class MarketDataStore:
    """In-memory and cached repository for simulated NSE universe."""
    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        self._universe: Dict[str, pd.DataFrame] = {}
        self._cache_stale: Dict[str, pd.DataFrame] = {}
        self._initialize()

    def _initialize(self):
        for idx, (ticker, meta) in enumerate(TICKER_METADATA.items()):
            ticker_seed = self.seed + idx * 101
            df = generate_regime_switching_gbm(
                n_days=504,
                base_price=meta["base_price"],
                base_vol=meta["base_volume"],
                seed=ticker_seed,
                beta=meta["beta"],
                ticker=ticker,
            )
            self._universe[ticker] = df
            # Create a 5-day stale copy for failure mode testing
            self._cache_stale[ticker] = df.iloc[:-5].copy()

    def get_ticker_df(self, ticker: str, simulate_stale: bool = False) -> pd.DataFrame:
        """Retrieve price history, optionally simulating stale feed failure mode."""
        if ticker not in self._universe:
            raise KeyError(f"Ticker {ticker} not found in universe.")
        if simulate_stale:
            return self._cache_stale[ticker].copy()
        return self._universe[ticker].copy()

    def get_all_tickers(self) -> List[str]:
        return list(TICKER_METADATA.keys())

    def get_metadata(self, ticker: str) -> Dict[str, Any]:
        return TICKER_METADATA.get(ticker, {})

    def get_headlines(self, ticker: str) -> List[Dict[str, Any]]:
        return SYNTHETIC_HEADLINES.get(ticker, [])

    def get_universe_summary(self) -> pd.DataFrame:
        """Summary table for Dashboard Tab 1 (Signals & Analysis)."""
        rows = []
        for ticker, meta in TICKER_METADATA.items():
            df = self._universe[ticker]
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            ret_1d = ((latest["Close"] - prev["Close"]) / prev["Close"]) * 100
            ret_20d = float(latest.get("ret_20d", 0.0)) * 100
            rsi = float(latest.get("rsi_14", 50.0))
            vol_z = float(latest.get("vol_zscore_30d", 0.0))
            
            rows.append({
                "Ticker": ticker,
                "Company": meta["name"],
                "Sector": meta["sector"],
                "Close (₹)": round(float(latest["Close"]), 2),
                "1D Change (%)": round(ret_1d, 2),
                "20D Return (%)": round(ret_20d, 2),
                "RSI (14)": round(rsi, 1),
                "Vol Z-Score": round(vol_z, 2),
            })
        return pd.DataFrame(rows)


# Singleton instance
_store = None

def get_market_store() -> MarketDataStore:
    global _store
    if _store is None:
        _store = MarketDataStore()
    return _store


if __name__ == "__main__":
    store = get_market_store()
    print(f"Loaded {len(store.get_all_tickers())} tickers.")
    summary = store.get_universe_summary()
    print(summary.head())
