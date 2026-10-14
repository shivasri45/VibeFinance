"""
VibeFinance Technical Indicators (Pandas implementation by hand)
Computes: RSI-14, MACD (12, 26, 9), SMA20/50, 20d return, volume z-score vs 30d.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index using Wilder's smoothing method."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Exponential moving average with alpha = 1 / period
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """Calculate MACD line, Signal line, and MACD Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_hist": macd_hist,
        }
    )


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all standard indicators for a single ticker's OHLCV DataFrame.
    Expected columns: ['Open', 'High', 'Low', 'Close', 'Volume']
    Adds:
      - rsi_14
      - macd, macd_signal, macd_hist
      - sma_20, sma_50
      - ret_20d
      - vol_zscore_30d
    """
    df = df.copy()
    if df.empty or len(df) < 5:
        return df

    close = df["Close"]
    volume = df["Volume"]

    # 1. RSI-14
    df["rsi_14"] = compute_rsi(close, 14)

    # 2. MACD (12, 26, 9)
    macd_df = compute_macd(close, 12, 26, 9)
    df["macd"] = macd_df["macd"]
    df["macd_signal"] = macd_df["macd_signal"]
    df["macd_hist"] = macd_df["macd_hist"]

    # 3. SMAs
    df["sma_20"] = close.rolling(window=20, min_periods=10).mean()
    df["sma_50"] = close.rolling(window=50, min_periods=20).mean()

    # 4. 20-day return (% change)
    df["ret_20d"] = close.pct_change(periods=20)

    # 5. Volume z-score vs 30d
    vol_mean_30 = volume.rolling(window=30, min_periods=10).mean()
    vol_std_30 = volume.rolling(window=30, min_periods=10).std()
    df["vol_zscore_30d"] = (volume - vol_mean_30) / (vol_std_30 + 1e-6)

    return df


def get_latest_indicators(df: pd.DataFrame) -> Dict[str, Any]:
    """Extract latest indicators formatted cleanly for agent evidence."""
    if df.empty:
        return {}
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    
    rsi = float(latest.get("rsi_14", 50.0))
    macd = float(latest.get("macd", 0.0))
    macd_signal = float(latest.get("macd_signal", 0.0))
    macd_hist = float(latest.get("macd_hist", 0.0))
    sma20 = float(latest.get("sma_20", latest["Close"]))
    sma50 = float(latest.get("sma_50", latest["Close"]))
    close = float(latest["Close"])
    ret_20d = float(latest.get("ret_20d", 0.0))
    vol_z = float(latest.get("vol_zscore_30d", 0.0))
    volume = float(latest["Volume"])
    
    # Flags
    bullish_cross = (prev.get("sma_20", 0) <= prev.get("sma_50", 0)) and (sma20 > sma50)
    bearish_cross = (prev.get("sma_20", 0) >= prev.get("sma_50", 0)) and (sma20 < sma50)
    
    return {
        "close": close,
        "volume": volume,
        "rsi_14": round(rsi, 2) if not np.isnan(rsi) else 50.0,
        "macd": round(macd, 2) if not np.isnan(macd) else 0.0,
        "macd_signal": round(macd_signal, 2) if not np.isnan(macd_signal) else 0.0,
        "macd_hist": round(macd_hist, 2) if not np.isnan(macd_hist) else 0.0,
        "sma_20": round(sma20, 2),
        "sma_50": round(sma50, 2),
        "ret_20d": round(ret_20d, 4) if not np.isnan(ret_20d) else 0.0,
        "vol_zscore_30d": round(vol_z, 2) if not np.isnan(vol_z) else 0.0,
        "price_vs_sma20_pct": round(((close - sma20) / sma20) * 100, 2) if sma20 > 0 else 0.0,
        "price_vs_sma50_pct": round(((close - sma50) / sma50) * 100, 2) if sma50 > 0 else 0.0,
        "bullish_cross": bool(bullish_cross),
        "bearish_cross": bool(bearish_cross),
    }
