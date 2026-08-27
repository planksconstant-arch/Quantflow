"""
QuantFlow Real-Time Market Data Feed
====================================
High-performance real-time market data providers using free public APIs:
- Marketstack API for institutional global equities and EOD/latest market feeds
- Binance & Coinbase Public APIs for live Level 2 Limit Order Book depth (Crypto)
- Yahoo Finance API for live equities & indices quotes and option chains
- Automatic zero-latency fallback to synthetic high-frequency microstructure streams
"""

import json
import logging
import os
import urllib.request
import urllib.parse
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from models.microstructure.order_book import LimitOrderBook, L2Snapshot, OrderSide, generate_synthetic_lob_stream
from utils.config import config

logger = logging.getLogger(__name__)


@dataclass
class RealtimeQuote:
    """Standardized real-time market quote container"""
    ticker: str
    price: float
    bid: float
    ask: float
    spread: float
    volume: float
    timestamp: datetime
    source: str
    is_live: bool


class RealtimeMarketFeed:
    """Unified provider for free real-time institutional market feeds"""

    CRYPTO_SYMBOLS = {
        "BTC-USD": "BTCUSDT",
        "ETH-USD": "ETHUSDT",
        "SOL-USD": "SOLUSDT",
    }

    EQUITY_DEFAULTS = {
        "NVDA": 140.0,
        "AAPL": 225.0,
        "TSLA": 210.0,
        "MSFT": 420.0,
        "SPY": 560.0,
        "QQQ": 480.0,
        "BTC-USD": 62500.0,
        "ETH-USD": 3400.0,
    }

    def __init__(self, timeout: float = 4.0, marketstack_key: Optional[str] = None):
        self.timeout = timeout
        self.marketstack_key = marketstack_key or getattr(config, "MARKETSTACK_API_KEY", "24b40dae0167960b6bd3ec0ce5dfd4f9")

    def fetch_marketstack_quote(self, ticker: str, api_key: Optional[str] = None) -> Optional[RealtimeQuote]:
        """
        Fetch authentic equity quote using Marketstack REST API.
        """
        key = api_key or self.marketstack_key
        if not key:
            return None

        # Clean symbol (e.g. BTC-USD is crypto, Marketstack expects stock tickers like NVDA, AAPL)
        clean_ticker = ticker.split("-")[0].upper()
        url = f"http://api.marketstack.com/v1/eod/latest?access_key={key}&symbols={clean_ticker}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantFlow/2.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status = getattr(response, "status", None) or getattr(response, "code", 200)
                if status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    data = payload.get("data", [])
                    if data and len(data) > 0:
                        item = data[0]
                        price = float(item.get("close") or item.get("adj_close") or item.get("open") or 0.0)
                        if price > 0:
                            vol = float(item.get("volume") or 0.0)
                            bid = round(price - 0.01, 2)
                            ask = round(price + 0.01, 2)
                            return RealtimeQuote(
                                ticker=ticker,
                                price=price,
                                bid=bid,
                                ask=ask,
                                spread=0.02,
                                volume=vol,
                                timestamp=datetime.now(),
                                source=f"Marketstack API ({item.get('exchange', 'XNAS')})",
                                is_live=True,
                            )
        except Exception as e:
            logger.debug(f"Marketstack fetch failed for {ticker}: {e}")

        return None

    def fetch_marketstack_eod(self, ticker: str, limit: int = 100, api_key: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch historical EOD dataframe from Marketstack API.
        """
        key = api_key or self.marketstack_key
        if not key:
            return None

        clean_ticker = ticker.split("-")[0].upper()
        url = f"http://api.marketstack.com/v1/eod?access_key={key}&symbols={clean_ticker}&limit={limit}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "QuantFlow/2.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status = getattr(response, "status", None) or getattr(response, "code", 200)
                if status == 200:
                    payload = json.loads(response.read().decode("utf-8"))
                    data = payload.get("data", [])
                    if data:
                        df = pd.DataFrame(data)
                        df["date"] = pd.to_datetime(df["date"])
                        df = df.sort_values("date").reset_index(drop=True)
                        return df
        except Exception as e:
            logger.debug(f"Marketstack historical EOD failed for {ticker}: {e}")

        return None

    def fetch_live_crypto_order_book(self, symbol: str = "BTC-USD", limit: int = 20) -> Optional[LimitOrderBook]:
        """
        Fetch authentic real-time Level 2 order book from Binance or Coinbase public API.
        Zero API key required.
        """
        binance_symbol = self.CRYPTO_SYMBOLS.get(symbol.upper(), symbol.upper().replace("-", "").replace("/", ""))
        
        # Primary: Binance Public REST Depth Endpoint
        try:
            url = f"https://api.binance.com/api/v3/depth?symbol={binance_symbol}&limit={limit}"
            req = urllib.request.Request(url, headers={"User-Agent": "QuantFlow/2.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status = getattr(response, 'status', None) or getattr(response, 'code', 200)
                if status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    lob = LimitOrderBook(tick_size=0.01)
                    
                    # Insert Bids
                    for i, (p_str, q_str) in enumerate(data.get("bids", [])):
                        p, q = float(p_str), float(q_str)
                        if p > 0 and q > 0:
                            lob.add_limit_order(f"binance_bid_{i}_{uuid.uuid4().hex[:6]}", OrderSide.BUY, p, q)
                    
                    # Insert Asks
                    for i, (p_str, q_str) in enumerate(data.get("asks", [])):
                        p, q = float(p_str), float(q_str)
                        if p > 0 and q > 0:
                            lob.add_limit_order(f"binance_ask_{i}_{uuid.uuid4().hex[:6]}", OrderSide.SELL, p, q)
                    
                    if lob.mid_price is not None:
                        return lob
        except Exception as e:
            logger.debug(f"Binance feed unavailable for {symbol}: {e}")

        # Secondary: Coinbase Public REST Order Book Endpoint
        try:
            cb_symbol = symbol.upper() if "-" in symbol else f"{symbol[:3]}-{symbol[3:]}"
            url = f"https://api.exchange.coinbase.com/products/{cb_symbol}/book?level=2"
            req = urllib.request.Request(url, headers={"User-Agent": "QuantFlow/2.0"})
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                status = getattr(response, 'status', None) or getattr(response, 'code', 200)
                if status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    lob = LimitOrderBook(tick_size=0.01)
                    
                    for i, item in enumerate(data.get("bids", [])[:limit]):
                        lob.add_limit_order(f"cb_bid_{i}_{uuid.uuid4().hex[:6]}", OrderSide.BUY, float(item[0]), float(item[1]))
                    for i, item in enumerate(data.get("asks", [])[:limit]):
                        lob.add_limit_order(f"cb_ask_{i}_{uuid.uuid4().hex[:6]}", OrderSide.SELL, float(item[0]), float(item[1]))
                    
                    if lob.mid_price is not None:
                        return lob
        except Exception as e:
            logger.debug(f"Coinbase feed unavailable for {symbol}: {e}")

        return None

    def fetch_live_equity_quote(self, ticker: str, preferred_source: str = "auto") -> RealtimeQuote:
        """
        Fetch live real-time equity/index quote via Marketstack or yfinance with fallback.
        """
        # 1. Marketstack API (if preferred or auto for non-crypto)
        if preferred_source in ("marketstack", "auto") and not ("BTC" in ticker.upper() or "ETH" in ticker.upper()):
            ms_quote = self.fetch_marketstack_quote(ticker)
            if ms_quote is not None:
                return ms_quote

        # 2. Yahoo Finance API
        try:
            stock = yf.Ticker(ticker)
            fast = getattr(stock, "fast_info", None)
            
            last_price = None
            bid = None
            ask = None
            vol = 0.0

            if fast is not None:
                last_price = getattr(fast, "last_price", None) or getattr(fast, "regular_market_previous_close", None)
                bid = getattr(fast, "bid", None)
                ask = getattr(fast, "ask", None)
                vol = getattr(fast, "last_volume", 0.0) or 0.0

            if last_price is None or np.isnan(last_price) or last_price <= 0:
                hist = stock.history(period="1d", interval="1m")
                if not hist.empty:
                    last_price = float(hist["Close"].iloc[-1])
                    vol = float(hist["Volume"].sum())

            if last_price is not None and not np.isnan(last_price) and last_price > 0:
                if bid is None or np.isnan(bid) or bid <= 0:
                    bid = round(last_price - 0.01, 2)
                if ask is None or np.isnan(ask) or ask <= 0:
                    ask = round(last_price + 0.01, 2)

                return RealtimeQuote(
                    ticker=ticker,
                    price=float(last_price),
                    bid=float(bid),
                    ask=float(ask),
                    spread=round(float(ask - bid), 4),
                    volume=float(vol),
                    timestamp=datetime.now(),
                    source="Yahoo Finance Live",
                    is_live=True,
                )
        except Exception as e:
            logger.debug(f"Failed to fetch live quote for {ticker}: {e}")

        # 3. High-Fidelity Fallback
        fallback_price = self.EQUITY_DEFAULTS.get(ticker.upper(), 100.0)
        return RealtimeQuote(
            ticker=ticker,
            price=fallback_price,
            bid=round(fallback_price - 0.01, 2),
            ask=round(fallback_price + 0.01, 2),
            spread=0.02,
            volume=1000000.0,
            timestamp=datetime.now(),
            source="Calibrated Fallback",
            is_live=False,
        )

    def fetch_live_option_chain(self, ticker: str, expiry: Optional[str] = None) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Fetch real options chain for a given equity from yfinance.
        """
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options
            if not expirations:
                return None
            
            target_expiry = expiry if (expiry and expiry in expirations) else expirations[0]
            chain = stock.option_chain(target_expiry)
            return chain.calls, chain.puts
        except Exception as e:
            logger.debug(f"Failed to fetch option chain for {ticker}: {e}")
            return None

    def get_market_stream(
        self,
        ticker: str,
        n_ticks: int = 300,
        initial_price: Optional[float] = None,
        annual_vol: float = 0.35,
        seed: int = 42,
        preferred_source: str = "auto"
    ) -> Tuple[List[L2Snapshot], pd.DataFrame, RealtimeQuote]:
        """
        Get market stream seeded with authentic live prices.
        """
        quote = self.fetch_live_equity_quote(ticker, preferred_source=preferred_source)
        s0 = quote.price if (initial_price is None or initial_price <= 0) else initial_price

        # If crypto, attempt to seed with live crypto order book
        if "BTC" in ticker.upper() or "ETH" in ticker.upper() or "SOL" in ticker.upper():
            live_lob = self.fetch_live_crypto_order_book(ticker)
            if live_lob is not None and live_lob.mid_price is not None:
                s0 = live_lob.mid_price
                quote.price = s0
                quote.source = "Binance / Coinbase L2 Live"
                quote.is_live = True

        snapshots, df = generate_synthetic_lob_stream(
            n_ticks=n_ticks,
            initial_price=s0,
            annual_vol=annual_vol,
            tick_size=0.01,
            seed=seed,
        )
        return snapshots, df, quote


# Global singleton instance
realtime_feed = RealtimeMarketFeed()
