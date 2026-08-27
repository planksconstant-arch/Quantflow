"""Unit tests for free Real-Time Market Data Provider & Marketstack API."""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.realtime_feed import RealtimeMarketFeed, RealtimeQuote, realtime_feed
from models.microstructure.order_book import OrderSide


class TestRealtimeMarketFeed:
    def test_singleton_initialization(self):
        feed = RealtimeMarketFeed()
        assert feed is not None
        assert "BTC-USD" in feed.CRYPTO_SYMBOLS
        assert "NVDA" in feed.EQUITY_DEFAULTS
        assert feed.marketstack_key == "24b40dae0167960b6bd3ec0ce5dfd4f9"

    def test_marketstack_mocked_quote(self):
        feed = RealtimeMarketFeed(marketstack_key="test_key")
        mock_payload = b'{"data": [{"close": 210.50, "open": 209.00, "volume": 5000000.0, "exchange": "XNAS", "symbol": "NVDA"}]}'
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.code = 200
        mock_resp.read.return_value = mock_payload
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            quote = feed.fetch_marketstack_quote("NVDA")
            assert quote is not None
            assert quote.price == pytest.approx(210.50)
            assert quote.ticker == "NVDA"
            assert "Marketstack" in quote.source

    def test_marketstack_mocked_eod(self):
        feed = RealtimeMarketFeed(marketstack_key="test_key")
        mock_payload = b'{"data": [{"date": "2026-08-25T00:00:00+0000", "close": 209.0, "open": 208.0, "volume": 1000000.0}, {"date": "2026-08-26T00:00:00+0000", "close": 212.0, "open": 210.0, "volume": 1200000.0}]}'
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.code = 200
        mock_resp.read.return_value = mock_payload
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            df = feed.fetch_marketstack_eod("NVDA", limit=2)
            assert df is not None
            assert len(df) == 2
            assert "close" in df.columns

    def test_fallback_equity_quote(self):
        feed = RealtimeMarketFeed()
        quote = feed.fetch_live_equity_quote("NVDA", preferred_source="fallback")
        assert quote is not None
        assert quote.ticker == "NVDA"
        assert quote.price > 0
        assert quote.bid <= quote.ask
        assert quote.spread >= 0

    def test_market_stream_generation(self):
        feed = RealtimeMarketFeed()
        snapshots, df, quote = feed.get_market_stream(ticker="NVDA", n_ticks=50, seed=42)
        assert len(snapshots) == 50
        assert len(df) == 50
        assert quote.price > 0
        assert snapshots[-1].mid_price > 0
        assert snapshots[-1].spread >= 0

    def test_crypto_stream_generation(self):
        feed = RealtimeMarketFeed()
        snapshots, df, quote = feed.get_market_stream(ticker="BTC-USD", n_ticks=30, seed=42)
        assert len(snapshots) == 30
        assert quote.price > 0

    def test_mocked_binance_depth_success(self):
        feed = RealtimeMarketFeed()
        mock_payload = b'{"bids": [["60000.0", "1.5"], ["59990.0", "2.0"]], "asks": [["60010.0", "1.2"], ["60020.0", "3.0"]]}'
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.code = 200
        mock_resp.read.return_value = mock_payload
        mock_resp.__enter__.return_value = mock_resp

        with patch("urllib.request.urlopen", return_value=mock_resp):
            lob = feed.fetch_live_crypto_order_book("BTC-USD", limit=2)
            assert lob is not None
            assert lob.mid_price == pytest.approx(60005.0)
            assert lob.spread == pytest.approx(10.0)
