import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math

from analysis.research_paper_analyzer import ResearchPaperAnalyzer


ARXIV_SAMPLE = """<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2412.12345v1</id>
    <published>2024-12-21T00:00:00Z</published>
    <title>NVDA Option Pricing with volatility 25% and risk-free rate 3%</title>
    <summary>
      We study a contract with spot price 120 and strike price 110,
      maturity 1.5 and dividend yield 1%.
    </summary>
    <author><name>Jane Quant</name></author>
  </entry>
</feed>
"""


class _Resp:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


class _FakeMarketDataFetcher:
    def __init__(self, ticker=None, use_cache=True):
        self.ticker = ticker

    def get_current_spot_price(self):
        return 222.0


def test_extract_option_inputs_from_text():
    analyzer = ResearchPaperAnalyzer()
    parsed = analyzer.extract_option_inputs(
        "spot price 150 strike 140 volatility 30% risk-free rate 5% maturity 0.5 dividend yield 1%"
    )

    assert parsed["S"] == 150
    assert parsed["K"] == 140
    assert parsed["sigma"] == 0.3
    assert parsed["r"] == 0.05
    assert parsed["T"] == 0.5
    assert parsed["q"] == 0.01


def test_analyze_and_price_with_mocked_arxiv(monkeypatch):
    analyzer = ResearchPaperAnalyzer()

    def _fake_get(*args, **kwargs):
        return _Resp(ARXIV_SAMPLE)

    monkeypatch.setattr("analysis.research_paper_analyzer.requests.get", _fake_get)

    result = analyzer.analyze_and_price("2412.12345", option_type="call")

    assert result["paper"]["title"].startswith("NVDA Option")
    assert result["ticker"] == "NVDA"
    assert result["extracted_params"]["sigma"] == 0.25
    assert result["pricing"]["price"] > 0
    assert math.isfinite(result["pricing"]["gamma"])


def test_realtime_update_overrides_spot_with_market_data(monkeypatch):
    analyzer = ResearchPaperAnalyzer()

    def _fake_get(*args, **kwargs):
        return _Resp(ARXIV_SAMPLE)

    monkeypatch.setattr("analysis.research_paper_analyzer.requests.get", _fake_get)
    monkeypatch.setattr("analysis.research_paper_analyzer.MarketDataFetcher", _FakeMarketDataFetcher)

    base = analyzer.analyze_and_price("2412.12345", option_type="call")
    updated = analyzer.update_pricing_with_realtime_market(base, option_type="call")

    assert updated["extracted_params"]["S"] == 222.0
    assert "updated_at" in updated
    assert updated["pricing"]["price"] > 0


def test_realtime_stream_produces_multiple_snapshots(monkeypatch):
    analyzer = ResearchPaperAnalyzer()

    def _fake_get(*args, **kwargs):
        return _Resp(ARXIV_SAMPLE)

    monkeypatch.setattr("analysis.research_paper_analyzer.requests.get", _fake_get)
    monkeypatch.setattr("analysis.research_paper_analyzer.MarketDataFetcher", _FakeMarketDataFetcher)

    snapshots = list(analyzer.stream_realtime_updates("2412.12345", interval_seconds=0.0, iterations=2))

    assert len(snapshots) == 2
    assert all(s["extracted_params"]["S"] == 222.0 for s in snapshots)
