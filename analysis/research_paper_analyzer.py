"""Research paper ingestion and Black-Scholes integration utilities."""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional

import requests

from data.fetch_market_data import MarketDataFetcher
from models.pricing.black_scholes import BlackScholesModel


@dataclass
class PaperMetadata:
    """Normalized metadata extracted from arXiv response."""

    arxiv_id: str
    title: str
    abstract: str
    authors: List[str]
    published: str
    url: str


class ResearchPaperAnalyzer:
    """Fetch and transform research-paper signals into pricing analytics."""

    ARXIV_APIS = ("https://export.arxiv.org/api/query", "http://export.arxiv.org/api/query")

    def __init__(self, llm_api_key_env: str = "OPENAI_API_KEY", http_timeout: int = 20):
        self.llm_api_key_env = llm_api_key_env
        self.http_timeout = http_timeout

    def fetch_arxiv_paper(self, arxiv_id: str) -> PaperMetadata:
        """Fetch metadata and abstract for a single arXiv paper."""
        response = None
        last_error = None
        for endpoint in self.ARXIV_APIS:
            try:
                response = requests.get(
                    endpoint,
                    params={"id_list": arxiv_id.strip()},
                    timeout=self.http_timeout,
                )
                response.raise_for_status()
                break
            except requests.RequestException as exc:
                last_error = exc

        if response is None:
            raise ConnectionError(f"Unable to fetch arXiv paper: {last_error}")

        if not response.text.lstrip().startswith("<"):
            preview = response.text.strip()[:120]
            raise ConnectionError(f"arXiv returned non-XML payload: {preview}")

        ns = {"atom": "http://www.w3.org/2005/Atom"}
        try:
            root = ET.fromstring(response.text)
        except ET.ParseError as exc:
            preview = response.text.strip()[:120]
            raise ConnectionError(f"Unable to parse arXiv response: {preview}") from exc
        entry = root.find("atom:entry", ns)
        if entry is None:
            raise ValueError(f"No paper found for arXiv id '{arxiv_id}'")

        title = self._text(entry, "atom:title", ns)
        abstract = self._text(entry, "atom:summary", ns)
        published = self._text(entry, "atom:published", ns)
        url = self._text(entry, "atom:id", ns)
        authors = [a.find("atom:name", ns).text.strip() for a in entry.findall("atom:author", ns)]

        return PaperMetadata(
            arxiv_id=arxiv_id,
            title=title,
            abstract=abstract,
            authors=authors,
            published=published,
            url=url,
        )

    def analyze_and_price(self, arxiv_id: str, option_type: str = "call") -> Dict:
        """Fetch paper, infer parameters from text, and return pricing analytics."""
        paper = self.fetch_arxiv_paper(arxiv_id)
        params = self.extract_option_inputs(f"{paper.title}. {paper.abstract}")

        model = BlackScholesModel(
            S=params["S"],
            K=params["K"],
            T=params["T"],
            r=params["r"],
            sigma=params["sigma"],
            q=params["q"],
        )

        return {
            "paper": asdict(paper),
            "ticker": self.extract_ticker(f"{paper.title} {paper.abstract}"),
            "llm_key_detected": bool(os.getenv(self.llm_api_key_env)),
            "extracted_params": params,
            "pricing": model.price_and_greeks(option_type),
            "option_type": option_type.lower(),
        }

    def update_pricing_with_realtime_market(
        self,
        paper_payload: Dict,
        ticker: Optional[str] = None,
        option_type: str = "call",
    ) -> Dict:
        """Refresh paper-derived model inputs with latest market spot and decayed maturity."""
        params = dict(paper_payload["extracted_params"])
        inferred_ticker = ticker or paper_payload.get("ticker") or self.extract_ticker(
            f"{paper_payload['paper'].get('title', '')} {paper_payload['paper'].get('abstract', '')}"
        )

        if inferred_ticker:
            fetcher = MarketDataFetcher(ticker=inferred_ticker, use_cache=False)
            params["S"] = float(fetcher.get_current_spot_price())

        paper_published = paper_payload["paper"].get("published", "")
        if paper_published:
            try:
                published_dt = datetime.fromisoformat(paper_published.replace("Z", "+00:00"))
                elapsed_years = max((datetime.now(timezone.utc) - published_dt).total_seconds(), 0.0) / (365.0 * 24 * 3600)
                params["T"] = max(params["T"] - elapsed_years, 1e-6)
            except ValueError:
                pass

        model = BlackScholesModel(
            S=params["S"],
            K=params["K"],
            T=params["T"],
            r=params["r"],
            sigma=params["sigma"],
            q=params["q"],
        )

        return {
            **paper_payload,
            "ticker": inferred_ticker,
            "extracted_params": params,
            "pricing": model.price_and_greeks(option_type),
            "option_type": option_type.lower(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def stream_realtime_updates(
        self,
        arxiv_id: str,
        option_type: str = "call",
        interval_seconds: float = 30.0,
        iterations: int = 3,
    ) -> Iterator[Dict]:
        """Yield repeated real-time pricing updates from paper-derived assumptions."""
        base_payload = self.analyze_and_price(arxiv_id=arxiv_id, option_type=option_type)
        for idx in range(iterations):
            yield self.update_pricing_with_realtime_market(base_payload, option_type=option_type)
            if idx < iterations - 1:
                time.sleep(interval_seconds)

    def extract_option_inputs(self, text: str) -> Dict[str, float]:
        """Heuristically infer option inputs from paper title/abstract text."""
        cleaned = " ".join(text.replace("\n", " ").split())

        values = {
            "S": self._extract_number(cleaned, [r"(?:spot|stock|asset)\s*(?:price)?\s*(?:=|of)?\s*\$?([0-9]+(?:\.[0-9]+)?)"], 100.0),
            "K": self._extract_number(cleaned, [r"(?:strike|exercise)\s*(?:price)?\s*(?:=|of)?\s*\$?([0-9]+(?:\.[0-9]+)?)"], 100.0),
            "sigma": self._extract_number(cleaned, [r"(?:volatility|sigma|σ)\s*(?:=|of)?\s*([0-9]+(?:\.[0-9]+)?)"], 0.2),
            "r": self._extract_number(cleaned, [r"(?:risk[-\s]?free\s*rate|interest\s*rate|rate\s*r)\s*(?:=|of)?\s*([0-9]+(?:\.[0-9]+)?)"], 0.03),
            "T": self._extract_number(cleaned, [r"(?:maturity|tenor|expiry|T)\s*(?:=|of)?\s*([0-9]+(?:\.[0-9]+)?)"], 1.0),
            "q": self._extract_number(cleaned, [r"(?:dividend\s*yield|yield\s*q|q)\s*(?:=|of)?\s*([0-9]+(?:\.[0-9]+)?)"], 0.0),
        }

        if re.search(r"volatility[^.]{0,20}%", cleaned, re.IGNORECASE) and values["sigma"] >= 1:
            values["sigma"] /= 100.0
        if re.search(r"risk[-\s]?free[^.]{0,20}%", cleaned, re.IGNORECASE) and values["r"] >= 1:
            values["r"] /= 100.0
        if re.search(r"dividend[^.]{0,20}%", cleaned, re.IGNORECASE) and values["q"] >= 1:
            values["q"] /= 100.0

        values["sigma"] = max(values["sigma"], 1e-6)
        values["T"] = max(values["T"], 1e-6)
        return values

    @staticmethod
    def extract_ticker(text: str) -> Optional[str]:
        """Extract likely market ticker symbols from text."""
        candidates = re.findall(r"\b[A-Z]{1,5}\b", text)
        ignore = {"BS", "PDE", "SDE", "HMM", "GAN", "RL", "AI", "ML", "USD"}
        for symbol in candidates:
            if symbol not in ignore:
                return symbol
        return None

    @staticmethod
    def _extract_number(text: str, patterns: List[str], default: float) -> float:
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return float(match.group(1))
        return default

    @staticmethod
    def _text(entry: ET.Element, path: str, ns: Dict[str, str]) -> str:
        node = entry.find(path, ns)
        if node is None or node.text is None:
            return ""
        return " ".join(node.text.split())
