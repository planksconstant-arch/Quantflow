"""
High-Performance Limit Order Book (LOB) Engine
==============================================
Event-driven Level 2 / Level 3 Limit Order Book with price-time priority matching,
queue position estimation, multi-level depth ladders, and synthetic tick generator.
"""

from dataclasses import dataclass, field
from enum import Enum
import heapq
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    LIMIT = "limit"
    MARKET = "market"
    CANCEL = "cancel"


@dataclass
class LimitOrder:
    """Individual Level 3 limit order in queue"""
    order_id: str
    side: OrderSide
    price: float
    volume: float
    timestamp: float
    remaining_volume: float = field(init=False)
    
    def __post_init__(self):
        self.remaining_volume = self.volume


@dataclass
class L2Snapshot:
    """Level 2 aggregated order book snapshot"""
    timestamp: float
    mid_price: float
    micro_price: float
    spread: float
    bid_prices: np.ndarray      # Shape (K,) descending
    bid_volumes: np.ndarray     # Shape (K,)
    ask_prices: np.ndarray      # Shape (K,) ascending
    ask_volumes: np.ndarray     # Shape (K,)
    total_bid_depth: float
    total_ask_depth: float
    book_imbalance: float       # (BidVol - AskVol) / (BidVol + AskVol)


class LimitOrderBook:
    """
    Level 2 / Level 3 Limit Order Book with fast dictionary and sorted-list index.
    """
    def __init__(self, tick_size: float = 0.01):
        self.tick_size = tick_size
        self.bids: Dict[float, List[LimitOrder]] = {}  # Price -> list of buy orders (FIFO)
        self.asks: Dict[float, List[LimitOrder]] = {}  # Price -> list of sell orders (FIFO)
        self.orders: Dict[str, LimitOrder] = {}        # Order ID -> LimitOrder
        self.last_update_time: float = 0.0
        self.trade_history: List[Dict] = []

    def round_tick(self, price: float) -> float:
        """Round price to nearest tick size"""
        return round(round(price / self.tick_size) * self.tick_size, 4)

    @property
    def best_bid(self) -> Optional[float]:
        valid_bids = [p for p in self.bids if len(self.bids[p]) > 0]
        return max(valid_bids) if valid_bids else None

    @property
    def best_ask(self) -> Optional[float]:
        valid_asks = [p for p in self.asks if len(self.asks[p]) > 0]
        return min(valid_asks) if valid_asks else None

    @property
    def mid_price(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return round((bb + ba) / 2.0, 4)
        return bb or ba

    @property
    def spread(self) -> Optional[float]:
        bb = self.best_bid
        ba = self.best_ask
        if bb is not None and ba is not None:
            return round(ba - bb, 4)
        return None

    def add_limit_order(
        self,
        order_id: str,
        side: OrderSide,
        price: float,
        volume: float,
        timestamp: float = 0.0,
    ) -> LimitOrder:
        """Add limit order to queue with price-time priority"""
        price = self.round_tick(price)
        order = LimitOrder(
            order_id=order_id,
            side=side,
            price=price,
            volume=float(volume),
            timestamp=timestamp,
        )
        self.orders[order_id] = order
        self.last_update_time = timestamp

        target_dict = self.bids if side == OrderSide.BUY else self.asks
        if price not in target_dict:
            target_dict[price] = []
        target_dict[price].append(order)

        return order

    def cancel_order(self, order_id: str) -> bool:
        """Cancel existing order from book"""
        if order_id not in self.orders:
            return False
        order = self.orders.pop(order_id)
        target_dict = self.bids if order.side == OrderSide.BUY else self.asks
        if order.price in target_dict:
            target_dict[order.price] = [o for o in target_dict[order.price] if o.order_id != order_id]
            if len(target_dict[order.price]) == 0:
                del target_dict[order.price]
        return True

    def execute_market_order(
        self,
        side: OrderSide,
        volume: float,
        timestamp: float = 0.0,
    ) -> Dict:
        """
        Execute aggressive market order against resting limit orders.
        Returns fill details (filled_volume, vwap_price, slippage).
        """
        remaining_fill = float(volume)
        total_cost = 0.0
        filled_trades = []
        
        # Buying consumes asks (lowest price first)
        # Selling consumes bids (highest price first)
        if side == OrderSide.BUY:
            while remaining_fill > 1e-6 and self.best_ask is not None:
                ask_p = self.best_ask
                queue = self.asks[ask_p]
                while queue and remaining_fill > 1e-6:
                    resting = queue[0]
                    fill_amt = min(remaining_fill, resting.remaining_volume)
                    resting.remaining_volume -= fill_amt
                    remaining_fill -= fill_amt
                    total_cost += fill_amt * ask_p
                    
                    filled_trades.append({
                        "order_id": resting.order_id,
                        "price": ask_p,
                        "volume": fill_amt,
                        "side": "buy_against_ask",
                        "timestamp": timestamp,
                    })
                    
                    if resting.remaining_volume <= 1e-6:
                        queue.pop(0)
                        self.orders.pop(resting.order_id, None)
                        
                if len(queue) == 0:
                    del self.asks[ask_p]
        else: # OrderSide.SELL
            while remaining_fill > 1e-6 and self.best_bid is not None:
                bid_p = self.best_bid
                queue = self.bids[bid_p]
                while queue and remaining_fill > 1e-6:
                    resting = queue[0]
                    fill_amt = min(remaining_fill, resting.remaining_volume)
                    resting.remaining_volume -= fill_amt
                    remaining_fill -= fill_amt
                    total_cost += fill_amt * bid_p
                    
                    filled_trades.append({
                        "order_id": resting.order_id,
                        "price": bid_p,
                        "volume": fill_amt,
                        "side": "sell_against_bid",
                        "timestamp": timestamp,
                    })
                    
                    if resting.remaining_volume <= 1e-6:
                        queue.pop(0)
                        self.orders.pop(resting.order_id, None)
                        
                if len(queue) == 0:
                    del self.bids[bid_p]

        filled_vol = volume - remaining_fill
        vwap = (total_cost / filled_vol) if filled_vol > 0 else 0.0
        
        trade_record = {
            "side": side.value,
            "requested_volume": volume,
            "filled_volume": filled_vol,
            "vwap_price": vwap,
            "trades": filled_trades,
            "timestamp": timestamp,
        }
        self.trade_history.append(trade_record)
        return trade_record

    def get_queue_position(self, order_id: str) -> Optional[int]:
        """Get number of shares ahead of this order in the price queue"""
        if order_id not in self.orders:
            return None
        order = self.orders[order_id]
        target_dict = self.bids if order.side == OrderSide.BUY else self.asks
        if order.price not in target_dict:
            return None
        ahead = 0
        for o in target_dict[order.price]:
            if o.order_id == order_id:
                break
            ahead += o.remaining_volume
        return ahead

    def get_l2_snapshot(self, depth: int = 10) -> L2Snapshot:
        """Construct Level 2 aggregated order book snapshot"""
        # Sorted bids (descending)
        valid_bids = sorted([p for p in self.bids if len(self.bids[p]) > 0], reverse=True)[:depth]
        bid_prices = np.array(valid_bids, dtype=float)
        bid_volumes = np.array([sum(o.remaining_volume for o in self.bids[p]) for p in valid_bids], dtype=float)

        # Sorted asks (ascending)
        valid_asks = sorted([p for p in self.asks if len(self.asks[p]) > 0])[:depth]
        ask_prices = np.array(valid_asks, dtype=float)
        ask_volumes = np.array([sum(o.remaining_volume for o in self.asks[p]) for p in valid_asks], dtype=float)

        bb = bid_prices[0] if len(bid_prices) > 0 else 100.0
        ba = ask_prices[0] if len(ask_prices) > 0 else 100.05
        mid = round((bb + ba) / 2.0, 4)
        spread = round(ba - bb, 4)

        tot_bid = float(np.sum(bid_volumes)) if len(bid_volumes) > 0 else 1.0
        tot_ask = float(np.sum(ask_volumes)) if len(ask_volumes) > 0 else 1.0
        imbalance = (tot_bid - tot_ask) / (tot_bid + tot_ask)

        # Stoikov micro-price: Mid + ((Vb - Va) / (Vb + Va)) * (Spread / 2)
        top_vb = bid_volumes[0] if len(bid_volumes) > 0 else 1.0
        top_va = ask_volumes[0] if len(ask_volumes) > 0 else 1.0
        top_imb = (top_vb - top_va) / (top_vb + top_va)
        micro_price = round(mid + top_imb * (spread / 2.0), 4)

        return L2Snapshot(
            timestamp=self.last_update_time,
            mid_price=mid,
            micro_price=micro_price,
            spread=spread,
            bid_prices=bid_prices,
            bid_volumes=bid_volumes,
            ask_prices=ask_prices,
            ask_volumes=ask_volumes,
            total_bid_depth=tot_bid,
            total_ask_depth=tot_ask,
            book_imbalance=float(imbalance),
        )


def generate_synthetic_lob_stream(
    n_ticks: int = 500,
    initial_price: float = 140.0,
    annual_vol: float = 0.35,
    tick_size: float = 0.01,
    seed: int = 42,
) -> Tuple[List[L2Snapshot], pd.DataFrame]:
    """
    Generate realistic high-frequency Level 2 LOB sequence with Poisson arrival,
    Hawkes jump clusters, and dynamic liquidity replenishment.
    """
    np.random.seed(seed)
    book = LimitOrderBook(tick_size=tick_size)
    
    dt = 1.0 / (252 * 6.5 * 3600)  # Micro-time step (~1 second in trading time)
    vol_step = annual_vol * np.sqrt(dt) * initial_price
    
    current_mid = initial_price
    snapshots: List[L2Snapshot] = []
    tick_records = []
    order_counter = 0

    # Initial order book population
    for i in range(1, 15):
        p_bid = book.round_tick(current_mid - i * tick_size)
        p_ask = book.round_tick(current_mid + i * tick_size)
        v_bid = int(np.random.gamma(shape=3.0, scale=100))
        v_ask = int(np.random.gamma(shape=3.0, scale=100))
        
        book.add_limit_order(f"init_b_{order_counter}", OrderSide.BUY, p_bid, v_bid, 0.0)
        order_counter += 1
        book.add_limit_order(f"init_a_{order_counter}", OrderSide.SELL, p_ask, v_ask, 0.0)
        order_counter += 1

    # Simulate dynamic market events
    for t in range(1, n_ticks + 1):
        timestamp = t * 0.1
        
        # Underlying mid-price drift & jump
        jump_occurred = np.random.uniform(0, 1) < 0.05
        jump_size = np.random.normal(0, 3 * vol_step) if jump_occurred else 0.0
        current_mid += np.random.normal(0, vol_step) + jump_size
        current_mid = max(1.0, current_mid)

        # Poisson order flow arrival
        num_events = np.random.poisson(lam=4)
        for _ in range(num_events):
            event_type = np.random.choice(["limit_buy", "limit_sell", "market_buy", "market_sell", "cancel"], p=[0.35, 0.35, 0.12, 0.12, 0.06])
            
            if event_type == "limit_buy":
                # Price placed near best bid
                best_b = book.best_bid or current_mid - tick_size
                p = book.round_tick(best_b - np.random.choice([0, 1, 2, 3]) * tick_size)
                v = int(np.random.gamma(shape=2.5, scale=80))
                book.add_limit_order(f"ord_{order_counter}", OrderSide.BUY, p, v, timestamp)
                order_counter += 1
                
            elif event_type == "limit_sell":
                best_a = book.best_ask or current_mid + tick_size
                p = book.round_tick(best_a + np.random.choice([0, 1, 2, 3]) * tick_size)
                v = int(np.random.gamma(shape=2.5, scale=80))
                book.add_limit_order(f"ord_{order_counter}", OrderSide.SELL, p, v, timestamp)
                order_counter += 1
                
            elif event_type == "market_buy":
                v = int(np.random.gamma(shape=2.0, scale=60))
                book.execute_market_order(OrderSide.BUY, v, timestamp)
                
            elif event_type == "market_sell":
                v = int(np.random.gamma(shape=2.0, scale=60))
                book.execute_market_order(OrderSide.SELL, v, timestamp)
                
            elif event_type == "cancel":
                if len(book.orders) > 10:
                    random_oid = np.random.choice(list(book.orders.keys()))
                    book.cancel_order(random_oid)

        # Replenish if book becomes too thin
        if book.best_bid is None or book.best_ask is None or (book.best_ask - book.best_bid) > 10 * tick_size:
            p_bid = book.round_tick(current_mid - tick_size)
            p_ask = book.round_tick(current_mid + tick_size)
            book.add_limit_order(f"repl_b_{order_counter}", OrderSide.BUY, p_bid, 500, timestamp)
            order_counter += 1
            book.add_limit_order(f"repl_a_{order_counter}", OrderSide.SELL, p_ask, 500, timestamp)
            order_counter += 1

        snap = book.get_l2_snapshot(depth=10)
        snapshots.append(snap)
        
        tick_records.append({
            "timestamp": timestamp,
            "mid_price": snap.mid_price,
            "micro_price": snap.micro_price,
            "spread": snap.spread,
            "best_bid": snap.bid_prices[0] if len(snap.bid_prices) > 0 else np.nan,
            "best_ask": snap.ask_prices[0] if len(snap.ask_prices) > 0 else np.nan,
            "bid_vol_l1": snap.bid_volumes[0] if len(snap.bid_volumes) > 0 else 0,
            "ask_vol_l1": snap.ask_volumes[0] if len(snap.ask_volumes) > 0 else 0,
            "total_bid_depth": snap.total_bid_depth,
            "total_ask_depth": snap.total_ask_depth,
            "book_imbalance": snap.book_imbalance,
        })

    df = pd.DataFrame(tick_records)
    return snapshots, df
