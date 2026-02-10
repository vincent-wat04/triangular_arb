import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import telebot
import websocket
from dotenv import load_dotenv

from core.arb_engine import VectorizedTriangleArbitrage
from core.rates import calculate_rates


class ArbBotConfig:
    def __init__(self):
        load_dotenv()
        self.bot_token = os.getenv("BOT_TOKEN", "")
        self.chat_id = os.getenv("CHAT_ID", "")
        self.min_profit = float(os.getenv("MIN_PROFIT", "0.0001"))
        self.fee_rate = float(os.getenv("FEE_RATE", "0.001"))
        self.slippage_bps = float(os.getenv("SLIPPAGE_BPS", "3.0"))
        self.latency_ms = float(os.getenv("LATENCY_MS", "50.0"))
        self.drift_bps_per_sec = float(os.getenv("DRIFT_BPS_PER_SEC", "2.0"))
        self.compute_interval_sec = float(os.getenv("COMPUTE_INTERVAL_SEC", "0"))
        self.use_incremental = os.getenv("USE_INCREMENTAL", "true").lower() == "true"
        self.hourly_interval_sec = int(os.getenv("HOURLY_INTERVAL_SEC", "3600"))
        self.opp_throttle_sec = int(os.getenv("OPP_THROTTLE_SEC", "5"))
        self.watchlist_path = Path(os.getenv("WATCHLIST_PATH", "data/binance_watchlist.json"))


class ArbBot:
    def __init__(self, config: ArbBotConfig):
        self.config = config
        if not self.config.bot_token or not self.config.chat_id:
            raise SystemExit("Missing BOT_TOKEN or CHAT_ID in .env")

        self.bot = telebot.TeleBot(self.config.bot_token)
        self.detector = VectorizedTriangleArbitrage(fee_rate=self.config.fee_rate)

        data = json.loads(self.config.watchlist_path.read_text())
        self.watchlist = data["binance_watchlist"]

        self.last_ts: Dict[str, float] = {}
        self.intervals: Dict[str, List[float]] = {s: [] for s in self.watchlist}
        self.latest_quotes: Dict[str, np.ndarray] = {}
        self.data_lock = threading.Lock()

        self.last_hourly_push = 0.0
        self.last_opp_push = 0.0

    def send_message(self, text: str) -> None:
        self.bot.send_message(self.config.chat_id, text)

    def on_message(self, ws, message):
        payload = json.loads(message)
        symbol = payload["s"]
        bid = float(payload["b"])
        ask = float(payload["a"])
        now = time.perf_counter()
        with self.data_lock:
            self.latest_quotes[symbol] = np.array([bid, ask], dtype=np.float64)
            if symbol in self.last_ts:
                self.intervals[symbol].append((now - self.last_ts[symbol]) * 1000)
            self.last_ts[symbol] = now

    def on_open(self, ws):
        def run(*args):
            subscribe_message = {
                "method": "SUBSCRIBE",
                "params": [f"{symbol.lower()}@bookTicker" for symbol in self.watchlist],
                "id": 1,
            }
            ws.send(json.dumps(subscribe_message))
        threading.Thread(target=run).start()

    def on_error(self, ws, error):
        self.send_message(f"WebSocket error: {error}")

    def on_close(self, ws, close_status_code, close_msg):
        self.send_message("WebSocket closed")

    def _build_edges(self, symbol_indices: List[int]) -> List[Tuple[int, int]]:
        bases = self.detector.base_indices[symbol_indices]
        quotes = self.detector.quote_indices[symbol_indices]
        return list(set(zip(np.minimum(bases, quotes), np.maximum(bases, quotes))))

    def _snapshot(self) -> Tuple[List[str], np.ndarray]:
        with self.data_lock:
            if not self.latest_quotes:
                return [], np.array([])
            symbols = list(self.latest_quotes.keys())
            prices = list(self.latest_quotes.values())
        return symbols, np.vstack(prices)

    def compute_loop(self):
        while True:
            symbols, price_arr = self._snapshot()
            if price_arr.size == 0:
                time.sleep(0.01)
                continue

            symbol_indices = [self.detector.symbol_to_idx[s] for s in symbols]
            ij_rates, ji_rates = calculate_rates(
                price_arr[:, 0],
                price_arr[:, 1],
                self.config.fee_rate,
                self.config.slippage_bps,
                self.config.latency_ms,
                self.config.drift_bps_per_sec,
            )

            self.detector.update_price_vectorized_v2(symbol_indices, ij_rates, ji_rates)
            edges = self._build_edges(symbol_indices)

            if self.config.use_incremental:
                opps = self.detector.find_arbitrage_incremental(edges, min_profit=self.config.min_profit)
            else:
                opps = self.detector.find_arbitrage_vectorized(min_profit=self.config.min_profit)

            now = time.time()
            if now - self.last_hourly_push >= self.config.hourly_interval_sec:
                self.last_hourly_push = now
                self.send_message(
                    f"heartbeat: {len(opps)} opportunities at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            if opps and now - self.last_opp_push >= self.config.opp_throttle_sec:
                self.last_opp_push = now
                top = opps[0]
                msg = f"opportunity: profit={top['profit']:.4f}% path={top['path']}"
                self.send_message(msg)

            if self.config.compute_interval_sec > 0:
                time.sleep(self.config.compute_interval_sec)

    def run(self):
        socket_url = "wss://stream.binance.com:9443/ws"
        ws = websocket.WebSocketApp(
            socket_url,
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )

        t = threading.Thread(target=ws.run_forever)
        t.daemon = True
        t.start()

        self.send_message("triangular arb bot started")
        self.compute_loop()
