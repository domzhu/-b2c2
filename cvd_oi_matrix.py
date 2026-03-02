from __future__ import annotations

"""
Multi-timeframe CVD/OI monitor for open Bybit linear USDT positions.

"""

import argparse
import os
import time
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

from risk_guard.bybit_api import BybitClient


TIMEFRAMES_MIN = [1, 5, 15, 30, 60]
BYBIT_BASE_URL = "https://api.bybit.com"
HTTP_TIMEOUT_SEC = 10


@dataclass
class TFMetrics:
    symbol: str
    timeframe_min: int
    price_pct: Optional[float]
    oi_pct: Optional[float]
    cvd: Optional[float]
    cvd_pct: Optional[float]
    price_span_min: Optional[float]
    oi_span_min: Optional[float]
    cvd_span_min: Optional[float]


@dataclass
class PositionInfo:
    side: str
    size: float
    notional: float


@dataclass(frozen=True)
class TradeRecord:
    ts_ms: int
    side: str
    size: float
    key: str


def _load_env() -> None:
    """Load .env from repo root if python-dotenv is available."""
    if load_dotenv is None:
        return
    here = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(here, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:  # pragma: no cover - defensive
        load_dotenv()


def _build_client() -> BybitClient:
    _load_env()
    api_key = os.getenv("BYBIT_API_KEY") or os.getenv("BYBIT_KEY")
    api_secret = os.getenv("BYBIT_API_SECRET") or os.getenv("BYBIT_SECRET")
    use_testnet = (os.getenv("BYBIT_TESTNET") or "false").strip().lower() == "true"
    if not api_key or not api_secret:
        raise SystemExit("BYBIT_API_KEY / BYBIT_API_SECRET not set (see .env).")
    return BybitClient(api_key=api_key, api_secret=api_secret, testnet=use_testnet)


def _get_open_positions(client: BybitClient):
    account = client.build_account_state(categories=["linear"], settle_coins=["USDT"], symbols=None)
    return [p for p in account.positions if abs(p.size) > 0]


def _build_http_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.3,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _bybit_public_get(
    session: requests.Session,
    endpoint: str,
    *,
    params: Dict[str, object],
    symbol: str,
) -> Dict:
    url = f"{BYBIT_BASE_URL}{endpoint}"
    resp = session.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
    resp.raise_for_status()
    data = resp.json()
    if data.get("retCode") != 0:
        raise RuntimeError(f"{endpoint} error for {symbol}: {data}")
    return data


def _fetch_kline_1m(
    session: requests.Session,
    symbol: str,
    *,
    limit: int = 120,
) -> List[Tuple[int, float]]:
    """Return list of (timestamp_ms, close_price) for the last up-to-`limit` 1m candles."""
    params = {"category": "linear", "symbol": symbol, "interval": "1", "limit": limit}
    data = _bybit_public_get(session, "/v5/market/kline", params=params, symbol=symbol)
    rows = data.get("result", {}).get("list", []) or []
    series: List[Tuple[int, float]] = []
    for row in rows:
        try:
            ts = int(row[0])
            close = float(row[4])
            series.append((ts, close))
        except (ValueError, TypeError, IndexError):
            continue
    series.sort(key=lambda x: x[0])
    return series


def _fetch_oi_series(session: requests.Session, symbol: str) -> List[Tuple[int, float]]:
    """Return list of (timestamp_ms, open_interest) using 5m buckets."""
    params = {"category": "linear", "symbol": symbol, "intervalTime": "5min"}
    data = _bybit_public_get(session, "/v5/market/open-interest", params=params, symbol=symbol)
    rows = data.get("result", {}).get("list", []) or []
    series: List[Tuple[int, float]] = []
    for row in rows:
        try:
            ts = int(row["timestamp"])
            oi = float(row["openInterest"])
            series.append((ts, oi))
        except (KeyError, ValueError, TypeError):
            continue
    series.sort(key=lambda x: x[0])
    return series


def _fetch_trades(session: requests.Session, symbol: str) -> List[TradeRecord]:
    """Return list of recent trade records (max 1000)."""
    params = {"category": "linear", "symbol": symbol, "limit": 1000}
    data = _bybit_public_get(session, "/v5/market/recent-trade", params=params, symbol=symbol)
    rows = data.get("result", {}).get("list", []) or []
    trades: List[TradeRecord] = []
    for row in rows:
        try:
            ts = int(row["time"])
            side = str(row["side"])
            size = float(row["size"])
            trade_key = str(
                row.get("execId")
                or row.get("id")
                or f"{ts}:{side}:{size}:{row.get('price') or row.get('p') or ''}"
            )
            trades.append(TradeRecord(ts_ms=ts, side=side, size=size, key=trade_key))
        except (KeyError, ValueError, TypeError):
            continue
    trades.sort(key=lambda x: x.ts_ms)
    return trades


def _change_from_series(series: List[Tuple[int, float]], minutes: int) -> Tuple[Optional[float], Optional[float]]:
    """Generic helper: return (pct_change, span_min) for a time series."""
    if not series:
        return None, None
    last_ts, last_val = series[-1]
    target = last_ts - minutes * 60 * 1000
    candidates = [item for item in series if item[0] <= target]
    if not candidates:
        return None, None
    past_ts, past_val = candidates[-1]
    if past_val == 0:
        return None, None
    delta = last_val - past_val
    pct = delta / past_val * 100.0
    span_min = (last_ts - past_ts) / 60000.0
    return pct, span_min


def _oi_change(series: List[Tuple[int, float]], minutes: int) -> Tuple[Optional[float], Optional[float]]:
    """OI % change over `minutes`, or (None, None) if unsupported/insufficient."""
    if minutes < 5:
        # Bybit OI history is in 5m+ buckets; skip 1m.
        return None, None
    return _change_from_series(series, minutes)


class TradeHistoryStore:
    """Maintains per-symbol trade history from script start."""

    def __init__(self, *, retention_minutes: int) -> None:
        self.start_ts_ms = int(time.time() * 1000)
        self.retention_ms = max(5, int(retention_minutes)) * 60 * 1000
        self._by_symbol: Dict[str, List[TradeRecord]] = {}
        self._seen_keys: Dict[str, set[str]] = {}

    def refresh_symbol(self, session: requests.Session, symbol: str) -> None:
        fresh = _fetch_trades(session, symbol)
        bucket = self._by_symbol.setdefault(symbol, [])
        seen = self._seen_keys.setdefault(symbol, set())

        appended = False
        for trade in fresh:
            if trade.ts_ms < self.start_ts_ms:
                continue
            if trade.key in seen:
                continue
            seen.add(trade.key)
            bucket.append(trade)
            appended = True

        if appended:
            bucket.sort(key=lambda t: t.ts_ms)
            self._trim_symbol(symbol)

    def _trim_symbol(self, symbol: str) -> None:
        bucket = self._by_symbol.get(symbol) or []
        if not bucket:
            return
        latest_ts = bucket[-1].ts_ms
        cutoff = max(self.start_ts_ms, latest_ts - self.retention_ms)
        if bucket[0].ts_ms >= cutoff:
            return
        trimmed = [t for t in bucket if t.ts_ms >= cutoff]
        self._by_symbol[symbol] = trimmed
        self._seen_keys[symbol] = {t.key for t in trimmed}

    def trades(self, symbol: str) -> List[TradeRecord]:
        return list(self._by_symbol.get(symbol, []))


def _price_change(series: List[Tuple[int, float]], minutes: int) -> Tuple[Optional[float], Optional[float]]:
    """Price % change over `minutes` based on 1m closes."""
    return _change_from_series(series, minutes)


def _cvd_change(
    trades: List[TradeRecord],
    minutes: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """CVD over `minutes` window based on recent-trade history."""
    if not trades:
        return None, None, None
    last_ts = trades[-1].ts_ms
    cutoff = last_ts - minutes * 60 * 1000
    window_trades = [t for t in trades if t.ts_ms >= cutoff]
    if not window_trades:
        return 0.0, 0.0, 0.0
    buy_vol = 0.0
    sell_vol = 0.0
    for trade in window_trades:
        if trade.side == "Buy":
            buy_vol += trade.size
        elif trade.side == "Sell":
            sell_vol += trade.size
    cvd = buy_vol - sell_vol
    gross = buy_vol + sell_vol
    cvd_pct = (cvd / gross * 100.0) if gross > 0 else 0.0
    span_min = (window_trades[-1].ts_ms - window_trades[0].ts_ms) / 60000.0
    return cvd, cvd_pct, span_min


def _fetch_symbol_metrics(
    symbol: str,
    timeframes: Sequence[int],
    trades: List[TradeRecord],
) -> List[TFMetrics]:
    local_session = _build_http_session()
    try:
        kline = _fetch_kline_1m(local_session, symbol, limit=max(120, max(timeframes) + 10))
        oi_series = _fetch_oi_series(local_session, symbol)
        metrics: List[TFMetrics] = []
        for minutes in timeframes:
            price_pct, price_span = _price_change(kline, minutes)
            oi_pct, oi_span = _oi_change(oi_series, minutes)
            cvd, cvd_pct, cvd_span = _cvd_change(trades, minutes)
            metrics.append(
                TFMetrics(
                    symbol=symbol,
                    timeframe_min=minutes,
                    price_pct=price_pct,
                    oi_pct=oi_pct,
                    cvd=cvd,
                    cvd_pct=cvd_pct,
                    price_span_min=price_span,
                    oi_span_min=oi_span,
                    cvd_span_min=cvd_span,
                )
            )
        return metrics
    finally:
        local_session.close()


def build_metrics(
    client: BybitClient,
    session: requests.Session,
    trade_store: TradeHistoryStore,
    *,
    timeframes: Sequence[int],
) -> Tuple[List[TFMetrics], Dict[str, PositionInfo]]:
    positions = _get_open_positions(client)
    by_symbol: Dict[str, PositionInfo] = {}
    for p in positions:
        by_symbol[p.symbol] = PositionInfo(
            side=p.side,
            size=float(p.size),
            notional=float(p.notional),
        )
    symbols = sorted(by_symbol.keys())

    metrics: List[TFMetrics] = []
    if not symbols:
        return metrics, by_symbol

    for symbol in symbols:
        try:
            trade_store.refresh_symbol(session, symbol)
        except Exception:
            # If one refresh fails, keep the existing in-memory history.
            continue

    workers = min(8, max(1, len(symbols)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_symbol = {
            executor.submit(
                _fetch_symbol_metrics,
                symbol,
                list(timeframes),
                trade_store.trades(symbol),
            ): symbol
            for symbol in symbols
        }
        for fut in as_completed(future_to_symbol):
            try:
                metrics.extend(fut.result())
            except Exception:
                # Keep rendering other symbols if one symbol fetch fails.
                continue
    return metrics, by_symbol


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "   n/a"
    return f"{v:+6.2f}%"


def _fmt_cvd(v: Optional[float]) -> str:
    if v is None:
        return "   n/a"
    return f"{v:+8.0f}"


def _fmt_cvd_pct(v: Optional[float]) -> str:
    if v is None:
        return "   n/a"
    return f"{v:+6.2f}%"


def _tf_label(minutes: int) -> str:
    if minutes < 60:
        return f"{minutes}m"
    return f"{minutes // 60}h"


def render_table(
    metrics: Iterable[TFMetrics],
    positions_info: Dict[str, PositionInfo],
    *,
    use_color: bool,
) -> None:
    os.system("cls" if os.name == "nt" else "clear")

    print("Bybit linear USDT multi-TF CVD/OI matrix  [Ctrl+C to exit]\n")
    if not positions_info:
        print("No open positions found.")
        return

    RED = "\x1b[91m"
    GREEN = "\x1b[92m"
    DIM = "\x1b[2m"
    RESET = "\x1b[0m"

    header = (
        f"{'SYMBOL':<10} {'SIDE':<4} {'SIZE':>10} {'NOTION':>10} "
        f"{'TF':<4} {'PRICEΔ':>7} {'OIΔ%':>7} {'CVD':>9} {'CVDΔ%':>7}"
    )
    print(header)
    print("-" * len(header))

    sorted_metrics = sorted(metrics, key=lambda m: (m.symbol, m.timeframe_min))
    for m in sorted_metrics:
        pos_info = positions_info.get(m.symbol)
        if not pos_info:
            continue
        side = pos_info.side
        size = pos_info.size
        notional = pos_info.notional
        price_str = _fmt_pct(m.price_pct)
        oi_str = _fmt_pct(m.oi_pct)
        cvd_str = _fmt_cvd(m.cvd)
        cvd_pct_str = _fmt_cvd_pct(m.cvd_pct)

        tf_label = _tf_label(m.timeframe_min)
        line = (
            f"{m.symbol:<10} {side[:4]:<4} "
            f"{size:>10.2f} {notional:>10.2f} "
            f"{tf_label:<4} {price_str:>7} {oi_str:>7} {cvd_str:>9} {cvd_pct_str:>7}"
        )
        color = ""
        if m.oi_pct is not None and m.price_pct is not None and m.cvd is not None:
            if m.oi_pct > 0 and m.price_pct < 0 and m.cvd < 0:
                color = GREEN
            elif m.oi_pct > 0 and m.price_pct > 0 and m.cvd > 0:
                color = RED

        if use_color and color:
            print(f"{color}{line}{RESET}")
        else:
            print(line)

    note = (
        "OI uses 5m buckets. CVDΔ% = (buy volume - sell volume) / "
        "(buy volume + sell volume) * 100 over each rolling timeframe window. "
        "CVD is built from script-start trade history."
    )
    row_signal = "Row color: green = OI↑ + price↓ + CVD<0; red = OI↑ + price↑ + CVD>0."
    if use_color:
        note = f"{DIM}{note}{RESET}"
        row_signal = f"{GREEN}green{RESET} = OI↑ + price↓ + CVD<0; {RED}red{RESET} = OI↑ + price↑ + CVD>0."

    print(f"\nLegend: {row_signal} {note}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-timeframe CVD/OI view for current Bybit linear USDT positions.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=30.0,
        help="Refresh interval in seconds (default: 30).",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    parser.add_argument(
        "--retention-min",
        type=int,
        default=180,
        help="Trade-history retention in minutes (default: 180).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    client = _build_client()
    session = _build_http_session()
    trade_store = TradeHistoryStore(retention_minutes=max(args.retention_min, max(TIMEFRAMES_MIN)))
    interval = max(5.0, float(args.interval))
    try:
        while True:
            try:
                metrics, positions = build_metrics(
                    client,
                    session,
                    trade_store,
                    timeframes=TIMEFRAMES_MIN,
                )
                render_table(
                    metrics,
                    positions,
                    use_color=not args.no_color,
                )
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                os.system("cls" if os.name == "nt" else "clear")
                print(f"[cvd_oi_matrix] error: {exc}")
            time.sleep(interval)
    finally:
        session.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass

