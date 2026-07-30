import os
import asyncio
import requests
import pandas as pd
from datetime import datetime
from typing import cast
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import market_state
import database
from binance_stream import ATIVOS

load_dotenv()

app = FastAPI(
    title="Trader.AI API — V8",
    description="Motor de trend-following BTC+ Alpha (SMA200 + ATH-25% + Momentum Relativo).",
    version="8.0.0"
)


class PriceInput(BaseModel):
    symbol: str
    price: float
    volume: int


def _carregar_diario(symbol: str) -> pd.Series:
    """
    Carrega closes diarios reais da Binance publica (250 dias).
    Fallback: parquet local se a API falhar.
    """
    try:
        resp = requests.get(
            "https://api.binance.com/api/v3/klines",
            params={"symbol": symbol, "interval": "1d", "limit": 250},
            timeout=10
        )
        resp.raise_for_status()
        cols = ['t', 'o', 'h', 'l', 'c', 'v',
                'close_time', 'qav', 'num_trades',
                'taker_base_vol', 'taker_quote_vol', 'ignore']
        df = pd.DataFrame(resp.json(), columns=cols)
        df['date'] = pd.to_datetime(df['t'], unit='ms')
        df.set_index('date', inplace=True)
        series = cast(pd.Series, df['c'].astype(float))
        print(f"  Diario {symbol}: {len(series)} dias (Binance real)")
        return series
    except Exception as e:
        print(f"  API real falhou para {symbol} ({e}), usando parquet...")
        path = os.path.join("data", f"{symbol}_1m.parquet")
        df_pq = pd.read_parquet(path)
        if 'date' in df_pq.columns:
            df_pq.set_index('date', inplace=True)
        series = df_pq['close'].resample('1D').last().dropna()
        print(f"  Diario {symbol}: {len(series)} dias (parquet local)")
        return series


def _carregar_todos_historicos():
    """
    Carrega todos os historicos de forma sincrona.
    Executado em thread separada para nao bloquear o event loop.
    """
    from binance.client import Client

    api_key = os.getenv("BINANCE_API_KEY")
    secret  = os.getenv("BINANCE_SECRET_KEY")

    if not api_key or not secret:
        print("[STARTUP] Chaves API ausentes — sem historico inicial.")
        return

    client     = Client(api_key, secret, testnet=True)
    cols_kline = ['t', 'o', 'h', 'l', 'c', 'v',
                  'close_time', 'qav', 'num_trades',
                  'taker_base_vol', 'taker_quote_vol', 'ignore']

    for symbol in ATIVOS:
        print(f"\n[STARTUP] Carregando {symbol}...")

        # 1M — 15 dias
        klines = client.get_historical_klines(
            symbol, Client.KLINE_INTERVAL_1MINUTE, "15 days ago UTC")
        df_1m = pd.DataFrame(klines, columns=cols_kline)
        df_1m['date'] = pd.to_datetime(df_1m['t'], unit='ms')
        df_1m.set_index('date', inplace=True)
        df_1m = df_1m[['o', 'h', 'l', 'c', 'v']].astype(float)
        df_1m.columns = ['open', 'high', 'low', 'close', 'volume']
        market_state.set_historical_data(symbol, df_1m)
        print(f"  1M: {len(df_1m)} candles")

        # 1H — 30 dias
        klines_h = client.get_historical_klines(
            symbol, Client.KLINE_INTERVAL_1HOUR, "30 days ago UTC")
        df_1h = pd.DataFrame(klines_h, columns=cols_kline)
        df_1h['date'] = pd.to_datetime(df_1h['t'], unit='ms')
        df_1h.set_index('date', inplace=True)
        df_1h = df_1h[['o', 'h', 'l', 'c', 'v']].astype(float)
        df_1h.columns = ['open', 'high', 'low', 'close', 'volume']
        market_state.set_historical_1h_data(symbol, df_1h)
        print(f"  1H: {len(df_1h)} candles")

        # Diario — 250 dias da Binance real (necessario para SMA200)
        daily = _carregar_diario(symbol)
        market_state.set_historical_daily(symbol, daily)

    print("\n[STARTUP] Historicos carregados com sucesso.")


@app.on_event("startup")
async def startup_event():
    """Inicia carregamento de dados em background e sobe o WebSocket."""
    from binance_stream import start_stream
    # Roda as chamadas bloqueantes em thread separada — nao trava o event loop
    await asyncio.to_thread(_carregar_todos_historicos)
    asyncio.create_task(start_stream())


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    return {
        "status":          "ok",
        "version":         "8.0.0",
        "symbols_tracked": list(market_state.history_data.keys())
    }


@app.get("/trend")
def trend():
    """Estado atual da estrategia V8: pesos-alvo, SMA200, momentum relativo."""
    from trend_strategy import describe_signal
    daily = market_state.get_daily_closes()
    if not daily:
        return {"status": "aguardando dados diarios", "assets": {}}
    return describe_signal(daily)


@app.get("/positions")
def positions():
    result = {}
    for symbol in ATIVOS:
        pos = database.get_position(symbol)
        if pos["qty"] > 0:
            result[symbol] = pos
    return {"open_positions": result, "count": len(result)}


@app.get("/performance")
def performance():
    summary = database.get_performance_summary()
    summary["daily_drawdown"] = database.get_drawdown_summary()
    return summary


@app.get("/trades")
def recent_trades(limit: int = 20):
    if limit > 200:
        limit = 200
    trades = database.get_recent_trades(limit)
    return {"trades": trades, "count": len(trades)}


@app.get("/regime")
def regime():
    return {
        "regimes": {
            symbol: {
                "regime":       market_state.market_regime.get(symbol, "UNKNOWN"),
                "signal_5m":    market_state._get_5m_signal(symbol),
                "daily_return": round(market_state.daily_returns.get(symbol, 0) * 100, 3)
            }
            for symbol in ATIVOS
            if symbol in market_state.history_data
        }
    }


@app.get("/learning")
def learning():
    from learn_from_trades import stats
    return stats()


_retrain_running = False


@app.post("/retrain")
async def retrain(background_tasks: BackgroundTasks,
                  pair: str | None = None,
                  skip_download: bool = False):
    global _retrain_running
    if _retrain_running:
        raise HTTPException(status_code=409, detail="Retreino ja em andamento.")
    pairs = [pair.upper()] if pair else None

    def _run():
        global _retrain_running
        _retrain_running = True
        try:
            from retrain_scheduler import run_retrain
            run_retrain(pairs=pairs or [], skip_download=skip_download)
        finally:
            _retrain_running = False

    background_tasks.add_task(_run)
    return {"status": "started", "pairs": pairs or "todos"}


@app.get("/retrain/status")
def retrain_status():
    from retrain_scheduler import get_retrain_log
    return {"running": _retrain_running, "recent_runs": get_retrain_log(limit=5)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
