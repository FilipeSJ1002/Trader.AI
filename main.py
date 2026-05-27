import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import market_state
import database

load_dotenv()

app = FastAPI(
    title="Trader.AI API — V4",
    description="Motor de scalping multi-asset com ML, MTF e log de performance.",
    version="4.0.0"
)


class PriceInput(BaseModel):
    symbol: str
    price: float
    volume: int


@app.on_event("startup")
async def startup_event():
    """Inicializa os dados históricos reais ao iniciar a API (Multi-Asset)."""
    try:
        from binance.client import Client
        api_key = os.getenv("BINANCE_API_KEY")
        secret  = os.getenv("BINANCE_SECRET_KEY")

        if not api_key or not secret:
            print("Chaves API Binance ausentes. Não foi possível baixar histórico inicial.")
            return

        client = Client(api_key, secret, testnet=True)
        ativos = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]

        for symbol in ativos:
            print(f"Carregando histórico 1M para {symbol} da Binance...")
            klines_1m = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, "7 days ago UTC")
            df_1m = pd.DataFrame(klines_1m, columns=[
                't', 'o', 'h', 'l', 'c', 'v',
                'close_time', 'qav', 'num_trades',
                'taker_base_vol', 'taker_quote_vol', 'ignore'
            ])
            df_1m['date'] = pd.to_datetime(df_1m['t'], unit='ms')
            df_1m.set_index('date', inplace=True)
            for col in ['o', 'h', 'l', 'c', 'v']:
                df_1m[col] = df_1m[col].astype(float)
            df_1m = df_1m[['o', 'h', 'l', 'c', 'v']]
            df_1m.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
            market_state.set_historical_data(symbol, df_1m)
            print(f"Histórico 1M carregado para {symbol}: {len(df_1m)} candles.")

            print(f"Carregando histórico 1H para {symbol} da Binance...")
            klines_1h = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "30 days ago UTC")
            df_1h = pd.DataFrame(klines_1h, columns=[
                't', 'o', 'h', 'l', 'c', 'v',
                'close_time', 'qav', 'num_trades',
                'taker_base_vol', 'taker_quote_vol', 'ignore'
            ])
            df_1h['date'] = pd.to_datetime(df_1h['t'], unit='ms')
            df_1h.set_index('date', inplace=True)
            for col in ['o', 'h', 'l', 'c', 'v']:
                df_1h[col] = df_1h[col].astype(float)
            df_1h = df_1h[['o', 'h', 'l', 'c', 'v']]
            df_1h.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
            market_state.set_historical_1h_data(symbol, df_1h)
            print(f"Histórico 1H carregado para {symbol}: {len(df_1h)} candles.")

        import asyncio
        from binance_stream import start_stream
        asyncio.create_task(start_stream())

    except Exception as e:
        print(f"Erro fatal ao carregar dados: {e}")
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/analisar_mercado")
async def analisar_mercado(input_data: PriceInput):
    """
    Recebe um novo preço, cria um candle temporário,
    adiciona ao histórico real e executa a análise técnica + ML.
    """
    symbol = input_data.symbol.upper()

    if symbol not in market_state.history_data or market_state.history_data[symbol].empty:
        raise HTTPException(status_code=503, detail=f"Dados históricos não carregados para {symbol}.")

    new_candle = {
        "date":   datetime.now(),
        "open":   input_data.price,
        "high":   input_data.price,
        "low":    input_data.price,
        "close":  input_data.price,
        "volume": input_data.volume
    }

    try:
        resultado = market_state.append_new_candle(symbol, new_candle)
        return resultado
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")


@app.get("/status")
def status():
    """Retorna o status do bot e os símbolos monitorados."""
    return {
        "status":          "ok",
        "version":         "4.0.0",
        "symbols_tracked": list(market_state.history_data.keys())
    }


@app.get("/performance")
def performance():
    """
    Retorna o resumo de performance do bot:
    win rate, P&L total, P&L médio por trade, melhor/pior trade,
    e P&L acumulado dos últimos 7 dias.
    """
    return database.get_performance_summary()


@app.get("/trades")
def recent_trades(limit: int = 20):
    """
    Retorna os N trades mais recentes do log.
    Parâmetro: ?limit=20 (padrão)
    """
    if limit > 200:
        limit = 200
    trades = database.get_recent_trades(limit)
    return {"trades": trades, "count": len(trades)}


@app.get("/positions")
def positions():
    """Retorna o estado atual de todas as posições abertas."""
    ativos = ["BTCUSDT", "ETHUSDT", "XRPUSDT"]
    result = {}
    for symbol in ativos:
        pos = database.get_position(symbol)
        if pos["qty"] > 0:
            result[symbol] = pos
    return {"open_positions": result, "count": len(result)}
