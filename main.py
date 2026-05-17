import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
from data_loader import load_historical_data
import market_state

app = FastAPI(title="Trader Bot API - Stage 2")

class PriceInput(BaseModel):
    price: float
    volume: int

@app.on_event("startup")
async def startup_event():
    """Inicializa os dados históricos reais ao iniciar a API."""
    try:
        print("Iniciando carregamento de dados históricos...")
        df_hist = load_historical_data()
        market_state.set_historical_data(df_hist)
        print(f"Setup concluído. Total de candles: {len(market_state.history_data)}")
        
        # Carregar 4H
        from binance.client import Client
        import os
        api_key = os.getenv("BINANCE_API_KEY")
        secret = os.getenv("BINANCE_SECRET_KEY")
        if api_key and secret:
            client = Client(api_key, secret, testnet=True)
            print("Carregando histórico 4H da Binance...")
            klines_4h = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_4HOUR, "30 days ago UTC")
            df_4h = pd.DataFrame(klines_4h, columns=['t', 'o', 'h', 'l', 'c', 'v', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
            df_4h['date'] = pd.to_datetime(df_4h['t'], unit='ms')
            df_4h.set_index('date', inplace=True)
            for col in ['o', 'h', 'l', 'c', 'v']:
                df_4h[col] = df_4h[col].astype(float)
            df_4h = df_4h[['o', 'h', 'l', 'c', 'v']]
            df_4h.rename(columns={'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}, inplace=True)
            market_state.set_historical_4h_data(df_4h)
            print(f"Histórico 4H carregado: {len(df_4h)} candles.")
        
        import asyncio
        from binance_stream import start_stream
        asyncio.create_task(start_stream())
        
    except Exception as e:
        print(f"Erro fatal ao carregar dados: {e}")
        pass

@app.post("/analisar_mercado")
async def analisar_mercado(input_data: PriceInput):
    """
    Recebe um novo preço, cria um candle temporário, adiciona ao histórico real e executa a análise.
    """
    if market_state.history_data.empty:
        raise HTTPException(status_code=503, detail="Dados históricos não carregados.")

    new_candle = {
        "date": datetime.now(),
        "open": input_data.price,
        "high": input_data.price,
        "low": input_data.price,
        "close": input_data.price,
        "volume": input_data.volume
    }
    
    try:
        resultado = market_state.append_new_candle(new_candle)
        return resultado
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")

@app.get("/status")
def status():
    return {
        "status": "ok", 
        "candles_count": len(market_state.history_data),
        "last_candle_date": str(market_state.history_data.index[-1]) if not market_state.history_data.empty else None
    }
