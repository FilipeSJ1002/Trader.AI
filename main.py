import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
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
        
        # Iniciar a conexão WebSocket em background para não bloquear o event loop
        import asyncio
        from binance_stream import start_stream
        asyncio.create_task(start_stream())
        
    except Exception as e:
        print(f"Erro fatal ao carregar dados: {e}")
        # Em produção, poderíamos impedir o startup, mas aqui apenas logamos
        pass

@app.post("/analisar_mercado")
async def analisar_mercado(input_data: PriceInput):
    """
    Recebe um novo preço, cria um candle temporário, adiciona ao histórico real e executa a análise.
    """
    if market_state.history_data.empty:
        raise HTTPException(status_code=503, detail="Dados históricos não carregados.")

    # Criar novo candle simulando a mesma abertura/max/min do close
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
