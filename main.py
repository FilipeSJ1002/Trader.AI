from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
from data_loader import load_historical_data
from strategy import TechnicalAnalysis

app = FastAPI(title="Trader Bot API - Stage 2")

# Global variables to hold state
history_data = pd.DataFrame()
strategy = TechnicalAnalysis()

class PriceInput(BaseModel):
    price: float
    volume: int

@app.on_event("startup")
async def startup_event():
    """Inicializa os dados históricos reais ao iniciar a API."""
    global history_data
    try:
        print("Iniciando carregamento de dados históricos...")
        history_data = load_historical_data()
        print(f"Setup concluído. Total de candles: {len(history_data)}")
    except Exception as e:
        print(f"Erro fatal ao carregar dados: {e}")
        # Em produção, poderíamos impedir o startup, mas aqui apenas logamos
        pass

@app.post("/analisar_mercado")
async def analisar_mercado(input_data: PriceInput):
    """
    Recebe um novo preço, cria um candle temporário, adiciona ao histórico real e executa a análise.
    """
    global history_data
    
    if history_data.empty:
        raise HTTPException(status_code=503, detail="Dados históricos não carregados.")

    # Timestamp atual para o novo candle
    new_timestamp = datetime.now()
    
    # Criar novo candle com base no input
    # Assumindo que o input representa o fechamento do candle atual
    # Para simulação, OHLC são iguais ao preço atual
    new_candle = {
        "open": input_data.price,
        "high": input_data.price,
        "low": input_data.price,
        "close": input_data.price,
        "volume": input_data.volume,
        "date": new_timestamp # Necessário para o set_index temporário, se usarmos reset_index ou similar
    }
    
    # Converter para DataFrame
    new_df = pd.DataFrame([new_candle])
    new_df.set_index('date', inplace=True)
    
    # Concatenar com histórico para análise (cópia para não mutar o histórico base infinitamente se for apenas simulação de tick)
    # NOTA: Se o objetivo for acumular candles reais ao longo do tempo, deveríamos atualizar 'history_data'.
    # O prompt diz: "Adicione esse candle ao final do histórico real carregado."
    # Assumirei que é para atualizar o estado global, simulando a passagem do tempo.
    
    history_data = pd.concat([history_data, new_df])
    
    # Manter tamanho do histórico gerenciável (10.000 como definido no loader)
    if len(history_data) > 10005: # pequena margem
        history_data = history_data.tail(10000)
        
    # Executar estratégia
    try:
        # Passamos uma cópia para garantir que a estratégia não modifique o dados globais inadvertidamente
        resultado = strategy.analisar_compra_venda(history_data.copy())
        return resultado
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")

@app.get("/status")
def status():
    return {
        "status": "ok", 
        "candles_count": len(history_data),
        "last_candle_date": str(history_data.index[-1]) if not history_data.empty else None
    }
