from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from datetime import datetime
from mock_data import generate_mock_data
from strategy import TechnicalAnalysis

app = FastAPI(title="Trader Bot API - Stage 1")

# Global variables to hold state
mock_history = pd.DataFrame()
strategy = TechnicalAnalysis()

class PriceInput(BaseModel):
    price: float
    volume: int

@app.on_event("startup")
async def startup_event():
    """Inicializa os dados mockados ao iniciar a API."""
    global mock_history
    print("Gerando dados históricos mockados...")
    mock_history = generate_mock_data(n_candles=300)
    print(f"Dados gerados: {len(mock_history)} candles.")

@app.post("/analisar_mercado")
async def analisar_mercado(input_data: PriceInput):
    """
    Recebe um novo preço, adiciona ao histórico e executa a análise técnica.
    """
    global mock_history
    
    # Criar novo candle
    # Para simplificar na Etapa 1, assumimos que o input é um candle fechado ou update
    # Aqui vamos simular que é um novo candle completo baseado no preço atual
    # Para ser mais realista, o 'input_data' do usuário deveria ser um candle OHLC completo
    # Mas conforme o prompt, recebemos apenas {"price": ..., "volume": ...}
    
    # Vamos criar um candle fictício apenas com o preço de fechamento igual ao input
    # e OHLC próximos para não quebrar a lógica de candles
    # Timestamp atual
    new_timestamp = datetime.now()
    
    new_candle = {
        "timestamp": new_timestamp,
        "open": input_data.price,
        "high": input_data.price,
        "low": input_data.price,
        "close": input_data.price,
        "volume": input_data.volume
    }
    
    # Converter para DataFrame e concatenar
    new_df = pd.DataFrame([new_candle])
    new_df.set_index('timestamp', inplace=True)
    
    # Adicionar ao histórico
    mock_history = pd.concat([mock_history, new_df])
    
    # Manter tamanho do histórico gerenciável (opcional, mas bom para performance)
    if len(mock_history) > 500:
        mock_history = mock_history.iloc[-500:]
        
    # Executar estratégia
    try:
        resultado = strategy.analisar_compra_venda(mock_history.copy())
        return resultado
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Erro na análise: {str(e)}")

@app.get("/status")
def status():
    return {"status": "ok", "candles_count": len(mock_history)}
