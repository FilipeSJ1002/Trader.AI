import pandas as pd
from strategy import TechnicalAnalysis
from fastapi import HTTPException

history_data = pd.DataFrame()
strategy = TechnicalAnalysis()

def set_historical_data(df: pd.DataFrame):
    """Inicializa o histórico de dados na memória."""
    global history_data
    history_data = df

def append_new_candle(candle: dict) -> dict:
    """
    Recebe um novo candle, adiciona ao histórico real e executa a análise.
     Mantém a memória limitada aos últimos 10.000 candles.
    
    candle = {
        'date': datetime,
        'open': float,
        'high': float,
        'low': float,
        'close': float,
        'volume': float
    }
    """
    global history_data
    
    if history_data.empty:
        raise Exception("Dados históricos não carregados.")

    new_df = pd.DataFrame([candle])
    new_df.set_index('date', inplace=True)
    
    history_data = pd.concat([history_data, new_df])
    
    if len(history_data) > 10005:
        history_data = history_data.tail(10000)
        
    resultado = strategy.analisar_compra_venda(history_data.copy())
    return resultado
