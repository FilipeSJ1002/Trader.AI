import pandas as pd
from strategy import TechnicalAnalysis
from fastapi import HTTPException

# Global state
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

    # Converter para DataFrame
    new_df = pd.DataFrame([candle])
    new_df.set_index('date', inplace=True)
    
    # Concatenar com histórico. Utilizando copy p/ evitar Warning de concatenação
    history_data = pd.concat([history_data, new_df])
    
    # Manter tamanho do histórico gerenciável
    if len(history_data) > 10005: # pequena margem para evitar operação expensiva a cada tick
        history_data = history_data.tail(10000)
        
    # Executar estratégia (usando cópia para evitar Warning de SettingWithCopy)
    resultado = strategy.analisar_compra_venda(history_data.copy())
    return resultado
