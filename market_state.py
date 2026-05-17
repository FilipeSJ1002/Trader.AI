import pandas as pd
from strategy import TechnicalAnalysis
from fastapi import HTTPException

history_data = pd.DataFrame()
history_4h = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
htf_weakness = False
strategy = TechnicalAnalysis()

def set_historical_data(df: pd.DataFrame):
    """Inicializa o histórico de dados na memória."""
    global history_data
    history_data = df

def set_historical_4h_data(df: pd.DataFrame):
    """Inicializa o histórico 4H na memória."""
    global history_4h
    history_4h = df

import pandas_ta as ta
def update_4h_candle(candle: dict):
    """
    Recebe um candle de 4H do WebSocket, insere no estado e recalcula O(1) a fraqueza.
    """
    global history_4h, htf_weakness
    
    candle_date = candle['date']
    history_4h.loc[candle_date] = [
        candle['open'], candle['high'], candle['low'], candle['close'], candle['volume']
    ]
    
    if len(history_4h) > 200:
        history_4h = history_4h.tail(200)
        
    if len(history_4h) < 30:
        htf_weakness = False
        return
        
    df_temp = history_4h.copy()
    df_temp.ta.rsi(length=14, append=True)
    df_temp.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    last_4h = df_temp.iloc[-1]
    prev_4h = df_temp.iloc[-2]
    
    cols = df_temp.columns
    try:
        macd_hist_col = [c for c in cols if c.startswith('MACDh_')][0]
        rsi_col = [c for c in cols if c.startswith('RSI_')][0]
    except IndexError:
        return
        
    macd_hist = last_4h[macd_hist_col]
    prev_macd_hist = prev_4h[macd_hist_col]
    rsi = last_4h[rsi_col]
    
    weakness = False
    if rsi < 45:
        weakness = True
    elif macd_hist < 0 and macd_hist < prev_macd_hist:
        weakness = True
    elif rsi > 70 and macd_hist < prev_macd_hist:
        weakness = True
        
    htf_weakness = weakness

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
    
    if htf_weakness and resultado.get("decision", "").startswith("COMPRA_"):
        resultado["decision"] = "BLOQUEADO_POR_HTF"
        
    if "analysis" in resultado:
        resultado["analysis"]["htf_weakness"] = htf_weakness
        
    return resultado
