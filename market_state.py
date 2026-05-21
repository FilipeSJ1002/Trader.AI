import pandas as pd
from strategy import TechnicalAnalysis
from fastapi import HTTPException
import pandas_ta as ta

# Estados por símbolo
history_data = {}
history_1h = {}
htf_weakness = {}
daily_returns = {}
strategy = TechnicalAnalysis()

def set_historical_data(symbol: str, df: pd.DataFrame):
    """Inicializa o histórico de dados na memória para um símbolo."""
    global history_data
    history_data[symbol] = df
    if symbol not in htf_weakness:
        htf_weakness[symbol] = False

def set_historical_1h_data(symbol: str, df: pd.DataFrame):
    """Inicializa o histórico 1H na memória para um símbolo."""
    global history_1h
    history_1h[symbol] = df

def update_1h_candle(symbol: str, candle: dict):
    """
    Recebe um candle de 1H do WebSocket para o símbolo correspondente, 
    insere no estado e recalcula O(1) a fraqueza (HTF).
    """
    global history_1h, htf_weakness
    
    if symbol not in history_1h:
        history_1h[symbol] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        
    candle_date = candle['date']
    history_1h[symbol].loc[candle_date] = [
        candle['open'], candle['high'], candle['low'], candle['close'], candle['volume']
    ]
    
    if len(history_1h[symbol]) > 200:
        history_1h[symbol] = history_1h[symbol].tail(200)
        
    # Calcular Força Relativa (Daily Return)
    if len(history_1h[symbol]) > 0:
        lookback = 24 if len(history_1h[symbol]) >= 24 else len(history_1h[symbol])
        primeiro_1h = history_1h[symbol].iloc[-lookback]
        ultimo_1h = history_1h[symbol].iloc[-1]
        daily_returns[symbol] = (ultimo_1h['close'] - primeiro_1h['close']) / primeiro_1h['close']
        
    if len(history_1h[symbol]) < 30:
        htf_weakness[symbol] = False
        return
        
    df_temp = history_1h[symbol].copy()
    df_temp.ta.rsi(length=14, append=True)
    df_temp.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    last_1h = df_temp.iloc[-1]
    prev_1h = df_temp.iloc[-2]
    
    cols = df_temp.columns
    try:
        macd_hist_col = [c for c in cols if c.startswith('MACDh_')][0]
        rsi_col = [c for c in cols if c.startswith('RSI_')][0]
    except IndexError:
        return
        
    macd_hist = last_1h[macd_hist_col]
    prev_macd_hist = prev_1h[macd_hist_col]
    rsi = last_1h[rsi_col]
    
    weakness = False
    if rsi < 45:
        weakness = True
    elif macd_hist < 0 and macd_hist < prev_macd_hist:
        weakness = True
    elif rsi > 70 and macd_hist < prev_macd_hist:
        weakness = True
        
    htf_weakness[symbol] = weakness

def append_new_candle(symbol: str, candle: dict) -> dict:
    """
    Recebe um novo candle, adiciona ao histórico real do símbolo e executa a análise.
    Mantém a memória limitada aos últimos 10.000 candles por símbolo.
    """
    global history_data
    
    if symbol not in history_data or history_data[symbol].empty:
        raise Exception(f"Dados históricos não carregados para {symbol}.")

    new_df = pd.DataFrame([candle])
    new_df.set_index('date', inplace=True)
    
    history_data[symbol] = pd.concat([history_data[symbol], new_df])
    history_data[symbol] = history_data[symbol][~history_data[symbol].index.duplicated(keep='last')]
    
    if len(history_data[symbol]) > 10005:
        history_data[symbol] = history_data[symbol].tail(10000)
        
    resultado = strategy.analisar_compra_venda(history_data[symbol].copy())
    
    weakness = htf_weakness.get(symbol, False)
    if weakness and resultado.get("decision", "").startswith("COMPRA_"):
        resultado["decision"] = "BLOQUEADO_POR_HTF"
        
    # Relative Strength Bônus e Punição
    if daily_returns:
        max_symbol = max(daily_returns, key=lambda k: daily_returns[k])
        if symbol == max_symbol and daily_returns[symbol] > 0:
            resultado["analysis"]["asset_strength"] = "STRONG"
            if resultado.get("decision", "") == "COMPRA_MODERADA":
                resultado["decision"] = "COMPRA_FORTE"  # Foca no cavalo mais rápido
        elif daily_returns.get(symbol, 0) < 0 and symbol != max_symbol:
            resultado["analysis"]["asset_strength"] = "WEAK"
            if resultado.get("decision", "").startswith("COMPRA_"):
                resultado["decision"] = "BLOQUEADA_POR_FORÇA_RELATIVA_FRACA" # Não opera a moeda fraca
        else:
            resultado["analysis"]["asset_strength"] = "NEUTRAL"
        
    if "analysis" in resultado:
        resultado["analysis"]["htf_weakness"] = weakness
        
    return resultado
