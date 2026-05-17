import pandas as pd
import pandas_ta as ta
from datetime import datetime
class TechnicalAnalysis:
    def __init__(self):
        pass

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos ao DataFrame.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
        
        df.ta.rsi(length=14, append=True)
        
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        df.ta.bbands(length=20, std=2, append=True)
        
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        
        # ATR para volatilidade
        df.ta.atr(length=14, append=True)
        
        return df

    def analisar_compra_venda(self, df: pd.DataFrame) -> dict:
        """
        Executa a lógica de confluência para decisão de trading.
        Retorna um dicionário com a decisão e análise técnica detalhada.
        """
        df = self.add_indicators(df)
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        
        cols = df.columns.tolist()
        try:
            bb_lower_col = [c for c in cols if c.startswith('BBL_20_2')][0]
            bb_upper_col = [c for c in cols if c.startswith('BBU_20_2')][0]
            macd_hist_col = [c for c in cols if c.startswith('MACDh_12_26_9')][0]
            rsi_col = [c for c in cols if c.startswith('RSI_14')][0]
            ema_200_col = [c for c in cols if c.startswith('EMA_200')][0]
            atr_col = [c for c in cols if c.startswith('ATRr_14')][0]
        except IndexError:
            raise ValueError(f"Indicadores não encontrados. Colunas disponíveis: {cols}")

        rsi = last_candle[rsi_col]
        close_price = last_candle['close']
        atr_value = last_candle[atr_col]
        
        bb_lower = last_candle[bb_lower_col]
        bb_upper = last_candle[bb_upper_col]
        
        macd_hist = last_candle[macd_hist_col]
        prev_macd_hist = prev_candle[macd_hist_col]
        
        ema_200 = last_candle[ema_200_col]
        
        decision = "NEUTRO"
        
        buy_score = 0
        sell_score = 0
        
        if rsi < 30:
            buy_score += 40
        elif rsi < 40:
            buy_score += 20
            
        if rsi > 75:
            sell_score += 40
        elif rsi > 65:
            sell_score += 20
            
        if close_price <= bb_lower:
            buy_score += 35
            
        if close_price >= bb_upper:
            sell_score += 35
            
        if macd_hist > prev_macd_hist:
            buy_score += 25
        elif macd_hist < prev_macd_hist:
            sell_score += 25
            
        if buy_score >= 85:
            decision = "COMPRA_FORTE"
        elif buy_score >= 60:
            decision = "COMPRA_MODERADA"
        elif buy_score >= 45:
            decision = "COMPRA_LEVE"
        elif sell_score >= 60:
            decision = "VENDA_FORTE"
            
        trend = "BULLISH (Price > EMA200)" if close_price > ema_200 else "BEARISH (Price < EMA200)"
        
        bollinger_position = "INSIDE_BANDS"
        if close_price > bb_upper:
            bollinger_position = "UPPER_BAND_BREAKOUT"
        elif close_price < bb_lower:
            bollinger_position = "LOWER_BAND_BREAKOUT"
            
        macd_signal = "BULLISH" if macd_hist > 0 else "BEARISH"
        
        return {
            "decision": decision,
            "analysis": {
                "rsi": round(rsi, 2),
                "buy_score": buy_score,
                "sell_score": sell_score,
                "bollinger_position": bollinger_position,
                "macd_signal": macd_signal,
                "macd_hist_change": "INCREASING" if macd_hist > prev_macd_hist else "DECREASING",
                "trend": trend,
                "close_price": close_price,
                "ema_200": round(ema_200, 2),
                "atr": round(atr_value, 2)
            },
            "timestamp": str(last_candle.name) if hasattr(last_candle, 'name') else str(datetime.now())
        }
