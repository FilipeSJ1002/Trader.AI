import pandas as pd
import pandas_ta as ta

class TechnicalAnalysis:
    def __init__(self):
        pass

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adiciona indicadores técnicos ao DataFrame.
        """
        # Garantir que o index é datetime se não for
        if not isinstance(df.index, pd.DatetimeIndex):
            # Se tiver coluna timestamp
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
        
        # RSI 14
        df.ta.rsi(length=14, append=True)
        
        # MACD (12, 26, 9)
        # pandas-ta retorna colunas make MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # Bollinger Bands (20, 2)
        # Retorna BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        df.ta.bbands(length=20, std=2, append=True)
        
        # EMA 50 e 200
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        
        return df

    def analisar_compra_venda(self, df: pd.DataFrame) -> dict:
        """
        Executa a lógica de confluência para decisão de trading.
        Retorna um dicionário com a decisão e análise técnica detalhada.
        """
        # Calcular indicadores
        df = self.add_indicators(df)
        
        # Pegar o último candle (o mais recente)
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2] # Necessário para verificar evolução do MACD Hist
        
        # Nomes das colunas geradas pelo pandas-ta (podem variar, verificar pta documentation ou usar nomes padrão)
        # RSI_14
        # MACDh_12_26_9 (Histograma)
        # BBL_20_2.0 (Lower Band)
        # BBU_20_2.0 (Upper Band)
        # EMA_50
        # EMA_200
        
        # Tentar recuperar colunas de BB
        # Padrão: BBL_20_2.0 ou BBL_20_2
        # Vamos procurar colunas que começam com BBL
        cols = df.columns.tolist()
        try:
            bb_lower_col = [c for c in cols if c.startswith('BBL_20_2')][0]
            bb_upper_col = [c for c in cols if c.startswith('BBU_20_2')][0]
            macd_hist_col = [c for c in cols if c.startswith('MACDh_12_26_9')][0]
            rsi_col = [c for c in cols if c.startswith('RSI_14')][0]
            ema_200_col = [c for c in cols if c.startswith('EMA_200')][0]
        except IndexError:
            # Fallback debug ou erro
            raise ValueError(f"Indicadores não encontrados. Colunas disponíveis: {cols}")

        rsi = last_candle[rsi_col]
        close_price = last_candle['close']
        
        bb_lower = last_candle[bb_lower_col]
        bb_upper = last_candle[bb_upper_col]
        
        macd_hist = last_candle[macd_hist_col]
        prev_macd_hist = prev_candle[macd_hist_col]
        
        ema_200 = last_candle[ema_200_col]
        
        decision = "NEUTRO"
        
        # Lógica de CONFLUÊNCIA (Sistema de Scoring)
        buy_score = 0
        sell_score = 0
        
        # 1. Avaliação do RSI
        if rsi < 30:
            buy_score += 40
        elif rsi < 40:
            buy_score += 20
            
        if rsi > 75:
            sell_score += 40
        elif rsi > 65:
            sell_score += 20
            
        # 2. Avaliação de Bollinger Bands
        if close_price <= bb_lower:
            buy_score += 35
            
        if close_price >= bb_upper:
            sell_score += 35
            
        # 3. Avaliação do Histograma MACD
        if macd_hist > prev_macd_hist:
            buy_score += 25
        elif macd_hist < prev_macd_hist:
            sell_score += 25
            
        # 4. Matriz de Decisão Dinâmica
        if buy_score >= 85:
            decision = "COMPRA_FORTE"
        elif buy_score >= 60:
            decision = "COMPRA_MODERADA"
        elif buy_score >= 45:
            decision = "COMPRA_LEVE"
        elif sell_score >= 60:
            decision = "VENDA_FORTE"
            
        # Preparar dados de análise para retorno
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
                "ema_200": round(ema_200, 2)
            },
            "timestamp": str(last_candle.name) if hasattr(last_candle, 'name') else str(datetime.now())
        }
