import os
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import joblib
import logging
import numpy as np
from train_model import FEATURES  # fonte única da lista de features — V4

logger = logging.getLogger(__name__)


class TechnicalAnalysis:
    def __init__(self):
        self.model = None
        model_path = os.path.join(os.path.dirname(__file__), "scalper_model.pkl")
        try:
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("✅ Modelo de ML V4 carregado com sucesso!")
            else:
                logger.warning("⚠️ Modelo de ML não encontrado. O filtro preditivo estará inativo.")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar o modelo de ML: {e}")

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)

        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.atr(length=14, append=True)
        df.ta.obv(append=True)
        df.ta.vwap(append=True)
        df.ta.roc(length=5, append=True)
        df.ta.roc(length=15, append=True)

        # Retornos passados
        df['ret_1'] = df['close'].pct_change(1)
        df['ret_2'] = df['close'].pct_change(2)
        df['ret_3'] = df['close'].pct_change(3)

        # Geometria dos candles (normalizados pelo close — V4)
        df['Body_Size']  = abs(df['close'] - df['open']) / df['close']
        df['Upper_Wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
        df['Lower_Wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
        df['Wick_Ratio'] = df['Lower_Wick'] / (df['Upper_Wick'] + 0.00001)

        # Padrões de candle
        cdl = df.ta.cdl_pattern(name=["engulfing", "hammer", "morningstar"])
        if cdl is not None and isinstance(cdl, pd.DataFrame):
            for col in ['CDL_ENGULFING', 'CDL_HAMMER', 'CDL_MORNINGSTAR']:
                if col in cdl.columns and col not in df.columns:
                    df[col] = cdl[col]

        for col in ['CDL_ENGULFING', 'CDL_HAMMER', 'CDL_MORNINGSTAR']:
            if col not in df.columns:
                df[col] = 0
            else:
                df[col] = df[col].fillna(0).astype(int)

        # Cup and Handle
        H1 = df['high'].shift(5).rolling(window=35).max()
        L1 = df['low'].shift(5).rolling(window=35).min()
        H2 = df['high'].rolling(window=5).max()
        cup_depth       = (H1 - L1) / H1
        cup_edges_match = abs(H1 - H2) / H1
        handle_drop     = (H2 - df['close']) / H2
        cup_cond = (cup_depth > 0.015) & (cup_edges_match < 0.01) & (handle_drop > 0.002) & (handle_drop < 0.01)
        df['Cup_and_Handle'] = cup_cond.astype(int)

        return df

    def analisar_compra_venda(self, df: pd.DataFrame) -> dict:
        df = self.add_indicators(df)
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        cols = df.columns.tolist()
        try:
            bb_lower_col  = [c for c in cols if c.startswith('BBL_20_2')][0]
            bb_upper_col  = [c for c in cols if c.startswith('BBU_20_2')][0]
            macd_hist_col = [c for c in cols if c.startswith('MACDh_12_26_9')][0]
            rsi_col       = [c for c in cols if c.startswith('RSI_14')][0]
            ema_200_col   = [c for c in cols if c.startswith('EMA_200')][0]
            atr_col       = [c for c in cols if c.startswith('ATRr_14')][0]
            obv_col       = [c for c in cols if c.startswith('OBV')][0]
            vwap_col      = [c for c in cols if c.startswith('VWAP')][0]
            roc_5_col     = [c for c in cols if c.startswith('ROC_5')][0]
            roc_15_col    = [c for c in cols if c.startswith('ROC_15')][0]
        except IndexError:
            raise ValueError(f"Indicadores não encontrados. Colunas disponíveis: {cols}")

        rsi           = last_candle[rsi_col]
        close_price   = last_candle['close']
        atr_value     = last_candle[atr_col]
        bb_lower      = last_candle[bb_lower_col]
        bb_upper      = last_candle[bb_upper_col]
        macd_hist     = last_candle[macd_hist_col]
        prev_macd_hist = prev_candle[macd_hist_col]
        ema_200       = last_candle[ema_200_col]
        obv_val       = last_candle[obv_col]
        prev_obv      = df.iloc[-2][obv_col]
        vwap_val      = last_candle[vwap_col]

        decision   = "NEUTRO"
        buy_score  = 0
        sell_score = 0

        # ── Sinais de compra ──────────────────────────────────────────
        if rsi < 30:   buy_score += 40
        elif rsi < 40: buy_score += 20
        if close_price <= bb_lower: buy_score += 35
        if macd_hist > prev_macd_hist: buy_score += 25
        elif macd_hist < prev_macd_hist: sell_score += 25

        # ── Sinais de venda ───────────────────────────────────────────
        if rsi > 75:   sell_score += 40
        elif rsi > 65: sell_score += 20
        if close_price >= bb_upper: sell_score += 35

        # ── Bônus de padrões ──────────────────────────────────────────
        if last_candle['Cup_and_Handle'] == 1:
            buy_score += 40
            logger.info("☕ Padrão Xícara e Alça Detectado!")
        if (last_candle['CDL_ENGULFING'] == 100 or
                last_candle['CDL_HAMMER'] == 100 or
                last_candle['CDL_MORNINGSTAR'] == 100):
            if close_price <= bb_lower * 1.01:
                buy_score += 30
                logger.info("🕯️ Padrão Candlestick de Alta em suporte!")

        # ── Filtro de tendência (EMA 200) ─────────────────────────────
        tendencia_alta = close_price > ema_200
        if not tendencia_alta:
            buy_score = 0

        if   buy_score >= 85:  decision = "COMPRA_FORTE"
        elif buy_score >= 60:  decision = "COMPRA_MODERADA"
        elif buy_score >= 45:  decision = "COMPRA_LEVE"
        elif sell_score >= 60: decision = "VENDA_FORTE"

        # ── Inferência do modelo de ML V4 ────────────────────────────
        ml_prob_success = 0.0
        ml_prob_fail    = 0.0
        ml_status       = "NOT_EVALUATED"

        if self.model is not None:
            try:
                atr_pct    = atr_value / close_price
                dist_bbu   = (close_price - bb_upper) / close_price
                dist_bbl   = (close_price - bb_lower) / close_price
                # Features normalizadas V4
                obv_pct    = ((obv_val - prev_obv) / abs(prev_obv)) if prev_obv != 0 else 0.0
                obv_pct    = max(-1.0, min(1.0, obv_pct))  # clip
                vwap_dist  = (close_price - vwap_val) / vwap_val if vwap_val != 0 else 0.0

                X_last = pd.DataFrame([{
                    'RSI':          rsi,
                    'MACDh':        macd_hist,
                    'ATR_pct':      atr_pct,
                    'Dist_BBU':     dist_bbu,
                    'Dist_BBL':     dist_bbl,
                    'ret_1':        last_candle['ret_1'],
                    'ret_2':        last_candle['ret_2'],
                    'ret_3':        last_candle['ret_3'],
                    'OBV_pct':      obv_pct,
                    'VWAP_dist':    vwap_dist,
                    'ROC_5':        last_candle[roc_5_col],
                    'ROC_15':       last_candle[roc_15_col],
                    'Body_Size':    last_candle['Body_Size'],
                    'Upper_Wick':   last_candle['Upper_Wick'],
                    'Lower_Wick':   last_candle['Lower_Wick'],
                    'Wick_Ratio':   last_candle['Wick_Ratio'],
                    'CDL_ENGULFING':   last_candle['CDL_ENGULFING'],
                    'CDL_HAMMER':      last_candle['CDL_HAMMER'],
                    'CDL_MORNINGSTAR': last_candle['CDL_MORNINGSTAR'],
                    'Cup_and_Handle':  last_candle['Cup_and_Handle']
                }])[FEATURES]  # garante a ordem exata das features do treino

                proba          = self.model.predict_proba(X_last)
                ml_prob_fail    = proba[0][0]
                ml_prob_success = proba[0][1]

                # ── Defesa Ativa ──────────────────────────────────────
                if ml_prob_fail > 0.95:
                    decision  = "VENDA_FORTE"
                    ml_status = "DEFENSE_ACTIVE_STRONG"
                elif ml_prob_fail > 0.92:
                    decision  = "VENDA_MODERADA"
                    ml_status = "DEFENSE_ACTIVE_MODERATE"
                elif ml_prob_fail > 0.85:
                    decision  = "VENDA_LEVE"
                    ml_status = "DEFENSE_ACTIVE_LIGHT"

                # ── Filtro de entrada ─────────────────────────────────
                elif decision in ["COMPRA_FORTE", "COMPRA_MODERADA"]:
                    if atr_pct <= 0.003:
                        decision  = "COMPRA_BLOQUEADA_BAIXA_VOLATILIDADE"
                        ml_status = "REJECTED_LOW_VOL"
                    elif ml_prob_success < 0.60:
                        decision  = "COMPRA_BLOQUEADA_POR_ML"
                        ml_status = "REJECTED_BUY"
                    else:
                        ml_status = "APPROVED_BUY"

            except Exception as e:
                logger.error(f"Erro ao inferir ML: {e}")
                ml_status = "ERROR"

        return {
            "decision": decision,
            "analysis": {
                "rsi":              round(rsi, 2),
                "buy_score":        buy_score,
                "sell_score":       sell_score,
                "close_price":      close_price,
                "atr":              round(atr_value, 4),
                "ml_prob_success":  round(ml_prob_success, 4),
                "ml_prob_fail":     round(ml_prob_fail, 4),
                "ml_status":        ml_status
            },
            "timestamp": str(last_candle.name) if hasattr(last_candle, 'name') else str(datetime.now())
        }
