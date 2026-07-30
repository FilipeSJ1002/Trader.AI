import os
import pandas as pd
import pandas_ta as ta
from datetime import datetime
import joblib
import logging
import numpy as np
from train_model import FEATURES, add_all_features  # fonte única de features — V5 (MTF)

logger = logging.getLogger(__name__)

# EXIT_FEATURES importado do módulo de treino do exit model (se existir)
try:
    from train_exit_model import EXIT_FEATURES
except ImportError:
    EXIT_FEATURES = None


def _load_model_pkl(path: str, label: str):
    """Carrega um arquivo .pkl — suporta formato legado (modelo direto) e V4 (dict)."""
    if not os.path.exists(path):
        logger.warning(f"[ML] {label}: arquivo nao encontrado em {path}")
        return None, {}
    try:
        loaded = joblib.load(path)
        if isinstance(loaded, dict):
            model = loaded.get('model')
            meta  = {k: v for k, v in loaded.items() if k != 'model'}
            logger.info(f"[ML] {label} carregado | {meta}")
            return model, meta
        else:
            logger.info(f"[ML] {label} legado carregado.")
            return loaded, {}
    except Exception as e:
        logger.error(f"[ML] Erro ao carregar {label}: {e}")
        return None, {}


class TechnicalAnalysis:
    def __init__(self):
        base = os.path.dirname(__file__)

        # Modelo de entrada (scalper_model.pkl)
        self.model, _      = _load_model_pkl(os.path.join(base, "scalper_model.pkl"),  "Modelo de Entrada")

        # Modelo de saida (exit_model.pkl) — Fase 3.3
        self.exit_model, _ = _load_model_pkl(os.path.join(base, "exit_model.pkl"), "Modelo de Saida")

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)

        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        df.ta.ema(length=20, append=True)
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

        # ── Filtro de Volume (V4) ─────────────────────────────────────
        # vol_ratio = volume do candle atual / média dos últimos 20 candles
        # Indica se há interesse real do mercado na movimentação atual
        df['vol_sma20'] = df['volume'].rolling(window=20).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_sma20'] + 1e-9)

        return df

    def analisar_compra_venda(self, df: pd.DataFrame) -> dict:
        # Copia OHLCV crua para a engenharia de features do ML (paridade com treino)
        df_raw = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df = self.add_indicators(df)
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        # ── Features do ML computadas pela MESMA funcao do treino (MTF) ──────
        # Garante que entrada e saida usem exatamente as mesmas 27 features.
        feat_row = None
        try:
            feat_df = add_all_features(df_raw)
            if not feat_df.empty:
                feat_row = feat_df.iloc[-1]
        except Exception as e:
            logger.error(f"[ML] Falha ao computar features MTF: {e}")

        cols = df.columns.tolist()
        try:
            bb_lower_col  = [c for c in cols if c.startswith('BBL_20_2')][0]
            bb_upper_col  = [c for c in cols if c.startswith('BBU_20_2')][0]
            macd_hist_col = [c for c in cols if c.startswith('MACDh_12_26_9')][0]
            rsi_col       = [c for c in cols if c.startswith('RSI_14')][0]
            ema_20_col    = next((c for c in cols if c.startswith('EMA_20')), None)
            ema_200_col   = [c for c in cols if c.startswith('EMA_200')][0]
            atr_col       = [c for c in cols if c.startswith('ATRr_14')][0]
            obv_col       = [c for c in cols if c.startswith('OBV')][0]
            vwap_col      = [c for c in cols if c.startswith('VWAP')][0]
            roc_5_col     = [c for c in cols if c.startswith('ROC_5')][0]
            roc_15_col    = [c for c in cols if c.startswith('ROC_15')][0]
        except IndexError:
            raise ValueError(f"Indicadores não encontrados. Colunas disponíveis: {cols}")

        rsi            = last_candle[rsi_col]
        close_price    = last_candle['close']
        atr_value      = last_candle[atr_col]
        # ATR relativo ao preco: o filtro de baixa volatilidade usa percentual,
        # nao valor absoluto (sem isto, o filtro abaixo dava NameError e caia
        # no except, desligando silenciosamente a camada de ML)
        atr_pct        = float(atr_value) / float(close_price) if close_price else 0.0
        bb_lower       = last_candle[bb_lower_col]
        bb_upper       = last_candle[bb_upper_col]
        macd_hist      = last_candle[macd_hist_col]
        prev_macd_hist = prev_candle[macd_hist_col]
        ema_200        = last_candle[ema_200_col]
        obv_val        = last_candle[obv_col]
        prev_obv       = df.iloc[-2][obv_col]
        vwap_val       = last_candle[vwap_col]
        vol_ratio      = last_candle.get('vol_ratio', 1.0)   # V4: volume relativo

        decision   = "NEUTRO"
        buy_score  = 0
        sell_score = 0

        # ── Filtro de Volume (V4) ─────────────────────────────────────
        # Baixo volume = mercado sem interesse = sinal não confiável
        volume_ok = vol_ratio >= 0.8
        if not volume_ok:
            logger.debug(f"Volume baixo (ratio={vol_ratio:.2f}) — entrada bloqueada preventivamente.")

        # ── Sinais de compra ──────────────────────────────────────────
        if rsi < 30:   buy_score += 40
        elif rsi < 40: buy_score += 20
        if close_price <= bb_lower: buy_score += 35
        if macd_hist > prev_macd_hist: buy_score += 25
        elif macd_hist < prev_macd_hist: sell_score += 25

        # ── Bônus de volume alto (V4) ─────────────────────────────────
        # Volume acima de 1.5× a média confirma que o movimento tem força
        if vol_ratio >= 1.5:
            buy_score  += 15
            sell_score += 10   # também aumenta confiança em sinais de venda

        # ── Sinais de venda ───────────────────────────────────────────
        if rsi > 75:   sell_score += 40
        elif rsi > 65: sell_score += 20
        if close_price >= bb_upper: sell_score += 35

        # ── Bônus de padrões ──────────────────────────────────────────
        if last_candle['Cup_and_Handle'] == 1:
            buy_score += 40
            logger.info("Padrao Xicara e Alca Detectado!")
        if (last_candle['CDL_ENGULFING'] == 100 or
                last_candle['CDL_HAMMER'] == 100 or
                last_candle['CDL_MORNINGSTAR'] == 100):
            if close_price <= bb_lower * 1.01:
                buy_score += 30
                logger.info("Padrao Candlestick de Alta em suporte!")

        # ── Filtro de tendência (EMA 200) ─────────────────────────────
        tendencia_alta = close_price > ema_200
        if not tendencia_alta:
            buy_score = 0

        if   buy_score >= 85:  decision = "COMPRA_FORTE"
        elif buy_score >= 60:  decision = "COMPRA_MODERADA"
        elif buy_score >= 45:  decision = "COMPRA_LEVE"
        elif sell_score >= 60: decision = "VENDA_FORTE"

        # ── Bloqueia entrada com volume insuficiente ──────────────────
        if not volume_ok and decision.startswith("COMPRA_"):
            decision = "BLOQUEADO_VOLUME_BAIXO"

        # ── Inferência do modelo de ML V4 ────────────────────────────
        ml_prob_success = 0.0
        ml_prob_fail    = 0.0
        ml_status       = "NOT_EVALUATED"

        # Features auxiliares ainda usadas pelo exit model / payload
        obv_pct    = ((obv_val - prev_obv) / abs(prev_obv)) if prev_obv != 0 else 0.0
        obv_pct    = max(-1.0, min(1.0, obv_pct))
        vwap_dist  = (close_price - vwap_val) / vwap_val if vwap_val != 0 else 0.0

        if self.model is not None and feat_row is not None:
            try:
                # Vetor de features identico ao treino (20 de 1M + 7 de MTF)
                X_last = pd.DataFrame([feat_row[FEATURES].to_dict()])[FEATURES]

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

        # ── Modelo de Saída (Exit Model) — Fase 3.3 ─────────────────────────────
        # Prediz a probabilidade de queda significativa nos próximos 15 minutos.
        # Útil para sair de posições lucrativas ANTES de o SL ser atingido.
        exit_prob = 0.0
        if self.exit_model is not None and EXIT_FEATURES is not None and feat_row is not None:
            try:
                # Reaproveita as features base do feat_row (paridade com treino)
                # e adiciona as 3 features exclusivas do exit model.
                rsi_series   = df[rsi_col]
                bb_squeeze_v = (float(last_candle[bb_upper_col]) - float(last_candle[bb_lower_col])) / (close_price + 1e-9)
                rsi_high_v   = float(rsi_series.iloc[-5:].max()) if len(rsi_series) >= 5 else rsi
                bear_mom_v   = float(macd_hist - df.iloc[-3][macd_hist_col]) if len(df) >= 3 else 0.0

                exit_data = {f: feat_row[f] for f in EXIT_FEATURES if f in feat_row.index}
                exit_data['RSI_high']      = rsi_high_v
                exit_data['BB_squeeze']    = bb_squeeze_v
                exit_data['Bear_momentum'] = bear_mom_v

                X_exit = pd.DataFrame([exit_data])[EXIT_FEATURES]
                exit_prob = float(self.exit_model.predict_proba(X_exit)[0][1])
            except Exception as e:
                logger.debug(f"[EXIT MODEL] Erro na inferencia: {e}")

        return {
            "decision": decision,
            "analysis": {
                "rsi":              round(rsi, 2),
                "buy_score":        buy_score,
                "sell_score":       sell_score,
                "close_price":      close_price,
                "atr":              round(atr_value, 4),
                "vol_ratio":        round(float(vol_ratio), 3),
                "ml_prob_success":  round(ml_prob_success, 4),
                "ml_prob_fail":     round(ml_prob_fail, 4),
                "ml_status":        ml_status,
                "exit_prob":        round(exit_prob, 4),  # prob de queda — Fase 3.3
                # Vetor de features no momento da analise — V5 (loop de aprendizado).
                # Salvo na entrada para reinjetar a experiencia real no retreino.
                "features": ({f: float(feat_row[f]) for f in FEATURES}
                             if feat_row is not None else None),
            },
            "timestamp": str(last_candle.name) if hasattr(last_candle, 'name') else str(datetime.now())
        }
