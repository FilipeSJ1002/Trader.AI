"""
market_state.py — V4
Gerencia o estado global do mercado em memória.

Novidades V4:
  - Resample dinâmico 1M → 5M e 15M (sem streams extras)
  - Detector de Regime de Mercado por símbolo (ADX + EMA slope no 15M)
  - Relative Strength entre todos os ativos ativos
  - HTF weakness mantida para retrocompatibilidade
"""
import logging
import pandas as pd
import pandas_ta as ta
from strategy import TechnicalAnalysis

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Estados globais por símbolo
# ─────────────────────────────────────────────────────────────────────────────
history_data:   dict[str, pd.DataFrame] = {}   # candles 1M
history_1h:     dict[str, pd.DataFrame] = {}   # candles 1H (do WebSocket)
history_5m:     dict[str, pd.DataFrame] = {}   # candles 5M (resampled do 1M)
history_15m:    dict[str, pd.DataFrame] = {}   # candles 15M (resampled do 1M)
history_daily:  dict[str, pd.Series]    = {}   # closes diarios (V6 — trend-following SMA200)

htf_weakness:   dict[str, bool]  = {}          # fraqueza no HTF (1H)
market_regime:  dict[str, str]   = {}          # BULL_TREND | SIDEWAYS | BEAR_TREND
daily_returns:  dict[str, float] = {}          # retorno das últimas 24h

strategy = TechnicalAnalysis()

# ─────────────────────────────────────────────────────────────────────────────
# Constantes de Regime
# ─────────────────────────────────────────────────────────────────────────────
REGIME_BULL    = "BULL_TREND"
REGIME_SIDE    = "SIDEWAYS"
REGIME_BEAR    = "BEAR_TREND"
REGIME_UNKNOWN = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Inicialização de histórico
# ─────────────────────────────────────────────────────────────────────────────
def set_historical_data(symbol: str, df: pd.DataFrame):
    """Inicializa o buffer de 1M e pré-calcula 5M / 15M."""
    global history_data, history_5m, history_15m
    history_data[symbol] = df
    if symbol not in htf_weakness:
        htf_weakness[symbol]  = False
    if symbol not in market_regime:
        market_regime[symbol] = REGIME_UNKNOWN

    # Pré-calcula timeframes maiores a partir do histórico inicial
    _resample_and_store(symbol, df)


def set_historical_1h_data(symbol: str, df: pd.DataFrame):
    """Inicializa o histórico 1H na memória."""
    global history_1h
    history_1h[symbol] = df


# ─────────────────────────────────────────────────────────────────────────────
# Dados diários — V6 (Trend-Following SMA200)
# ─────────────────────────────────────────────────────────────────────────────
def set_historical_daily(symbol: str, close_series: pd.Series):
    """Inicializa a série de closes diários (para a SMA200 do trend-following)."""
    global history_daily
    history_daily[symbol] = close_series.dropna()


def update_daily_close(symbol: str, candle_date, close: float):
    """
    Atualiza/insere o close do dia atual na série diária.
    Chamado a cada candle 1M fechado — sobrescreve o close do dia corrente.
    """
    global history_daily
    if symbol not in history_daily:
        history_daily[symbol] = pd.Series(dtype=float)
    day = pd.to_datetime(candle_date).normalize()
    history_daily[symbol].loc[day] = float(close)
    # mantém histórico suficiente para SMA200 com folga
    if len(history_daily[symbol]) > 400:
        history_daily[symbol] = history_daily[symbol].tail(400)


def get_daily_closes() -> dict:
    """Retorna {symbol: Series de closes diários} para o trend_strategy."""
    return {s: c for s, c in history_daily.items() if c is not None and len(c) > 0}


def update_price_only(symbol: str, candle: dict) -> float:
    """
    Atualização leve (V6 trend-following): adiciona o candle 1M ao buffer de preço
    e atualiza o close diário, SEM rodar a análise de scalping (não é mais usada).
    Retorna o preço de fechamento.
    """
    global history_data
    if symbol not in history_data:
        history_data[symbol] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    new_df = pd.DataFrame([candle]).set_index('date')
    history_data[symbol] = pd.concat([history_data[symbol], new_df])
    history_data[symbol] = history_data[symbol][~history_data[symbol].index.duplicated(keep='last')]
    if len(history_data[symbol]) > 2000:
        history_data[symbol] = history_data[symbol].tail(2000)

    update_daily_close(symbol, candle['date'], candle['close'])
    return float(candle['close'])


# ─────────────────────────────────────────────────────────────────────────────
# Resample 1M → 5M / 15M
# ─────────────────────────────────────────────────────────────────────────────
def _resample_and_store(symbol: str, df_1m: pd.DataFrame):
    """
    Recalcula buffers de 5M e 15M a partir dos últimos N candles de 1M.
    Chamado toda vez que um novo candle 1M é adicionado.
    Usa apenas os últimos 1500 candles de 1M para eficiência.
    """
    global history_5m, history_15m

    src = df_1m.tail(1500) if len(df_1m) > 1500 else df_1m

    try:
        for period, store in (('5min', history_5m), ('15min', history_15m)):
            rs = src.resample(period)
            store[symbol] = pd.DataFrame({
                'open':   rs['open'].first(),
                'high':   rs['high'].max(),
                'low':    rs['low'].min(),
                'close':  rs['close'].last(),
                'volume': rs['volume'].sum(),
            }).dropna()
    except Exception as e:
        logger.warning(f"[MTF] Erro ao resamplear {symbol}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Detector de Regime de Mercado (ADX + EMA slope no 15M)
# ─────────────────────────────────────────────────────────────────────────────
def _calculate_regime(symbol: str) -> str:
    """
    Classifica o regime de mercado atual do símbolo usando o timeframe de 15M.

    Lógica:
      - ADX > 25 E EMA20 subindo  → BULL_TREND
      - ADX > 25 E EMA20 caindo   → BEAR_TREND
      - ADX <= 25                 → SIDEWAYS
    """
    df = history_15m.get(symbol)
    if df is None or len(df) < 30:
        return REGIME_UNKNOWN

    df = df.copy()
    try:
        df.ta.adx(length=14, append=True)
        df.ta.ema(length=20, append=True)

        cols = df.columns.tolist()
        adx_col = [c for c in cols if c.startswith('ADX_')][0]
        ema_col = [c for c in cols if c.startswith('EMA_20')][0]

        last  = df.iloc[-1]
        prev  = df.iloc[-3]   # 3 candles de 15M = 45 minutos de slope

        adx_val   = last[adx_col]
        ema_now   = last[ema_col]
        ema_prev  = prev[ema_col]

        if pd.isna(adx_val) or pd.isna(ema_now) or pd.isna(ema_prev):
            return REGIME_UNKNOWN

        ema_slope_pct = (ema_now - ema_prev) / ema_prev * 100

        if adx_val > 25:
            return REGIME_BULL if ema_slope_pct > 0 else REGIME_BEAR
        else:
            return REGIME_SIDE

    except Exception as e:
        logger.warning(f"[REGIME] Erro ao calcular regime de {symbol}: {e}")
        return REGIME_UNKNOWN


# ─────────────────────────────────────────────────────────────────────────────
# Fraqueza no HTF (1H) — mantida para retrocompatibilidade
# ─────────────────────────────────────────────────────────────────────────────
def update_1h_candle(symbol: str, candle: dict):
    """Atualiza o buffer de 1H e recalcula a fraqueza HTF e o retorno diário."""
    global history_1h, htf_weakness, daily_returns

    if symbol not in history_1h:
        history_1h[symbol] = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    candle_date = candle['date']
    history_1h[symbol].loc[candle_date] = [
        candle['open'], candle['high'], candle['low'], candle['close'], candle['volume']
    ]

    if len(history_1h[symbol]) > 500:
        history_1h[symbol] = history_1h[symbol].tail(500)

    # Retorno das últimas 24 horas
    if len(history_1h[symbol]) > 0:
        lookback = 24 if len(history_1h[symbol]) >= 24 else len(history_1h[symbol])
        p_first = history_1h[symbol].iloc[-lookback]['close']
        p_last  = history_1h[symbol].iloc[-1]['close']
        daily_returns[symbol] = (p_last - p_first) / p_first

    if len(history_1h[symbol]) < 30:
        htf_weakness[symbol] = False
        return

    df_temp = history_1h[symbol].copy()
    df_temp.ta.rsi(length=14, append=True)
    df_temp.ta.macd(fast=12, slow=26, signal=9, append=True)

    last_1h = df_temp.iloc[-1]
    prev_1h = df_temp.iloc[-2]

    cols = df_temp.columns.tolist()
    try:
        macd_col = [c for c in cols if c.startswith('MACDh_')][0]
        rsi_col  = [c for c in cols if c.startswith('RSI_')][0]
    except IndexError:
        return

    rsi       = last_1h[rsi_col]
    macd_hist = last_1h[macd_col]
    prev_macd = prev_1h[macd_col]

    weakness = False
    if rsi < 45:
        weakness = True
    elif macd_hist < 0 and macd_hist < prev_macd:
        weakness = True
    elif rsi > 70 and macd_hist < prev_macd:
        weakness = True

    htf_weakness[symbol] = weakness


# ─────────────────────────────────────────────────────────────────────────────
# Sinal MTF de 5M (confirmação de momentum)
# ─────────────────────────────────────────────────────────────────────────────
def _get_5m_signal(symbol: str) -> str:
    """
    Analisa o timeframe de 5M para confirmar (ou negar) um sinal de compra.
    Retorna: 'BULL', 'BEAR' ou 'NEUTRAL'
    """
    df = history_5m.get(symbol)
    if df is None or len(df) < 20:
        return "NEUTRAL"

    df = df.copy()
    try:
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.ema(length=20, append=True)

        cols  = df.columns.tolist()
        rsi_col  = [c for c in cols if c.startswith('RSI_')][0]
        macd_col = [c for c in cols if c.startswith('MACDh_')][0]
        ema_col  = [c for c in cols if c.startswith('EMA_20')][0]

        last = df.iloc[-1]
        prev = df.iloc[-2]

        rsi       = last[rsi_col]
        macd_hist = last[macd_col]
        prev_macd = prev[macd_col]
        ema_val   = last[ema_col]
        close     = last['close']

        bullish = (rsi > 45) and (macd_hist > prev_macd) and (close > ema_val)
        bearish = (rsi < 55) and (macd_hist < prev_macd) and (close < ema_val)

        if bullish:  return "BULL"
        if bearish:  return "BEAR"
        return "NEUTRAL"

    except Exception as e:
        logger.warning(f"[5M] Erro ao calcular sinal 5M de {symbol}: {e}")
        return "NEUTRAL"


# ─────────────────────────────────────────────────────────────────────────────
# Processamento de novo candle 1M
# ─────────────────────────────────────────────────────────────────────────────
def append_new_candle(symbol: str, candle: dict) -> dict:
    """
    Adiciona um novo candle 1M ao buffer, atualiza os timeframes maiores,
    calcula o regime de mercado e executa a análise técnica completa.
    """
    global history_data, market_regime

    if symbol not in history_data or history_data[symbol].empty:
        raise Exception(f"Dados históricos não carregados para {symbol}.")

    new_df = pd.DataFrame([candle])
    new_df.set_index('date', inplace=True)

    history_data[symbol] = pd.concat([history_data[symbol], new_df])
    history_data[symbol] = history_data[symbol][~history_data[symbol].index.duplicated(keep='last')]

    # Buffer aumentado para 25k candles (~17 dias) — necessario para que a
    # EMA200 de 1H (≈230h) seja computavel no resample MTF do add_all_features.
    if len(history_data[symbol]) > 25005:
        history_data[symbol] = history_data[symbol].tail(25000)

    # ── Atualiza 5M / 15M e recalcula regime ─────────────────────────
    _resample_and_store(symbol, history_data[symbol])
    market_regime[symbol] = _calculate_regime(symbol)
    regime = market_regime[symbol]

    # ── Sinal de confirmação no 5M ────────────────────────────────────
    signal_5m = _get_5m_signal(symbol)

    # ── Análise técnica principal (1M) ────────────────────────────────
    resultado = strategy.analisar_compra_venda(history_data[symbol].copy())
    decision  = resultado.get("decision", "NEUTRO")

    # ── Filtro de Regime ──────────────────────────────────────────────
    if decision.startswith("COMPRA_"):
        if regime == REGIME_BEAR:
            decision  = "BLOQUEADO_REGIME_BAIXA"
            logger.info(f"[REGIME] {symbol} Compra bloqueada: mercado em BEAR_TREND.")
        elif regime == REGIME_SIDE and signal_5m == "BEAR":
            decision  = "BLOQUEADO_REGIME_LATERAL"
            logger.info(f"[REGIME] {symbol} Compra bloqueada: lateral + 5M bearish.")
        elif regime == REGIME_BULL and signal_5m == "BULL":
            # Bônus: regime favorável em todos os TFs → eleva a decisão
            if decision == "COMPRA_MODERADA":
                decision = "COMPRA_FORTE"
                logger.info(f"[REGIME] {symbol} COMPRA_MODERADA -> COMPRA_FORTE (regime+5M confirmados)")

    resultado["decision"] = decision

    # ── Fraqueza HTF (1H) ─────────────────────────────────────────────
    weakness = htf_weakness.get(symbol, False)
    if weakness and decision.startswith("COMPRA_"):
        resultado["decision"] = "BLOQUEADO_POR_HTF"

    # ── Relative Strength ─────────────────────────────────────────────
    if daily_returns:
        max_symbol = max(daily_returns, key=lambda k: daily_returns[k])
        min_symbol = min(daily_returns, key=lambda k: daily_returns[k])

        ret = daily_returns.get(symbol, 0)
        if symbol == max_symbol and ret > 0.005:   # >0.5% de retorno no dia
            resultado["analysis"]["asset_strength"] = "STRONG"
            if resultado["decision"] == "COMPRA_MODERADA":
                resultado["decision"] = "COMPRA_FORTE"
        elif symbol == min_symbol and ret < -0.005:
            resultado["analysis"]["asset_strength"] = "WEAK"
            if resultado["decision"].startswith("COMPRA_"):
                resultado["decision"] = "BLOQUEADO_FORCA_RELATIVA_FRACA"
        else:
            resultado["analysis"]["asset_strength"] = "NEUTRAL"

    # ── Enriquece o payload de análise ───────────────────────────────
    if "analysis" in resultado:
        resultado["analysis"]["htf_weakness"]   = weakness
        resultado["analysis"]["market_regime"]  = regime
        resultado["analysis"]["signal_5m"]      = signal_5m
        resultado["analysis"]["daily_return"]   = round(daily_returns.get(symbol, 0) * 100, 3)

    return resultado
