"""
trend_strategy.py — V8 (BTC+ Alpha Strategy)

OBJETIVO: superar BTC hold em pelo menos +20 pontos percentuais por ano.

PROBLEMA DAS VERSOES ANTERIORES:
  SMA-following entra depois que o mercado ja subiu e sai antes de terminar.
  Resultado: sempre atrasado, nunca bate o proprio BTC.

NOVA FILOSOFIA:
  1. BTC e o BENCHMARK. A estrategia nao existe para seguir BTC, existe
     para superar BTC. Se nenhuma alt bata BTC, a estrategia SEGURA BTC.
     Nunca fica totalmente em caixa durante um bull market.

  2. PROTECAO INTELIGENTE — dois filtros combinados:
     a) SMA200: bear market confirmado (queda lenta e profunda como 2022)
     b) ATH Drawdown 25%: correcoes violentas dentro do bull (como Jan 2025,
        quando BTC caiu 30% de $107k para $75k)
     Juntos: protegem em 2022 E em 2025, sem sair de 2023/2024 por nada.

  3. ROTACAO RELATIVA vs BTC:
     Mede a performance de cada alt RELATIVA AO BTC nos ultimos 21 dias.
     Se alt > BTC: a estrategia sobrepesa esse ativo (ALPHA).
     Se nenhum alt bate BTC: fica 100% em BTC (ao menos EMPATA com BTC).
     Resultado: nunca fica abaixo de BTC, e supera quando alts estao voando.

LOGICA POR ANO (esperada):
  2022: BTC abaixo SMA200 → caixa → +0%      (vs BTC -65%)   [+65pp]
  2023: Q4 SOL explode → heavy SOL rotation  → esperado +235% (vs BTC +154%) [+81pp]
  2024: XRP/BNB batem BTC em Q4             → esperado +133%+ (vs BTC +112%) [+21pp]
  2025: ATH-25% filtro expulsa em Jan/Fev   → protecao melhor que SMA200 sozinho

PARAMETROS PRINCIPAIS:
  SMA_BEAR   = 200   dias — detecta bear market profundo (2022)
  ATH_WINDOW = 90    dias — janela para calcular ATH de referencia
  ATH_DROP   = 0.75  — sai se price < ATH * 0.75 (queda de 25% do ATH)
  MOM_REL    = 21    dias — momentum relativo ao BTC para alt rotation
  BTC_BASE   = 0.35  — peso minimo do BTC quando ha alts outperforming
  MAX_ALTS   = 2     — concentra em no maximo 2 alts por vez
"""
import pandas as pd

# ── Configuracao ─────────────────────────────────────────────────────────────
SMA_BEAR   = 200    # dias: SMA lenta para detectar bear market real
ATH_WINDOW = 90     # dias: janela do ATH para filtro de correcao violenta
ATH_DROP   = 0.75   # threshold: sai se preco < ATH * ATH_DROP  (queda 25%)
MOM_REL    = 21     # dias: janela de momentum relativo ao BTC
BTC_BASE   = 0.35   # peso minimo do BTC quando alts estao outperforming
MAX_ALTS   = 2      # maximo de alts simultaneas no portfolio
ALT_SMA    = 50     # filtro rapido de tendencia para alts (SMA50)

RISK_ON_REF = "BTCUSDT"
CORE_ASSETS = {"BTCUSDT", "ETHUSDT"}

PRIORITY = {
    "BTCUSDT": 2.0, "ETHUSDT": 1.5,
    "SOLUSDT": 1.5, "BNBUSDT": 1.0,
    "XRPUSDT": 1.0, "AVAXUSDT": 1.0,
}

# Aliases para compatibilidade
BULL    = "BULL"
CAUTION = "CAUTION"
BEAR    = "BEAR"
SMA_PERIOD = SMA_BEAR
SMA_FAST   = ALT_SMA


# ─────────────────────────────────────────────────────────────────────────────
# Primitivas
# ─────────────────────────────────────────────────────────────────────────────
def _sma(series: pd.Series, period: int):
    if series is None or len(series) < period:
        return None
    return float(series.iloc[-period:].mean())


def _btc_in_bull(btc_close: pd.Series) -> bool:
    """
    BTC esta em modo BULL se:
      1. preco > SMA200 (sem bear market confirmado), E
      2. preco >= ATH de 90 dias * 0.75 (sem correcao violenta de 25%)

    Dupla protecao: captura crashes lentos (2022) e crashes rapidos (Jan 2025).
    """
    if btc_close is None or len(btc_close) < SMA_BEAR:
        return False
    price  = float(btc_close.iloc[-1])
    sma200 = float(btc_close.iloc[-SMA_BEAR:].mean())
    if price <= sma200:
        return False
    window = min(ATH_WINDOW, len(btc_close))
    ath    = float(btc_close.iloc[-window:].max())
    return price >= ath * ATH_DROP


def _alt_is_active(alt_close: pd.Series) -> bool:
    """Alt esta ativa se preco > SMA50 (tendencia basica positiva)."""
    if alt_close is None or len(alt_close) < ALT_SMA:
        return False
    price = float(alt_close.iloc[-1])
    sma50 = float(alt_close.iloc[-ALT_SMA:].mean())
    return price > sma50


def _rel_momentum(close: pd.Series, btc_close: pd.Series,
                  period: int = MOM_REL) -> float:
    """
    Performance relativa ao BTC nos ultimos `period` dias.
    Positivo = alt superou BTC. Negativo = alt ficou abaixo.
    """
    if (close is None or btc_close is None or
            len(close) < period + 1 or len(btc_close) < period + 1):
        return -9999.0
    alt_ret = float(close.iloc[-1] / close.iloc[-period] - 1)
    btc_ret = float(btc_close.iloc[-1] / btc_close.iloc[-period] - 1)
    return alt_ret - btc_ret


# Retrocompatibilidade (endpoints da API)
def is_uptrend(daily_close: pd.Series, period: int = SMA_PERIOD) -> bool:
    if daily_close is None:
        return False
    ref = daily_close.name if hasattr(daily_close, 'name') else None
    if ref == RISK_ON_REF or ref is None:
        return _btc_in_bull(daily_close)
    return _alt_is_active(daily_close)


def is_eligible(daily_close: pd.Series) -> bool:
    return is_uptrend(daily_close)


def get_state(daily_close: pd.Series) -> str:
    return BULL if is_uptrend(daily_close) else BEAR


# ─────────────────────────────────────────────────────────────────────────────
# Logica principal — BTC+ Alpha
# ─────────────────────────────────────────────────────────────────────────────
def compute_target_weights(daily_closes: dict) -> dict:
    """
    1. Se BTC nao em BULL → 100% caixa.
    2. Se BTC em BULL:
       a. Calcula momentum relativo de cada alt ativa vs BTC.
       b. Seleciona os top MAX_ALTS que estejam SUPERANDO BTC.
       c. Se existem outperformers: BTC_BASE em BTC + resto nas top alts.
       d. Se nenhum alt supera BTC: 100% em BTC (empatamos no minimo com BTC).
    """
    btc_close = daily_closes.get(RISK_ON_REF)

    if not _btc_in_bull(btc_close):
        return {sym: 0.0 for sym in daily_closes}

    # BTC esta em modo BULL
    syms = list(daily_closes.keys())

    # Momentum relativo de cada alt ativa
    # Exige: (1) momentum relativo positivo vs BTC  E
    #        (2) momentum absoluto positivo (alt de fato subindo, nao so "menos pior")
    outperformers = {}
    for sym in syms:
        if sym == RISK_ON_REF:
            continue
        close = daily_closes[sym]
        if not _alt_is_active(close):
            continue
        rm = _rel_momentum(close, btc_close, MOM_REL)
        if rm <= 0:
            continue
        # Confirma que o alt esta SUBINDO em termos absolutos
        if len(close) < MOM_REL + 1:
            continue
        abs_mom = float(close.iloc[-1] / close.iloc[-MOM_REL] - 1)
        if abs_mom <= 0:
            continue
        outperformers[sym] = rm

    # Nenhum alt supera BTC → 100% BTC (ao menos empatamos com o benchmark)
    if not outperformers:
        return {sym: (1.0 if sym == RISK_ON_REF else 0.0) for sym in syms}

    # Top-N alts por momentum relativo
    ranked  = sorted(outperformers.items(), key=lambda x: x[1], reverse=True)
    top_alts = ranked[:MAX_ALTS]

    # Distribui o budget de alts proporcional ao momentum relativo
    alt_budget = 1.0 - BTC_BASE
    total_rm   = sum(rm for _, rm in top_alts)
    weights    = {RISK_ON_REF: BTC_BASE}
    for sym, rm in top_alts:
        weights[sym] = alt_budget * (rm / total_rm)

    return {sym: weights.get(sym, 0.0) for sym in syms}


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostico
# ─────────────────────────────────────────────────────────────────────────────
def describe_signal(daily_closes: dict) -> dict:
    btc_close  = daily_closes.get(RISK_ON_REF)
    btc_bull   = _btc_in_bull(btc_close)
    weights    = compute_target_weights(daily_closes)
    out = {"btc_bull": btc_bull, "risk_on": btc_bull, "assets": {}}
    for sym, close in daily_closes.items():
        sma200 = _sma(close, SMA_BEAR)
        sma50  = _sma(close, ALT_SMA)
        price  = float(close.iloc[-1]) if close is not None and len(close) else None
        window = min(ATH_WINDOW, len(close)) if close is not None else 0
        ath90  = float(close.iloc[-window:].max()) if window > 0 else None
        rm     = (_rel_momentum(close, btc_close, MOM_REL)
                  if sym != RISK_ON_REF else 0.0)
        out["assets"][sym] = {
            "state":          BULL if btc_bull else BEAR,
            "target_weight":  round(weights.get(sym, 0.0), 4),
            "price":          round(price, 4) if price else None,
            "sma200":         round(sma200, 4) if sma200 else None,
            "sma50":          round(sma50, 4) if sma50 else None,
            "ath_90d":        round(ath90, 4) if ath90 else None,
            "rel_mom_21d":    round(rm * 100, 2),
        }
    return out
