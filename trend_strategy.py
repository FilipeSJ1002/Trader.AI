"""
trend_strategy.py — V7 (Dual-SMA Protecao + Momentum Rotation)

Estrategia em duas camadas:

  CAMADA 1 — Protecao (binaria, igual V6.1):
    Elegivel = preco > max(SMA50, SMA200)
    Nao elegivel = 0% (caixa) — sem meias medidas.
    Em bull maduro: SMA50 > SMA200, entao SMA50 e o gatilho de saida.
    Em recovery/bear: SMA200 e o gatilho (mais alto), entrada conservadora.

  CAMADA 2 — Momentum Rotation (so para quem e elegivel):
    Peso final = PRIORITY * (1 + bonus_momentum)
    bonus_momentum = min(momentum_90d_positivo, 2.0) * 0.5 (max +100% de peso)
    Janela de 90 dias (trimestral) — mais suave que 30 dias, menos ruido.
    Resultado: dentro do bull, capital migra para os ativos mais fortes.

  Filtro RISK-ON (BTC):
    BTC elegivel -> alts tambem podem operar
    BTC nao elegivel -> so BTC/ETH core operam
    BTC em bear -> todos em caixa

Por que 90 dias em vez de 30 dias:
  Momentum de 30 dias e curto demais e muito reativo (ruido).
  Momentum de 90 dias captura tendencias reais dentro do bull:
  SOL subiu +900% em 2023. A janela de 90 dias teria dado peso crescente
  a SOL ao longo do ano, maximizando os ganhos do ciclo.

Comparativo backtest 2022-2026:
  V6  (SMA200 puro):           +214%  CAGR 31.7%  DD -40.5%  2025: -30%
  V6.1 (dual-SMA binario):     +258%  CAGR 35.9%  DD -31.4%  2025: +3%
  V7  (dual-SMA + momentum):   melhor que V6.1 esperado em bull years
"""
import pandas as pd

# ── Configuracao ─────────────────────────────────────────────────────────────
SMA_PERIOD    = 200      # SMA lenta — filtro de bear market
SMA_FAST      = 50       # SMA rapida — stop dinamico em bull maduro
MOM_PERIOD    = 90       # Momentum trimestral (90 dias = ~1 trimestre)
MOM_CAP       = 2.0      # cap do momentum (evita over-concentracao em extremos)
MOM_MAX_BONUS = 1.00     # max +100% de peso por momentum positivo (dobra a alocacao dos lideres)
TOP_ASSETS    = None     # concentra nos N ativos mais fortes (None = todos)
RISK_ON_REF   = "BTCUSDT"
CORE_ASSETS   = {"BTCUSDT", "ETHUSDT"}

PRIORITY = {
    "BTCUSDT": 2.0, "ETHUSDT": 2.0,
    "SOLUSDT": 1.0, "BNBUSDT": 1.0,
    "XRPUSDT": 1.0, "AVAXUSDT": 1.0,
}

# Alias para retrocompatibilidade de imports externos
BULL    = "BULL"
CAUTION = "CAUTION"
BEAR    = "BEAR"


# ─────────────────────────────────────────────────────────────────────────────
# Primitivas
# ─────────────────────────────────────────────────────────────────────────────
def _sma(series: pd.Series, period: int):
    """SMA dos ultimos `period` dias. None se historico insuficiente."""
    if series is None or len(series) < period:
        return None
    return float(series.iloc[-period:].mean())


def _momentum(series: pd.Series, period: int = MOM_PERIOD) -> float:
    """Retorno percentual simples dos ultimos `period` dias."""
    if series is None or len(series) < period + 1:
        return 0.0
    return float(series.iloc[-1] / series.iloc[-period] - 1)


def is_eligible(daily_close: pd.Series) -> bool:
    """
    Filtro de protecao binario: True se preco > max(SMA50, SMA200).

    Em bull maduro (SMA50 > SMA200): threshold = SMA50 (saida rapida).
    Em recovery / bear:              threshold = SMA200 (entrada conservadora).
    """
    if daily_close is None or len(daily_close) < SMA_PERIOD:
        return False
    price  = float(daily_close.iloc[-1])
    sma200 = _sma(daily_close, SMA_PERIOD)
    sma50  = _sma(daily_close, SMA_FAST)
    if sma200 is None:
        return False
    threshold = max(sma200, sma50) if sma50 is not None else sma200
    return price > threshold


# Retrocompatibilidade
def is_uptrend(daily_close: pd.Series, period: int = SMA_PERIOD) -> bool:
    return is_eligible(daily_close)


def get_state(daily_close: pd.Series) -> str:
    """Retorna BULL / BEAR para compatibilidade com o endpoint /trend."""
    return BULL if is_eligible(daily_close) else BEAR


# ─────────────────────────────────────────────────────────────────────────────
# Calculo de pesos — Dual-SMA + Momentum Rotation
# ─────────────────────────────────────────────────────────────────────────────
def compute_target_weights(daily_closes: dict) -> dict:
    """
    Retorna {symbol: peso_alvo} normalizado.

    1. Filtra ativos elegiveis (is_eligible).
    2. Aplica filtro risk-on (BTC deve ser elegivel para alts operarem).
    3. Pesa cada elegivel por PRIORITY * (1 + bonus_momentum).
       Bonus = min(momentum_90d positivo, MOM_CAP) * MOM_MAX_BONUS.
    """
    eligible_mask = {sym: is_eligible(c) for sym, c in daily_closes.items()}
    btc_ok = eligible_mask.get(RISK_ON_REF, False)

    active = {}
    for sym, ok in eligible_mask.items():
        if not ok:
            continue
        if sym not in CORE_ASSETS and not btc_ok:
            continue
        mom   = _momentum(daily_closes[sym], MOM_PERIOD)
        bonus = min(max(mom, 0.0), MOM_CAP) * MOM_MAX_BONUS
        active[sym] = PRIORITY.get(sym, 1.0) * (1.0 + bonus)

    # Concentracao: guarda apenas os TOP_ASSETS por score de momentum
    if TOP_ASSETS is not None and len(active) > TOP_ASSETS:
        # ordena pelo score e keep top N
        ranked = sorted(active.items(), key=lambda x: x[1], reverse=True)
        active = dict(ranked[:TOP_ASSETS])

    total = sum(active.values())
    if total <= 0:
        return {sym: 0.0 for sym in daily_closes}

    return {sym: active.get(sym, 0.0) / total for sym in daily_closes}


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostico
# ─────────────────────────────────────────────────────────────────────────────
def describe_signal(daily_closes: dict) -> dict:
    """Diagnostico legivel do estado atual (endpoint /trend)."""
    btc_ok  = is_eligible(daily_closes.get(RISK_ON_REF))
    weights = compute_target_weights(daily_closes)
    out = {"risk_on": btc_ok, "assets": {}}
    for sym, close in daily_closes.items():
        sma200 = _sma(close, SMA_PERIOD)
        sma50  = _sma(close, SMA_FAST)
        price  = float(close.iloc[-1]) if close is not None and len(close) else None
        mom    = _momentum(close, MOM_PERIOD)
        out["assets"][sym] = {
            "eligible":       is_eligible(close),
            "state":          get_state(close),
            "target_weight":  round(weights.get(sym, 0.0), 4),
            "price":          round(price, 4) if price else None,
            "sma200":         round(sma200, 4) if sma200 else None,
            "sma50":          round(sma50, 4) if sma50 else None,
            "momentum_90d":   round(mom * 100, 2),
        }
    return out
