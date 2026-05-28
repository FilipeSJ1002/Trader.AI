"""
trend_strategy.py — V6 (Pivot para Trend-Following)
Nucleo da estrategia vencedora, validada em 4 anos de dados:
  +219% em 4 anos | CAGR +32%/ano | Max DD -41% | 226 trades

Logica (provada superior ao scalping e ao buy-and-hold):
  • Sinal por ativo: comprado se preco > SMA200 diaria (segue a tendencia macro).
  • Filtro RISK-ON: alts (SOL/BNB/XRP/AVAX) so operam quando o BTC tambem esta
    acima da sua SMA200. Isso evita comprar repiques de alt em bear market —
    foi o que transformou +34% em +219%.
  • Prioridade: BTC e ETH recebem peso 2x (nucleo solido).
  • Alocacao: divide o capital entre os ativos elegiveis, proporcional ao peso.

Este modulo e a FONTE UNICA da logica de decisao — usado tanto pelo backtest
quanto pela producao, garantindo paridade total.
"""
import pandas as pd

# ── Configuracao ─────────────────────────────────────────────────────────────
SMA_PERIOD   = 200          # media movel de 200 dias (filtro de tendencia macro)
RISK_ON_REF  = "BTCUSDT"    # ativo de referencia para o filtro risk-on
CORE_ASSETS  = {"BTCUSDT", "ETHUSDT"}   # operam mesmo quando BTC esta em baixa

# Peso relativo por ativo (BTC/ETH priorizados = mais capital alocado a eles)
PRIORITY = {
    "BTCUSDT": 2.0, "ETHUSDT": 2.0,
    "SOLUSDT": 1.0, "BNBUSDT": 1.0,
    "XRPUSDT": 1.0, "AVAXUSDT": 1.0,
}


def is_uptrend(daily_close: pd.Series, period: int = SMA_PERIOD) -> bool:
    """True se o ultimo preco esta acima da SMA de `period` dias."""
    if daily_close is None or len(daily_close) < period:
        return False
    sma_now = float(daily_close.iloc[-period:].mean())
    price   = float(daily_close.iloc[-1])
    return price > sma_now


def compute_target_weights(daily_closes: dict) -> dict:
    """
    Recebe {symbol: Series de closes diarios (ordem cronologica)} e devolve
    {symbol: peso_alvo} (somando 1.0 entre os elegiveis, 0.0 para os de fora).

    Regras:
      1. So entra quem esta em uptrend (preco > SMA200).
      2. Risk-on: ativos fora do CORE so entram se o BTC tambem estiver em uptrend.
      3. Peso proporcional a PRIORITY; normalizado para somar 1.0.
    """
    above = {}
    for sym, close in daily_closes.items():
        above[sym] = is_uptrend(close)

    btc_up = above.get(RISK_ON_REF, False)

    eligible = {}
    for sym, up in above.items():
        if not up:
            continue
        # Filtro risk-on: alts so quando o BTC esta em tendencia de alta
        if sym in CORE_ASSETS or btc_up:
            eligible[sym] = PRIORITY.get(sym, 1.0)

    total = sum(eligible.values())
    if total <= 0:
        return {sym: 0.0 for sym in daily_closes}   # tudo em caixa (bear market)

    return {sym: eligible.get(sym, 0.0) / total for sym in daily_closes}


def describe_signal(daily_closes: dict) -> dict:
    """Diagnostico legivel do estado atual de cada ativo (para logs/endpoint)."""
    btc_up = is_uptrend(daily_closes.get(RISK_ON_REF))
    weights = compute_target_weights(daily_closes)
    out = {"risk_on": btc_up, "assets": {}}
    for sym, close in daily_closes.items():
        up = is_uptrend(close)
        out["assets"][sym] = {
            "uptrend":      up,
            "target_weight": round(weights.get(sym, 0.0), 4),
            "price":        round(float(close.iloc[-1]), 4) if close is not None and len(close) else None,
            "sma200":       round(float(close.iloc[-SMA_PERIOD:].mean()), 4)
                            if close is not None and len(close) >= SMA_PERIOD else None,
        }
    return out
