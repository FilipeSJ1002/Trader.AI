"""
backtest_compare.py — Comparação de configuracoes da trend_strategy.
Testa multiplos parametros e exibe um ranking final.

Uso: python backtest_compare.py
"""
import os
import importlib
import pandas as pd
import sys

sys.path.insert(0, os.path.dirname(__file__))

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT"]


def load_px(start="2022-01-01", end="2026-02-28"):
    closes = {}
    for s in SYMBOLS:
        path = os.path.join("data", f"{s}_1m.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        closes[s] = df['close'].resample('1D').last()
    px = pd.DataFrame(closes).loc[start:end].dropna()
    return px


def simulate(px, fn_weights, fee=0.001, max_invest=0.98):
    """Roda simulacao com a funcao fn_weights(hist_dict) -> pesos."""
    syms = list(px.columns)
    rets = px.pct_change().fillna(0)
    equity = 10_000.0
    weights_prev = {s: 0.0 for s in syms}
    eq_curve, n_rebal = [], 0

    for i in range(len(px)):
        hist   = {s: px[s].iloc[:i + 1] for s in syms}
        target = fn_weights(hist)
        target = {s: target.get(s, 0.0) * max_invest for s in syms}

        turnover = sum(abs(target[s] - weights_prev[s]) for s in syms)
        if turnover > 0.001:
            n_rebal += 1
            equity *= (1 - turnover * fee)

        if i + 1 < len(px):
            day_ret = sum(target[s] * rets[s].iloc[i + 1] for s in syms)
            equity *= (1 + day_ret)

        weights_prev = target
        eq_curve.append(equity)

    eq = pd.Series(eq_curve, index=px.index)
    final = eq.iloc[-1]
    ret   = (final / 10_000 - 1) * 100
    cm    = eq.cummax()
    dd    = ((eq - cm) / cm).min() * 100
    days  = (eq.index[-1] - eq.index[0]).days
    cagr  = ((final / 10_000) ** (365 / max(days, 1)) - 1) * 100

    by_year = {}
    for yr in sorted(set(eq.index.year)):
        e = eq[eq.index.year == yr]
        if len(e) > 2:
            r = (e.iloc[-1] / e.iloc[0] - 1) * 100
            d = ((e - e.cummax()) / e.cummax()).min() * 100
            by_year[yr] = (r, d)

    return {'ret': ret, 'cagr': cagr, 'dd': dd, 'n_rebal': n_rebal, 'by_year': by_year}


def make_strategy(sma_slow, sma_fast, mom_period, mom_bonus, top_n=None):
    """Fabrica de funcoes de pesos com parametros customizados."""
    PRIORITY = {"BTCUSDT": 2.0, "ETHUSDT": 2.0,
                "SOLUSDT": 1.0, "BNBUSDT": 1.0,
                "XRPUSDT": 1.0, "AVAXUSDT": 1.0}
    CORE = {"BTCUSDT", "ETHUSDT"}
    REF  = "BTCUSDT"

    def _sma(s, p):
        if s is None or len(s) < p:
            return None
        return float(s.iloc[-p:].mean())

    def _mom(s, p):
        if s is None or len(s) < p + 1:
            return 0.0
        return float(s.iloc[-1] / s.iloc[-p] - 1)

    def eligible(c):
        if c is None or len(c) < sma_slow:
            return False
        price = float(c.iloc[-1])
        s200  = _sma(c, sma_slow)
        s50   = _sma(c, sma_fast)
        if s200 is None:
            return False
        th = max(s200, s50) if s50 else s200
        return price > th

    def weights(hist):
        mask   = {sym: eligible(c) for sym, c in hist.items()}
        btc_ok = mask.get(REF, False)
        active = {}
        for sym, ok in mask.items():
            if not ok:
                continue
            if sym not in CORE and not btc_ok:
                continue
            m   = _mom(hist[sym], mom_period)
            b   = min(max(m, 0.0), 2.0) * mom_bonus
            active[sym] = PRIORITY.get(sym, 1.0) * (1.0 + b)

        if top_n and len(active) > top_n:
            ranked = sorted(active.items(), key=lambda x: x[1], reverse=True)
            active = dict(ranked[:top_n])

        total = sum(active.values())
        if total <= 0:
            return {sym: 0.0 for sym in hist}
        return {sym: active.get(sym, 0.0) / total for sym in hist}

    return weights


def print_row(label, r, first=False):
    if first:
        print(f"\n{'─'*88}")
        print(f"  {'Config':<36}  {'Total':>7}  {'CAGR':>6}  {'DD':>7}  "
              f"{'2022':>6}  {'2023':>7}  {'2024':>7}  {'2025':>7}")
        print(f"{'─'*88}")

    yrs = r.get('by_year', {})
    y22 = f"{yrs[2022][0]:+.0f}%" if 2022 in yrs else "   n/a"
    y23 = f"{yrs[2023][0]:+.0f}%" if 2023 in yrs else "   n/a"
    y24 = f"{yrs[2024][0]:+.0f}%" if 2024 in yrs else "   n/a"
    y25 = f"{yrs[2025][0]:+.0f}%" if 2025 in yrs else "   n/a"
    print(f"  {label:<36}  {r['ret']:>6.0f}%  {r['cagr']:>5.1f}%  {r['dd']:>6.1f}%  "
          f"  {y22}  {y23:>7}  {y24:>7}  {y25:>7}")


if __name__ == "__main__":
    print("Carregando dados...")
    px = load_px()
    print(f"Periodo: {px.index[0].date()} a {px.index[-1].date()} "
          f"| {len(px)} dias | {len(px.columns)} ativos\n")

    configs = [
        # label,                          sma_slow, sma_fast, mom_p, mom_b, top_n
        ("V6   SMA200",                     200,  50,  90, 0.00, None),
        ("V6.1 SMA50+SMA200",               200,  50,  90, 0.00, None),  # mom=0 = V6.1
        ("V7   SMA50+mom90d",               200,  50,  90, 0.50, None),
        ("V7   SMA50+mom90d top3",          200,  50,  90, 0.50,    3),
        ("V7b  SMA30+mom90d",               200,  30,  90, 0.50, None),
        ("V7c  SMA50+mom60d",               200,  50,  60, 0.50, None),
        ("V7d  SMA50+mom90d bonus1.0",      200,  50,  90, 1.00, None),
        ("V7e  SMA100+SMA30+mom90d",        100,  30,  90, 0.50, None),
        ("V7f  SMA150+SMA40+mom90d",        150,  40,  90, 0.50, None),
    ]

    # Nota: para V6 sem momentum, precisamos desligar o bonus
    # configs[0] ja tem mom_b=0 então pesos = PRIORITY (igual V6 original)
    # configs[1] igual mas sma_fast=50 com mom_b=0 (= V6.1)

    results = []
    for i, (label, slow, fast, mom_p, mom_b, top_n) in enumerate(configs):
        fn = make_strategy(slow, fast, mom_p, mom_b, top_n)
        r  = simulate(px, fn)
        r['label'] = label
        results.append(r)
        print_row(label, r, first=(i == 0))

    print(f"{'─'*88}")
    best = max(results, key=lambda x: x['cagr'])
    print(f"\n  MELHOR CAGR: {best['label']} → {best['cagr']:.1f}%/ano | "
          f"Retorno {best['ret']:.0f}% | DD {best['dd']:.1f}%")
