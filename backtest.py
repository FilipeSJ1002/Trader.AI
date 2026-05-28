"""
backtest.py — V6 (Trend-Following oficial)
Usa trend_strategy.compute_target_weights() — MESMA logica da producao.

Estrategia: SMA200 diaria + filtro risk-on, 6 ativos, BTC/ETH peso 2x.
Resultado validado (2022-2026): +219% | CAGR +32%/ano | Max DD -41%.
"""
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from trend_strategy import compute_target_weights, PRIORITY, SMA_PERIOD


def run_backtest(start="2022-01-01", end="2026-02-28", initial_balance=10000.0,
                 fee=0.001, max_invest=0.98):
    load_dotenv()
    syms = list(PRIORITY.keys())

    print("Carregando closes diarios dos 6 ativos...")
    closes = {}
    for s in syms:
        path = os.path.join("data", f"{s}_1m.parquet")
        if not os.path.exists(path):
            print(f"  [AVISO] {path} ausente — pulando {s}")
            continue
        df = pd.read_parquet(path)
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        closes[s] = df['close'].resample('1D').last()
    px = pd.DataFrame(closes).loc[start:end].dropna()
    rets = px.pct_change().fillna(0)
    syms = list(px.columns)
    print(f"Periodo: {px.index[0].date()} a {px.index[-1].date()} | {len(px)} dias | {len(syms)} ativos")

    # ── Simulacao diaria com rebalanceamento ────────────────────────────────
    equity = initial_balance
    weights_prev = {s: 0.0 for s in syms}
    eq_curve, weight_hist, n_rebal = [], [], 0

    idx = px.index
    for i in range(len(idx)):
        day = idx[i]
        # Sinais usando dados ATE hoje (sem lookahead)
        hist = {s: px[s].iloc[:i + 1] for s in syms}
        target = compute_target_weights(hist)
        target = {s: target.get(s, 0.0) * max_invest for s in syms}  # deixa caixa de seguranca

        # Custo de rebalanceamento (turnover * fee)
        turnover = sum(abs(target[s] - weights_prev[s]) for s in syms)
        if turnover > 0.001:
            n_rebal += 1
            equity *= (1 - turnover * fee)

        # Aplica retorno do dia seguinte (se houver) com os pesos de hoje
        if i + 1 < len(idx):
            day_ret = sum(target[s] * rets[s].iloc[i + 1] for s in syms)
            equity *= (1 + day_ret)

        weights_prev = target
        eq_curve.append({'time': day, 'equity': equity})
        weight_hist.append({'time': day, **target})

    eq = pd.DataFrame(eq_curve).set_index('time')
    final = eq['equity'].iloc[-1]
    total_ret = (final - initial_balance) / initial_balance * 100
    eq['cm'] = eq['equity'].cummax()
    max_dd = ((eq['equity'] - eq['cm']) / eq['cm']).min() * 100
    days = (eq.index[-1] - eq.index[0]).days
    cagr = ((final / initial_balance) ** (365 / max(days, 1)) - 1) * 100

    # Benchmark buy & hold (equal weight, rebalanceado)
    bh = (1 + rets.mean(axis=1)).cumprod() * initial_balance
    bh_ret = (bh.iloc[-1] - initial_balance) / initial_balance * 100
    bh_dd = ((bh - bh.cummax()) / bh.cummax()).min() * 100

    print("\n" + "=" * 60)
    print("RESULTADOS — TREND-FOLLOWING V6 (SMA200 + risk-on)")
    print("=" * 60)
    print(f"  Saldo Inicial:        ${initial_balance:>12,.2f}")
    print(f"  Saldo Final:          ${final:>12,.2f}")
    print(f"  Retorno Total:        {total_ret:>11.1f}%   ({days} dias)")
    print(f"  CAGR:                 {cagr:>11.1f}%/ano")
    print(f"  Max Drawdown:         {max_dd:>11.1f}%")
    print(f"  Rebalanceamentos:     {n_rebal:>11}")
    print("-" * 60)
    print(f"  [Benchmark BuyHold]   ret {bh_ret:+.1f}%  |  Max DD {bh_dd:.1f}%")
    print("-" * 60)
    # Retorno por ano
    print("  Retorno por ano:")
    for yr in sorted(set(eq.index.year)):
        e = eq[eq.index.year == yr]['equity']
        if len(e) > 2:
            r = (e.iloc[-1] / e.iloc[0] - 1) * 100
            d = ((e - e.cummax()) / e.cummax()).min() * 100
            print(f"    {yr}:  {r:+7.1f}%   (DD {d:+.0f}%)")
    print("=" * 60)

    # ── Grafico ──────────────────────────────────────────────────────────────
    print("Gerando relatorio visual...")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq['equity'], name='Trend-Following',
                             line=dict(color='lime', width=2)))
    fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name='Buy & Hold',
                             line=dict(color='gray', width=1, dash='dot')))
    fig.update_layout(
        title=f"Trend-Following V6 | Retorno {total_ret:.0f}% | CAGR {cagr:.0f}%/ano | Max DD {max_dd:.0f}%",
        xaxis_title="Tempo", yaxis_title="Equity (USD)", template="plotly_dark",
        hovermode="x unified", yaxis_type="log")
    report_path = os.path.join(os.path.dirname(__file__), "backtest_report.html")
    fig.write_html(report_path)
    print(f"[OK] Relatorio salvo: {report_path}")


if __name__ == "__main__":
    run_backtest()
