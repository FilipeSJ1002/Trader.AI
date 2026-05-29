"""
backtest.py — V7 (Three-State Momentum Trend)
Usa trend_strategy.compute_target_weights() — MESMA logica da producao.

Estados: BULL (100% + momentum bonus) | CAUTION (50% so core) | BEAR (caixa)
Momentum rotation: mais capital nos ativos mais fortes no bull.
"""
import os
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv
from trend_strategy import compute_target_weights, PRIORITY, SMA_PERIOD, SMA_FAST


def _simulate(px: pd.DataFrame, initial_balance: float,
              fee: float, max_invest: float) -> tuple[pd.DataFrame, int]:
    """Roda simulacao diaria. Retorna (equity_curve_df, n_rebalances)."""
    syms = list(px.columns)
    rets = px.pct_change().fillna(0)
    equity = initial_balance
    weights_prev = {s: 0.0 for s in syms}
    eq_curve, n_rebal = [], 0

    for i in range(len(px)):
        hist   = {s: px[s].iloc[:i + 1] for s in syms}
        target = compute_target_weights(hist)
        target = {s: target.get(s, 0.0) * max_invest for s in syms}

        turnover = sum(abs(target[s] - weights_prev[s]) for s in syms)
        if turnover > 0.001:
            n_rebal += 1
            equity *= (1 - turnover * fee)

        if i + 1 < len(px):
            day_ret = sum(target[s] * rets[s].iloc[i + 1] for s in syms)
            equity *= (1 + day_ret)

        weights_prev = target
        eq_curve.append({'time': px.index[i], 'equity': equity})

    return pd.DataFrame(eq_curve).set_index('time'), n_rebal


def _stats(eq: pd.DataFrame, initial_balance: float):
    """Calcula metricas de performance."""
    final     = eq['equity'].iloc[-1]
    total_ret = (final - initial_balance) / initial_balance * 100
    eq2       = eq.copy()
    eq2['cm'] = eq2['equity'].cummax()
    max_dd    = ((eq2['equity'] - eq2['cm']) / eq2['cm']).min() * 100
    days      = (eq.index[-1] - eq.index[0]).days
    cagr      = ((final / initial_balance) ** (365 / max(days, 1)) - 1) * 100
    return {'final': final, 'ret': total_ret, 'cagr': cagr, 'dd': max_dd, 'days': days}


def _print_results(label: str, eq: pd.DataFrame, initial_balance: float, n_rebal: int):
    s = _stats(eq, initial_balance)
    print(f"\n{'=' * 62}")
    print(f"  {label}")
    print(f"{'=' * 62}")
    print(f"  Saldo Inicial :  ${initial_balance:>12,.2f}")
    print(f"  Saldo Final   :  ${s['final']:>12,.2f}")
    print(f"  Retorno Total :  {s['ret']:>10.1f}%   ({s['days']} dias)")
    print(f"  CAGR          :  {s['cagr']:>10.1f}% / ano")
    print(f"  Max Drawdown  :  {s['dd']:>10.1f}%")
    print(f"  Rebalances    :  {n_rebal:>10}")
    print(f"{'─' * 62}")
    print("  Retorno por ano:")
    eq2 = eq.copy()
    eq2['cm'] = eq2['equity'].cummax()
    for yr in sorted(set(eq.index.year)):
        e = eq[eq.index.year == yr]['equity']
        if len(e) > 2:
            r = (e.iloc[-1] / e.iloc[0] - 1) * 100
            d = ((e - e.cummax()) / e.cummax()).min() * 100
            print(f"    {yr}: {r:+8.1f}%   (DD {d:+.0f}%)")
    print(f"{'=' * 62}")
    return s


def run_backtest(start="2022-01-01", end="2026-02-28", initial_balance=10_000.0,
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

    px   = pd.DataFrame(closes).loc[start:end].dropna()
    rets = px.pct_change().fillna(0)
    print(f"Periodo: {px.index[0].date()} a {px.index[-1].date()} "
          f"| {len(px)} dias | {len(px.columns)} ativos")

    # ── V7 (estrategia ativa) ────────────────────────────────────────────────
    eq_v7, n7 = _simulate(px, initial_balance, fee, max_invest)
    sv7 = _print_results("V7 — Three-State + Momentum Rotation", eq_v7, initial_balance, n7)

    # ── Benchmark buy & hold (equal weight) ─────────────────────────────────
    bh     = (1 + rets.mean(axis=1)).cumprod() * initial_balance
    bh_ret = (bh.iloc[-1] - initial_balance) / initial_balance * 100
    bh_dd  = ((bh - bh.cummax()) / bh.cummax()).min() * 100
    print(f"\n  [Benchmark Buy & Hold]  ret {bh_ret:+.1f}%  |  Max DD {bh_dd:.1f}%")

    # ── Grafico ──────────────────────────────────────────────────────────────
    print("\nGerando relatorio visual...")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=eq_v7.index, y=eq_v7['equity'],
        name='V7 Three-State + Momentum',
        line=dict(color='#00ff88', width=2.5)))

    fig.add_trace(go.Scatter(
        x=bh.index, y=bh.values,
        name='Buy & Hold (equal weight)',
        line=dict(color='#888888', width=1.2, dash='dot')))

    fig.update_layout(
        title=(f"Trader.AI V7 | Retorno {sv7['ret']:.0f}% | "
               f"CAGR {sv7['cagr']:.0f}%/ano | Max DD {sv7['dd']:.0f}%"),
        xaxis_title="Data", yaxis_title="Portfolio (USD)",
        template="plotly_dark", hovermode="x unified",
        yaxis_type="log", legend=dict(x=0.01, y=0.99),
        annotations=[dict(
            text=("Estrategia: SMA50/SMA200 Three-State + Momentum Rotation<br>"
                  "BULL: 100% investido + bonus momentum | "
                  "CAUTION: 50% so BTC/ETH | BEAR: 100% caixa"),
            xref="paper", yref="paper", x=0.01, y=0.01,
            showarrow=False, font=dict(size=10, color="#aaaaaa"),
            align="left"
        )]
    )

    report_path = os.path.join(os.path.dirname(__file__), "backtest_report.html")
    fig.write_html(report_path)
    print(f"[OK] Relatorio salvo: {report_path}")
    return sv7


if __name__ == "__main__":
    run_backtest()
