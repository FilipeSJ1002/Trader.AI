"""
backtest.py — V7 (Three-State Momentum Trend)
Usa trend_strategy.compute_target_weights() — MESMA logica da producao.

Estados: BULL (100% + momentum bonus) | CAUTION (50% so core) | BEAR (caixa)
Momentum rotation: mais capital nos ativos mais fortes no bull.

Uso:
  python backtest.py              -> resultados anuais 2022-2026
  python backtest.py 2023         -> breakdown MENSAL do ano 2023
  python backtest.py 2024         -> breakdown MENSAL do ano 2024
"""
import os
import sys
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
    print(f"{'-' * 62}")
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


MONTHS_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Marco",    4: "Abril",
    5: "Maio",    6: "Junho",     7: "Julho",     8: "Agosto",
    9: "Setembro",10: "Outubro", 11: "Novembro", 12: "Dezembro",
}


def run_monthly(year: int, fee: float = 0.001, max_invest: float = 0.98):
    """
    Exibe o breakdown mensal do ano escolhido:
    retorno mensal, DD do mes, estado medio do BTC e ativos top por momentum.
    """
    load_dotenv()
    syms = list(PRIORITY.keys())

    # Carrega dados com 200 dias de warmup antes do ano escolhido
    warmup_start = f"{year - 1}-01-01"   # 1 ano antes garante SMA200 pronto
    end          = f"{year}-12-31"

    closes = {}
    for s in syms:
        path = os.path.join("data", f"{s}_1m.parquet")
        if not os.path.exists(path):
            continue
        df = pd.read_parquet(path)
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        closes[s] = df['close'].resample('1D').last()

    px_full = pd.DataFrame(closes).loc[warmup_start:end].dropna()
    if px_full.empty:
        print(f"[ERRO] Sem dados para {year}. Verifique os arquivos em data/")
        return

    rets_full = px_full.pct_change().fillna(0)

    # Simula o ano inteiro (com warmup para SMA200)
    eq_series, _ = _simulate(px_full, 10_000.0, fee, max_invest)

    # Filtra apenas o ano solicitado
    eq_year = eq_series[eq_series.index.year == year]
    if eq_year.empty:
        print(f"[ERRO] Ano {year} fora do intervalo de dados disponivel.")
        return

    # Preco inicial do ano (ultimo dia do ano anterior ou primeiro do ano)
    eq_start_idx = eq_series.index.searchsorted(eq_year.index[0])
    eq_start_val = eq_series['equity'].iloc[max(0, eq_start_idx - 1)]

    print(f"\n{'=' * 70}")
    print(f"  BACKTEST MENSAL — {year}   (Trader.AI V7)")
    print(f"{'=' * 70}")
    print(f"  {'Mes':<12}  {'Retorno':>8}  {'Acumulado':>10}  {'DD Mes':>8}  {'Estado BTC':>11}  {'Lider momentum'}")
    print(f"{'-' * 70}")

    accumulated = eq_start_val
    from trend_strategy import get_state, _rel_momentum, _sma, SMA_PERIOD, SMA_FAST, _btc_in_bull

    for month in range(1, 13):
        eq_m = eq_year[eq_year.index.month == month]
        if eq_m.empty:
            print(f"  {MONTHS_PT[month]:<12}  {'sem dados':>8}")
            continue

        # Retorno e drawdown do mes
        eq_m_vals   = eq_m['equity']
        start_val   = accumulated
        end_val     = eq_m_vals.iloc[-1]
        month_ret   = (end_val / start_val - 1) * 100
        month_dd    = ((eq_m_vals - eq_m_vals.cummax()) / eq_m_vals.cummax()).min() * 100
        accumulated = end_val
        acum_pct    = (accumulated / eq_start_val - 1) * 100

        # Ultimo dia do mes: estado do BTC e lider de momentum relativo
        last_day_idx = px_full.index.searchsorted(eq_m.index[-1])
        hist_at_end  = {s: px_full[s].iloc[:last_day_idx + 1] for s in syms}

        btc_hist     = hist_at_end.get("BTCUSDT")
        btc_bull     = _btc_in_bull(btc_hist)
        btc_state    = "BULL" if btc_bull else "BEAR"
        rel_scores   = {s: _rel_momentum(hist_at_end[s], btc_hist, 21)
                        for s in syms if s != "BTCUSDT"}
        top_sym      = max(rel_scores, key=lambda k: rel_scores[k])
        top_mom_pct  = rel_scores[top_sym] * 100

        # Barra visual de retorno
        bar_len = min(abs(int(month_ret / 3)), 12)
        bar     = ("+" if month_ret >= 0 else "-") + "#" * bar_len

        print(f"  {MONTHS_PT[month]:<12}  {month_ret:>+7.2f}%  {acum_pct:>+9.2f}%  "
              f"{month_dd:>7.1f}%  {btc_state:>11}  "
              f"{top_sym} {top_mom_pct:+.0f}%  {bar}")

    # Resumo do ano
    year_ret  = (accumulated / eq_start_val - 1) * 100
    # Benchmark BTC no ano
    btc_year  = px_full["BTCUSDT"][px_full.index.year == year]
    btc_ret   = (btc_year.iloc[-1] / btc_year.iloc[0] - 1) * 100 if len(btc_year) > 1 else 0

    print(f"{'-' * 70}")
    print(f"  {'ANO ' + str(year):<12}  {year_ret:>+7.2f}%")
    print(f"  {'BTC hold':<12}  {btc_ret:>+7.2f}%  (benchmark)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        try:
            year_arg = int(sys.argv[1])
            run_monthly(year_arg)
        except ValueError:
            print(f"[ERRO] Argumento invalido: '{sys.argv[1]}'. Use um ano, ex: python backtest.py 2023")
    else:
        run_backtest()
