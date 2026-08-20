# -*- coding: utf-8 -*-
"""
v7_tendencia_diaria.py — a porta 2: seguir tendencia em prazo diario
=====================================================================

Por que testar isto (12/08/2026): a sessao de medicao provou que preco e
volume em candles de 1 minuto nao contem sinal suficiente para pagar 0,08%
de custo por operacao. Ver [[Limite de Edge (12-08-2026)]].

A saida obvia e mudar a ESCALA. Num movimento de 10%, a taxa de 0,08% e
arredondamento — o custo deixa de ser o gargalo. E seguir tendencia
(momentum) e a anomalia mais documentada e persistente em mercados, cripto
incluido.

O que este script mede, com a mesma honestidade da sessao anterior:

  - retorno da estrategia CONTRA comprar e segurar (o baseline que importa
    aqui — nao adianta ganhar dinheiro se o BTC parado ganhou mais)
  - rebaixamento maximo (a dor real de segurar a estrategia)
  - numero de operacoes e taxa paga
  - robustez: por ativo e por ano

Estrategias testadas:
  sma200      comprado acima da SMA200 diaria, fora abaixo  (long-only)
  cruz        comprado quando SMA50 > SMA200                 (long-only)
  sma200_ls   comprado acima, VENDIDO abaixo                 (long/short)

Uso:
  python v7_tendencia_diaria.py
  python v7_tendencia_diaria.py --assets all
"""
import sys
import glob
import os
import argparse
import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_s, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

from v5_data_prep import ASSETS, _load_parquet
from v5_backtest import FEE


def metricas(equity, dias_por_ano=365):
    """Retorno total, CAGR e rebaixamento maximo de uma curva de capital."""
    total = equity.iloc[-1] / equity.iloc[0] - 1
    anos = len(equity) / dias_por_ano
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / anos) - 1 if anos > 0 else 0.0
    dd = (equity / equity.cummax() - 1).min()
    return total, cagr, dd


def roda_estrategia(diario, modo):
    """
    Devolve (posicao, retorno_estrategia) alinhados ao indice diario.
    A posicao do dia D e decidida com dados ATE D e aplicada em D+1 —
    sem olhar o futuro.
    """
    sma200 = diario.rolling(200).mean()
    sma50 = diario.rolling(50).mean()

    if modo == "sma200":
        pos = (diario > sma200).astype(float)
    elif modo == "cruz":
        pos = (sma50 > sma200).astype(float)
    elif modo == "sma200_ls":
        pos = np.sign(diario - sma200).astype(float)
    else:
        raise SystemExit(f"modo desconhecido: {modo}")

    pos = pos.fillna(0.0)
    ret = diario.pct_change().fillna(0.0)
    pos_ontem = pos.shift(1).fillna(0.0)          # decide hoje, opera amanha

    # Taxa cobrada sobre a MUDANCA de posicao (0 -> 1 paga uma perna)
    giro = pos_ontem.diff().abs().fillna(0.0)
    custo = giro * FEE

    return pos_ontem, pos_ontem * ret - custo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="6", help="'6' ou 'all'")
    ap.add_argument("--from", dest="dt_from", default="2019-01-01")
    ap.add_argument("--to", dest="dt_to", default="2026-12-31")
    a = ap.parse_args()

    if a.assets.lower() == "all":
        universo = sorted(os.path.basename(p).replace("_1m.parquet", "")
                          for p in glob.glob("data/*_1m.parquet"))
    else:
        universo = list(ASSETS)

    modos = ["sma200", "cruz", "sma200_ls"]
    print(f"\nTendencia diaria | {len(universo)} ativos | {a.dt_from} -> {a.dt_to}")
    print(f"Taxa: {FEE*100:.2f}% por perna\n")

    resultados = {m: {} for m in modos}
    buyhold = {}
    curvas = {m: [] for m in modos}
    curva_bh = []

    for sym in universo:
        df = _load_parquet(sym)
        diario = df["close"].resample("1D").last().dropna()
        diario = diario[(diario.index >= a.dt_from) & (diario.index <= a.dt_to)]
        if len(diario) < 400:
            print(f"  {sym}: historico curto, pulando")
            continue

        ret_bh = diario.pct_change().fillna(0.0)
        eq_bh = (1 + ret_bh).cumprod()
        buyhold[sym] = metricas(eq_bh)
        curva_bh.append(eq_bh.rename(sym))

        for m in modos:
            _pos, ret_s = roda_estrategia(diario, m)
            eq = (1 + ret_s).cumprod()
            resultados[m][sym] = metricas(eq) + (
                int((_pos.diff().abs() > 0).sum()),)   # n de mudancas de posicao
            curvas[m].append(eq.rename(sym))

        print(f"  {sym} ok", flush=True)

    if not buyhold:
        print("Sem dados suficientes.")
        return

    # ── Tabela por ativo ────────────────────────────────────────────────────
    for m in modos:
        print(f"\n{'='*86}")
        print(f"  ESTRATEGIA '{m}'")
        print(f"{'='*86}")
        print(f"  {'Ativo':<10} {'Retorno':>10} {'CAGR':>9} {'DrawDown':>10} "
              f"{'ops':>5} | {'B&H retorno':>12} {'B&H DD':>9} {'venceu?':>8}")
        print("  " + "-" * 84)
        venceu = 0
        for sym in sorted(resultados[m]):
            tot, cagr, dd, n = resultados[m][sym]
            b_tot, b_cagr, b_dd = buyhold[sym]
            ok = tot > b_tot
            venceu += int(ok)
            print(f"  {sym:<10} {tot*100:>+9.1f}% {cagr*100:>+8.1f}% {dd*100:>9.1f}% "
                  f"{n:>5} | {b_tot*100:>+11.1f}% {b_dd*100:>8.1f}% "
                  f"{'SIM' if ok else 'nao':>8}")
        print(f"  -> venceu buy-and-hold em {venceu} de {len(resultados[m])} ativos")

        # Carteira igualmente ponderada
        port = pd.concat(curvas[m], axis=1).ffill().dropna()
        port_eq = (port / port.iloc[0]).mean(axis=1)
        bh = pd.concat(curva_bh, axis=1).ffill().dropna()
        bh_eq = (bh / bh.iloc[0]).mean(axis=1)
        t1, c1, d1 = metricas(port_eq)
        t2, c2, d2 = metricas(bh_eq)
        print(f"\n  CARTEIRA (peso igual, {len(curvas[m])} ativos)")
        print(f"    Estrategia   : {t1*100:>+8.1f}%  | CAGR {c1*100:>+6.1f}%  "
              f"| DD max {d1*100:>6.1f}%")
        print(f"    Buy and hold : {t2*100:>+8.1f}%  | CAGR {c2*100:>+6.1f}%  "
              f"| DD max {d2*100:>6.1f}%")

        # Robustez por ano
        ret_port = port_eq.pct_change().fillna(0.0)
        ret_bh_p = bh_eq.pct_change().fillna(0.0)
        print(f"\n    {'Ano':<6} {'Estrategia':>12} {'Buy&Hold':>12} {'diferenca':>12}")
        print("    " + "-" * 44)
        pos_anos = 0
        anos = sorted(set(ret_port.index.year))
        for ano in anos:
            e = (1 + ret_port[ret_port.index.year == ano]).prod() - 1
            b = (1 + ret_bh_p[ret_bh_p.index.year == ano]).prod() - 1
            if e > b:
                pos_anos += 1
            print(f"    {ano:<6} {e*100:>+11.1f}% {b*100:>+11.1f}% "
                  f"{(e-b)*100:>+11.1f}pp")
        print(f"    -> bateu o buy-and-hold em {pos_anos} de {len(anos)} anos")

    print(f"\n{'='*86}")
    print("  COMO LER")
    print("    O baseline aqui e COMPRAR E SEGURAR, nao o passeio aleatorio:")
    print("    a estrategia so vale a pena se entregar mais retorno ou MUITO")
    print("    menos rebaixamento. Vencer em poucos ativos ou poucos anos = sorte.")
    print(f"{'='*86}")


if __name__ == "__main__":
    main()
