# -*- coding: utf-8 -*-
"""
v8_rank_honesto.py — ranking com margem de erro, o unico defensavel
====================================================================

Por que existe (25/08/2026)
--------------------------
Medido com v8_simulador: deslocar o inicio da simulacao em 7 MINUTOS muda o
resultado da config de producao de -4,90% para +8,24%, e o da variante sem
filtro de -9,75% para +33,68%. O desvio padrao entre rodadas que deveriam ser
identicas e de 5,3 a 16,5 pontos percentuais.

As diferencas entre configuracoes no ranking anterior cabiam em ~10 pontos.
O ruido e maior que o sinal: aquele ranking era ordenacao de sorte.

A causa e estrutural. Poucas operacoes (150 a 600), 3 posicoes simultaneas e
20% do saldo em cada fazem de cada operacao uma aposta grande. Sem vantagem
por aposta, o resultado depende de QUAIS operacoes calharam de entrar — e isso
muda com o relogio.

Este arquivo roda cada configuracao em varios deslocamentos de inicio e reporta
media +- erro padrao. Uma configuracao so merece atencao se a media ficar a
mais de 2 erros padrao de zero. Caso contrario, ela e indistinguivel de nada.
"""
import sys
import numpy as np
import pandas as pd

import v6_ciclo
from v8_simulador import preparar, simular, metricas
from v8_rank import PARAMETROS
from v5_data_prep import ASSETS, BTC, _load_parquet

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

DESLOCAMENTOS = [0, 1, 2, 3, 5, 7, 11, 13]      # minutos

CONFIGS = [
    ("PRODUCAO (config B)",   {}),
    ("A antiga 1%/0,5%/6h",   {"TP_PCT": 0.010, "SL_PCT": 0.005,
                               "MAX_HOLD_MIN": 360}),
    ("B sem alavancagem",     {"MAX_LEV": 1.0, "LEV_CURVE": "flat1"}),
    ("B sem filtro da rede",  {"DIRCONF_MIN": 0.0,
                               "LEV_CURVE": "regime_livre"}),
    ("B razao 1:1 (3%/3%)",   {"TP_PCT": 0.030, "SL_PCT": 0.030}),
]


def main():
    de, ate = pd.Timestamp("2021-01-01"), pd.Timestamp("2026-07-25")
    capital = 5000.0

    btc = _load_parquet(BTC)
    dados = {s: d for s in ASSETS if (d := preparar(s, btc)) is not None}
    orig = {k: getattr(v6_ciclo, k) for k in PARAMETROS}

    print(f"\n{len(CONFIGS)} configuracoes x {len(DESLOCAMENTOS)} deslocamentos"
          f" de inicio | {de:%Y-%m-%d} a {ate:%Y-%m-%d}\n")

    linhas = []
    for nome, mud in CONFIGS:
        for k in PARAMETROS:
            setattr(v6_ciclo, k, mud.get(k, orig[k]))
        rets, dds, ops = [], [], []
        print(f"  {nome:<24}", end="", flush=True)
        for d in DESLOCAMENTOS:
            try:
                m = metricas(simular(dados, de + pd.Timedelta(minutes=d), ate,
                                     capital, silencioso=True))
                rets.append(m["retorno"] * 100)
                dds.append(m["dd"] * 100)
                ops.append(m["n"])
                print(".", end="", flush=True)
            except Exception:
                print("x", end="", flush=True)
        for k in PARAMETROS:
            setattr(v6_ciclo, k, orig[k])
        if len(rets) < 3:
            print("  (poucas rodadas)")
            continue
        a = np.array(rets)
        media, erro = a.mean(), a.std(ddof=1) / np.sqrt(len(a))
        linhas.append({"nome": nome, "media": media, "erro": erro,
                       "t": media / erro if erro > 0 else 0.0,
                       "pior": a.min(), "melhor": a.max(),
                       "dd": float(np.mean(dds)), "ops": int(np.mean(ops))})
        print(f"  {media:+.1f}% +- {erro:.1f}", flush=True)

    linhas.sort(key=lambda x: x["media"], reverse=True)

    print("\n" + "=" * 96)
    print(f"  RANKING COM MARGEM DE ERRO — {de:%Y-%m-%d} a {ate:%Y-%m-%d}")
    print("=" * 96)
    print(f"\n  {'Configuracao':<24} {'media':>9} {'erro':>8} {'t':>7} "
          f"{'pior':>9} {'melhor':>9} {'queda':>8} {'ops':>6}")
    print("  " + "-" * 92)
    for m in linhas:
        marca = " *" if abs(m["t"]) >= 2 else ""
        print(f"  {m['nome']:<24} {m['media']:>+8.2f}% {m['erro']:>7.2f} "
              f"{m['t']:>+7.2f} {m['pior']:>+8.1f}% {m['melhor']:>+8.1f}% "
              f"{m['dd']:>7.1f}% {m['ops']:>6}{marca}")

    print("\n  " + "-" * 92)
    reais = [m for m in linhas if abs(m["t"]) >= 2]
    if reais:
        print(f"  Distinguiveis de zero (|t| >= 2): "
              + ", ".join(m["nome"] for m in reais))
    else:
        print("  NENHUMA configuracao e distinguivel de zero.")
        print("  Todas as diferencas do ranking anterior eram ruido de calendario.")
    print("\n  'pior' e 'melhor' sao rodadas da MESMA configuracao no MESMO")
    print("  periodo, mudando so o minuto de inicio. A distancia entre elas e")
    print("  o tamanho da sorte envolvida.")
    print("=" * 96)


if __name__ == "__main__":
    main()
