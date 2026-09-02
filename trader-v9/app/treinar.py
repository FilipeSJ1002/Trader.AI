# -*- coding: utf-8 -*-
"""
app/treinar.py — a Sprint 2: o classificador chega aos 56%?
============================================================

Monta as features, treina os modelos e mede a acuracia fora da amostra com
validacao walk-forward e embargo. NAO liga em operacao nenhuma: se o numero
nao passar, nao ha o que conectar.

O limiar foi fixado na Sprint 1, antes de existir modelo:
  ~50%  para nao perder dinheiro
  ~56%  para bater comprar-e-segurar   <- o que vale
"""
import sys
import time
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

import numpy as np
import polars as pl

from dados.fonte import FonteParquet, ler_config
from nucleo.tipos import Regime
from oraculo.features import colunas_de_feature, montar_tabela
from oraculo.modelo import avaliar, modelos, preparar
from oraculo.teto import OraculoPerfeito

ALVO_EMPATE = 0.50
ALVO_BATER_BH = 0.56
DE, ATE = datetime(2019, 6, 1), datetime(2026, 7, 25)


def main() -> None:
    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]

    print(f"Carregando {len(ativos)} ativos...", flush=True)
    t0 = time.time()
    h = FonteParquet(cfg=cfg).carregar(ativos)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    print("Montando features diarias (barras fechadas apenas)...", flush=True)
    t0 = time.time()
    tabela = montar_tabela(h, DE, ATE, referencia=ativos[0])
    print(f"  {len(tabela):,} dias x {len(colunas_de_feature(tabela))} features"
          f" em {time.time()-t0:.0f}s", flush=True)

    # Rotulo: o que o oraculo perfeito sabia de cada dia.
    mapa = OraculoPerfeito(h)._mapa
    y = [(1 if mapa.get(d.date()) is Regime.BULL
          else 0 if mapa.get(d.date()) is Regime.BEAR else None)
         for d in tabela["dia"]]
    tabela = tabela.with_columns(pl.Series("y", y, dtype=pl.Int8))

    colunas = colunas_de_feature(tabela)
    X, yv = preparar(tabela, colunas)
    base = max(yv.mean(), 1 - yv.mean())
    print(f"\n  {len(yv):,} dias rotulados | {yv.mean()*100:.1f}% de alta")
    print(f"  Piso ingenuo (responder sempre a classe maior): {base*100:.1f}%")
    print(f"  ALVO fixado na Sprint 1: {ALVO_BATER_BH*100:.0f}% "
          f"(balanceada)\n", flush=True)

    print(f"  {'Modelo':<24} {'balanceada':>12} {'crua':>12} "
          f"{'acerta alta':>12} {'acerta baixa':>13} {'diz alta':>10}")
    print("  " + "-" * 88, flush=True)

    resultados = []
    for nome, m in modelos().items():
        t0 = time.time()
        a = avaliar(nome, m, X, yv)
        if a is None:
            print(f"  {nome:<24} (dobras insuficientes)")
            continue
        resultados.append(a)
        print(f"  {nome:<24} {a.balanceada*100:>10.2f}% ± {a.balanceada_erro*100:<4.2f}"
              f" {a.acuracia*100:>10.2f}% {a.acerto_bull*100:>11.1f}% "
              f"{a.acerto_bear*100:>12.1f}% {a.fracao_bull_previsto*100:>9.1f}%"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    _veredito(resultados)


def _veredito(resultados) -> None:
    print("\n" + "=" * 92)
    print("  VEREDITO DA SPRINT 2")
    print("=" * 92)
    if not resultados:
        print("\n  Nenhum modelo pode ser avaliado.")
        return

    reais = [r for r in resultados if not r.nome.startswith(("sempre", "moeda"))]
    melhor = max(reais, key=lambda r: r.balanceada)
    dummies = [r for r in resultados if r.nome.startswith(("sempre", "moeda"))]
    piso = max(d.balanceada for d in dummies)

    print(f"\n  Melhor modelo real : {melhor.nome}")
    print(f"    balanceada       : {melhor.balanceada*100:.2f}% "
          f"± {melhor.balanceada_erro*100:.2f}")
    print(f"    piso dos ingenuos: {piso*100:.2f}%")
    print(f"    alvo da Sprint 1 : {ALVO_BATER_BH*100:.0f}%")
    print(f"    dobras           : {melhor.dobras} | dias testados: "
          f"{melhor.n_teste:,}")

    supera_piso = (melhor.balanceada - 2 * melhor.balanceada_erro) > piso
    bate_alvo = (melhor.balanceada - 2 * melhor.balanceada_erro) > ALVO_BATER_BH
    passa_empate = (melhor.balanceada - 2 * melhor.balanceada_erro) > ALVO_EMPATE

    print()
    for desc, ok in [("supera os modelos ingenuos", supera_piso),
                     (f"supera {ALVO_EMPATE*100:.0f}% (nao perder dinheiro)",
                      passa_empate),
                     (f"supera {ALVO_BATER_BH*100:.0f}% (bater comprar-e-segurar)",
                      bate_alvo)]:
        print(f"    [{'SIM' if ok else 'NAO'}] {desc}")

    print()
    if bate_alvo:
        print("  CONECTAR. O classificador passa do alvo com margem. O proximo")
        print("  passo e liga-lo ao replay e medir com curva de capital.")
    elif passa_empate and supera_piso:
        print("  ZONA CINZENTA. O modelo sabe algo — supera os ingenuos e o")
        print("  ponto de empate — mas nao chega ao alvo que justifica o")
        print("  sistema contra simplesmente segurar os ativos. Ligar isto em")
        print("  producao seria trabalhar de graca, com risco.")
    else:
        print("  NAO CONECTAR. O classificador nao se distingue do ingenuo.")
        print("  A direcao diaria do mercado, com estas features, nao e")
        print("  previsivel o bastante. O problema nao e o modelo — e a")
        print("  pergunta. Ver o documento de arquitetura.")
    print("=" * 92)


if __name__ == "__main__":
    main()
