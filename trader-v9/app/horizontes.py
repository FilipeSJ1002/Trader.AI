# -*- coding: utf-8 -*-
"""
app/horizontes.py — e se o regime for mais lento que um dia?
=============================================================

A arquitetura assume que o Oraculo decide todo dia. Mas tendencia costuma ser
mais previsivel em prazo maior: o ruido de um dia se cancela em sete. Antes de
declarar que prever regime nao funciona, e obrigatorio testar se a pergunta
estava sendo feita no prazo errado.

O rotulo passa a ser a direcao dos PROXIMOS N dias, e as features continuam as
mesmas — lidas de barras fechadas ate a vespera.
"""
import sys
import time
from datetime import datetime, timedelta

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

import numpy as np
import polars as pl

from dados.fonte import FonteParquet, ler_config
from oraculo.features import colunas_de_feature, montar_tabela
from oraculo.modelo import avaliar, modelos, preparar

HORIZONTES = [1, 2, 3, 5, 7, 14, 30]
DE, ATE = datetime(2019, 6, 1), datetime(2026, 7, 25)


def rotulos_por_horizonte(historicos, dias, n: int) -> list:
    """Direcao media do universo nos proximos `n` dias."""
    fechamentos = {}
    for sym, h in historicos.items():
        ts, _, _ = h.barras_entre(0, len(h) - 1)
        fech = h.em(len(h) - 1).serie("fechamento")
        d = (pl.DataFrame({"ts": ts, "f": fech})
             .group_by_dynamic("ts", every="1d", closed="left", label="left")
             .agg(pl.col("f").last()).drop_nulls())
        fechamentos[sym] = dict(zip(
            [x.astype("datetime64[us]").astype(datetime).date()
             for x in d["ts"].to_numpy()], d["f"].to_list()))

    saida = []
    for dia in dias:
        d0 = dia.date()
        d1 = (dia + timedelta(days=n)).date()
        rets = []
        for sym, mapa in fechamentos.items():
            a, b = mapa.get(d0), mapa.get(d1)
            if a and b:
                rets.append(b / a - 1)
        saida.append(None if not rets else (1 if np.mean(rets) > 0 else 0))
    return saida


def main() -> None:
    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]
    print(f"Carregando {len(ativos)} ativos...", flush=True)
    h = FonteParquet(cfg=cfg).carregar(ativos)

    print("Montando features (uma vez — nao mudam com o horizonte)...",
          flush=True)
    tabela = montar_tabela(h, DE, ATE, referencia=ativos[0])
    colunas = colunas_de_feature(tabela)
    dias = list(tabela["dia"])
    print(f"  {len(tabela):,} dias x {len(colunas)} features\n", flush=True)

    print(f"  {'horizonte':<12} {'melhor modelo':<22} {'balanceada':>13} "
          f"{'piso ingenuo':>14} {'ganho':>9}")
    print("  " + "-" * 76, flush=True)

    melhores = []
    for n in HORIZONTES:
        y = rotulos_por_horizonte(h, dias, n)
        t = tabela.with_columns(pl.Series("y", y, dtype=pl.Int8))
        X, yv = preparar(t, colunas)
        if len(np.unique(yv)) < 2:
            continue

        avaliacoes = []
        for nome, m in modelos().items():
            a = avaliar(nome, m, X, yv)
            if a is not None:
                avaliacoes.append(a)
        if not avaliacoes:
            continue

        reais = [a for a in avaliacoes
                 if not a.nome.startswith(("sempre", "moeda"))]
        piso = max(a.balanceada for a in avaliacoes
                   if a.nome.startswith(("sempre", "moeda")))
        melhor = max(reais, key=lambda a: a.balanceada)
        melhores.append((n, melhor, piso))
        ganho = melhor.balanceada - piso
        print(f"  {n:>3} dia(s)    {melhor.nome:<22} "
              f"{melhor.balanceada*100:>10.2f}% ± {melhor.balanceada_erro*100:<3.2f}"
              f" {piso*100:>12.2f}% {ganho*100:>+8.2f}pp", flush=True)

    print("\n" + "=" * 78)
    print("  O REGIME E MAIS PREVISIVEL EM ALGUM PRAZO?")
    print("=" * 78)
    if not melhores:
        print("\n  Nada pode ser avaliado.")
        return

    campeao = max(melhores, key=lambda x: x[1].balanceada - x[2])
    n, a, piso = campeao
    margem = a.balanceada - 2 * a.balanceada_erro
    print(f"\n  Melhor prazo: {n} dia(s) — {a.nome} com "
          f"{a.balanceada*100:.2f}% ± {a.balanceada_erro*100:.2f}")
    print(f"  Piso ingenuo nesse prazo: {piso*100:.2f}%")
    print(f"  Limite inferior (media - 2 erros): {margem*100:.2f}%\n")

    if margem > 0.56:
        print("  ACHAMOS. Este prazo passa do alvo de 56%. A arquitetura")
        print("  sobrevive com um Oraculo mais lento — refazer a Sprint 1 com")
        print("  este horizonte para recalibrar o alvo.")
    elif margem > piso:
        print("  Ha sinal, mas pequeno: o modelo supera o ingenuo e nao chega")
        print("  aos 56%. Nao basta para justificar operar.")
    else:
        print("  NENHUM prazo e previsivel o bastante. Nao e o modelo, nao e o")
        print("  horizonte: a direcao agregada deste universo nao e previsivel")
        print("  com features de preco e volume.")
    print("=" * 78)


if __name__ == "__main__":
    main()
