# -*- coding: utf-8 -*-
"""
app/janela_de_treino.py — quanto histórico o classificador precisa?
====================================================================

Motivo (04/09/2026): o servidor semeou 260 dias e treinou com 257, enquanto a
validação original usou 2.129. A pergunta não é se isso "parece pouco" — e sim
quanto de acurácia se perde, medido.

O teste: mesma validação walk-forward com embargo, variando apenas o TAMANHO
da janela de treino que precede cada dobra de teste.
"""
import sys
from datetime import datetime, timedelta

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

import numpy as np
import polars as pl

from dados.fonte import FonteParquet, ler_config
from oraculo.classificador import NOME_MODELO, _rotulos
from oraculo.features import colunas_de_feature, montar_tabela
from oraculo.modelo import EMBARGO_DIAS, modelos, preparar

JANELAS = [257, 500, 1000, 1500, None]     # None = tudo que houver
HORIZONTE = 3
DE, ATE = datetime(2019, 6, 1), datetime(2026, 7, 25)


def main() -> None:
    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]
    print(f"Carregando {len(ativos)} ativos...", flush=True)
    h = FonteParquet(cfg=cfg).carregar(ativos)

    tabela = montar_tabela(h, DE, ATE, referencia=ativos[0])
    colunas = colunas_de_feature(tabela)
    tabela = tabela.with_columns(pl.Series(
        "y", _rotulos(h, list(tabela["dia"]), HORIZONTE), dtype=pl.Int8))
    X, y = preparar(tabela, colunas)
    print(f"  {len(y):,} dias disponiveis\n", flush=True)

    print(f"  {'janela de treino':<20} {'balanceada':>12} {'dobras':>8}")
    print("  " + "-" * 44, flush=True)

    n = len(y)
    tamanho_teste = n // 6
    for janela in JANELAS:
        bals = []
        for k in range(5):
            fim_treino = tamanho_teste * (k + 1)
            ini_treino = 0 if janela is None else max(0, fim_treino - janela)
            ini_teste = fim_treino + EMBARGO_DIAS
            fim_teste = min(ini_teste + tamanho_teste, n)
            if fim_teste - ini_teste < 60 or fim_treino - ini_treino < 60:
                continue
            tr = np.arange(ini_treino, fim_treino)
            te = np.arange(ini_teste, fim_teste)
            if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
                continue
            m = modelos()[NOME_MODELO]
            m.fit(X[tr], y[tr])
            p = m.predict(X[te])
            bull = (p[y[te] == 1] == 1).mean()
            bear = (p[y[te] == 0] == 0).mean()
            bals.append((bull + bear) / 2)

        if len(bals) < 2:
            print(f"  {str(janela or 'tudo'):<20} (dobras insuficientes)")
            continue
        a = np.array(bals)
        erro = a.std(ddof=1) / np.sqrt(len(a))
        rotulo = f"{janela} dias" if janela else "tudo (~1.780)"
        print(f"  {rotulo:<20} {a.mean()*100:>10.2f}% ± {erro*100:<4.2f} "
              f"{len(bals):>6}", flush=True)

    print("\n" + "=" * 52)
    print("  Se a janela curta perder acuracia, o servidor PRECISA de mais")
    print("  historico. Se nao perder, 260 dias bastam e o deploy segue.")
    print("=" * 52)


if __name__ == "__main__":
    main()
