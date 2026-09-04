# -*- coding: utf-8 -*-
"""
app/confianca.py — o classificador acerta mais quando está confiante?
======================================================================

A pergunta que faltava (03/09/2026)
-----------------------------------
Todas as medições anteriores usaram um oráculo simulado que acerta 53,7% de
forma UNIFORME: cada dia tem a mesma chance de estar certo. Um classificador
real não funciona assim. Ele emite uma probabilidade, e há dias em que essa
probabilidade é 0,51 (não sabe de nada) e dias em que é 0,72 (viu alguma coisa).

Se a acurácia nos dias de alta confiança for substancialmente maior que a média,
existe uma estratégia que nunca testamos: **operar só nesses dias**. O limiar da
sobreposição é 58%; a média do classificador é 53,7%. Bastaria que o quintil
mais confiante chegasse a 58% para haver algo utilizável.

Se a acurácia for plana em relação à confiança, a probabilidade emitida não
carrega informação além do próprio palpite, e esta porta se fecha junto com as
outras.

O teste
-------
Validação walk-forward com embargo, idêntica à do app/treinar.py. Para cada
dobra, guarda-se a probabilidade prevista e o rótulo verdadeiro. Ao fim,
ordena-se tudo por confiança e mede-se a acurácia por faixa.
"""
from __future__ import annotations

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
from oraculo.features import colunas_de_feature, montar_tabela
from oraculo.modelo import dobras_temporais, modelos, preparar
from oraculo.teto import OraculoPerfeito
from nucleo.tipos import Regime

HORIZONTE = 3          # o unico com sinal solido (t = 3,40)
DE, ATE = datetime(2019, 6, 1), datetime(2026, 7, 25)
LIMIAR_SOBREPOSICAO = 0.58


def rotulos(historicos, dias, n: int):
    """Direcao media do universo nos proximos `n` dias."""
    from datetime import timedelta
    mapas = {}
    for sym, h in historicos.items():
        ts, _, _ = h.barras_entre(0, len(h) - 1)
        fech = h.em(len(h) - 1).serie("fechamento")
        d = (pl.DataFrame({"ts": ts, "f": fech})
             .group_by_dynamic("ts", every="1d", closed="left", label="left")
             .agg(pl.col("f").last()).drop_nulls())
        mapas[sym] = dict(zip(
            [x.astype("datetime64[us]").astype(datetime).date()
             for x in d["ts"].to_numpy()], d["f"].to_list()))

    saida = []
    for dia in dias:
        a_, b_ = dia.date(), (dia + timedelta(days=n)).date()
        rets = [m[b_] / m[a_] - 1 for m in mapas.values()
                if m.get(a_) and m.get(b_)]
        saida.append(None if not rets else (1 if np.mean(rets) > 0 else 0))
    return saida


def main() -> None:
    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]
    print(f"Carregando {len(ativos)} ativos...", flush=True)
    h = FonteParquet(cfg=cfg).carregar(ativos)

    print("Montando features...", flush=True)
    tabela = montar_tabela(h, DE, ATE, referencia=ativos[0])
    colunas = colunas_de_feature(tabela)
    tabela = tabela.with_columns(
        pl.Series("y", rotulos(h, list(tabela["dia"]), HORIZONTE), dtype=pl.Int8))
    X, y = preparar(tabela, colunas)
    print(f"  {len(y):,} dias | horizonte de {HORIZONTE} dias\n", flush=True)

    # Só os modelos que emitem probabilidade calibrável.
    candidatos = {k: v for k, v in modelos().items()
                  if k in ("regressao logistica", "floresta aleatoria",
                           "gradient boosting")}

    for nome, modelo in candidatos.items():
        probs, verdades = [], []
        for i_tr, i_te in dobras_temporais(len(y), 5):
            if len(np.unique(y[i_tr])) < 2:
                continue
            modelo.fit(X[i_tr], y[i_tr])
            p = modelo.predict_proba(X[i_te])[:, 1]
            probs.append(p)
            verdades.append(y[i_te])
        if not probs:
            continue
        p = np.concatenate(probs)
        v = np.concatenate(verdades)

        # Confianca = distancia de 0,5. Previsao = lado da probabilidade.
        conf = np.abs(p - 0.5)
        previu = (p > 0.5).astype(int)
        acertou = (previu == v)

        print(f"{nome} — {len(v):,} dias testados")
        print(f"  {'faixa de confiança':<22} {'dias':>6} {'acurácia':>10} "
              f"{'vs média':>10}")
        print("  " + "-" * 52)
        media = acertou.mean()
        ordem = np.argsort(-conf)
        for rotulo, frac in [("10% mais confiantes", 0.10),
                             ("20% mais confiantes", 0.20),
                             ("30% mais confiantes", 0.30),
                             ("50% mais confiantes", 0.50),
                             ("todos os dias", 1.00)]:
            k = max(int(len(v) * frac), 30)
            sel = ordem[:k]
            acc = acertou[sel].mean()
            marca = "  <-- passa" if acc > LIMIAR_SOBREPOSICAO else ""
            print(f"  {rotulo:<22} {k:>6} {acc*100:>9.2f}% "
                  f"{(acc-media)*100:>+9.2f}pp{marca}")
        print()

    print("=" * 60)
    print("  A CONFIANCA CARREGA INFORMACAO?")
    print("=" * 60)
    print(f"\n  Limiar da sobreposição: {LIMIAR_SOBREPOSICAO:.0%}.")
    print("  Se alguma faixa de alta confiança passar dele com folga, há uma")
    print("  estratégia utilizável: operar só nos dias em que o modelo está")
    print("  seguro e ficar comprado no resto do tempo.")
    print("\n  Se a acurácia for plana entre as faixas, a probabilidade emitida")
    print("  não carrega informação além do palpite e esta porta se fecha.")
    print("=" * 60)


if __name__ == "__main__":
    main()
