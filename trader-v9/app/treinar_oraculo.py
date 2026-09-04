# -*- coding: utf-8 -*-
"""
app/treinar_oraculo.py — treina e grava o classificador para uso ao vivo
========================================================================

Treina com TODO o histórico disponível e grava em modelos/oraculo_3d.joblib.
A avaliação honesta já foi feita em app/treinar.py, com walk-forward e embargo;
retreinar com tudo é o procedimento correto para colocar em operação.

Uso:  python -m app.treinar_oraculo
"""
import sys
import time
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

from dados.fonte import FonteParquet, ler_config
from oraculo.classificador import CAMINHO_PADRAO, OraculoTreinado, treinar


def main() -> None:
    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]
    print(f"Carregando {len(ativos)} ativos...", flush=True)
    t0 = time.time()
    h = FonteParquet(cfg=cfg).carregar(ativos)
    print(f"  {time.time()-t0:.1f}s")

    print("Treinando com todo o historico...", flush=True)
    t0 = time.time()
    r = treinar(h, referencia=ativos[0])
    print(f"  {r['dias']:,} dias x {r['colunas']} features em {time.time()-t0:.0f}s")
    print(f"  gravado em {r['caminho']}")

    o = OraculoTreinado(h, CAMINHO_PADRAO)
    ref = ativos[0]
    i = len(h[ref]) - 1
    v = h[ref].em(i)
    print(f"\n  Teste de leitura no ultimo dado ({v.ts:%Y-%m-%d}):")
    print(f"    regime         : {o.regime(v).value}")
    print(f"    prob. de alta  : {o.probabilidade(v):.3f}")
    print(f"\n  Lembrete gravado no modelo:")
    print(f"    acuracia medida fora da amostra : "
          f"{o.meta['acuracia_medida']:.2%} ± {o.meta['erro_medido']:.2%}")
    print(f"    limiar para bater comprar-e-manter: "
          f"{o.meta['limiar_necessario']:.0%}")
    print("    ou seja: ha sinal, e ele e MENOR que o necessario.")


if __name__ == "__main__":
    main()
