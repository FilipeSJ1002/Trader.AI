# -*- coding: utf-8 -*-
"""
app/alavancagem.py — "quanto maior o risco, maior o ganho" e verdade?
======================================================================

A frase e verdadeira quando ha vantagem positiva por aposta: aí a alavancagem
multiplica um numero positivo. Quando a vantagem e nula ou negativa, ela
multiplica um numero nulo ou negativo — e, pior, introduz o ARRASTO DE
VOLATILIDADE: perder 50% exige ganhar 100% para voltar ao ponto de partida,
entao oscilar mais reduz o crescimento composto mesmo com a media intacta.

Este arquivo mede a curva. Usa a configuracao REALISTA — oraculo de 3 dias com
os 53,7% de acuracia que o classificador de fato alcancou — e varia so a
alavancagem, com varias sementes e fases para cada nivel.
"""
import sys
import time
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

import numpy as np

from avaliacao.metricas import comprar_e_segurar, medir
from avaliacao.replay import replay
from dados.fonte import FonteParquet, ler_config
from execucao.papel import CorretoraPapel, SemSaldo
from execucao.risco import Risco
from motores.bear import MotorBear
from motores.bull import MotorBull
from nucleo.tipos import Regime
from oraculo.ruidoso import OraculoLento

NIVEIS = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
FRACOES = [0.20]                # so a fracao do artigo, para reverificar
SEMENTES = [1, 2, 3]
FASES = [0, 7]
ACURACIA = 0.537                # o que o classificador de fato alcancou
DE, ATE = datetime(2023, 1, 1), datetime(2026, 7, 25)
CAPITAL = 5000.0


def main() -> None:
    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]
    print(f"Carregando {len(ativos)} ativos...", flush=True)
    h = FonteParquet(cfg=cfg).carregar(ativos)
    ref = comprar_e_segurar(h, DE, ATE, CAPITAL)

    total = len(NIVEIS) * len(FRACOES) * len(SEMENTES) * len(FASES)
    print(f"Oraculo de 3 dias com {ACURACIA:.1%} de acuracia — o que temos.")
    print(f"{DE:%Y-%m-%d} a {ATE:%Y-%m-%d} | {total} rodadas")
    print(f"Comprar e segurar no periodo: {ref.retorno*100:+.1f}%\n", flush=True)

    print(f"  {'aposta':<9} {'lev':>5} {'media':>10} {'erro':>8} "
          f"{'pior':>10} {'melhor':>10} {'queda':>8} {'quebrou':>9}")
    print("  " + "-" * 74, flush=True)

    for fracao in FRACOES:
        for lev in NIVEIS:
            rets, quedas, quebras = [], [], 0
            for semente in SEMENTES:
                for fase in FASES:
                    c = CorretoraPapel(CAPITAL)
                    try:
                        r = replay(
                            h,
                            {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
                            OraculoLento(h, passo=3, acuracia=ACURACIA,
                                         semente=semente),
                            c, Risco(fracao_por_operacao=fracao,
                                     alavancagem=lev),
                            DE, ATE, a_cada=15, referencia=ativos[0], fase=fase,
                        )
                        m = medir(r)
                        rets.append(m.retorno)
                        quedas.append(m.rebaixamento)
                        if "zerada" in r.parou_por:
                            quebras += 1
                    except SemSaldo:
                        rets.append(-1.0)
                        quedas.append(-1.0)
                        quebras += 1

            a = np.array(rets)
            erro = (float(a.std(ddof=1) / np.sqrt(len(a)))
                    if len(a) > 1 else 0.0)
            print(f"  {fracao:>7.0%}   {lev:>4.0f}x {a.mean()*100:>+9.1f}% "
                  f"{erro*100:>7.1f} {a.min()*100:>+9.1f}% {a.max()*100:>+9.1f}% "
                  f"{np.mean(quedas)*100:>7.1f}% {quebras:>4}/{len(a)}",
                  flush=True)

    print("\n" + "=" * 78)
    print("  A ALAVANCAGEM AJUDA?")
    print("=" * 78)
    print("\n  A frase 'quanto maior o risco, maior o ganho' vale quando a")
    print("  vantagem por aposta e POSITIVA — aí a alavancagem multiplica um")
    print("  numero positivo. Com vantagem nula ou pequena, ela multiplica o")
    print("  ruido e liga o arrasto de volatilidade: perder 50% exige ganhar")
    print("  100% para voltar ao ponto de partida.")
    print("\n  A coluna 'quebrou' conta quantas rodadas zeraram a conta antes do")
    print("  fim do periodo. Numa conta zerada nao existe recuperacao — e a")
    print("  unica perda que nao e reversivel.")
    print("=" * 78)


if __name__ == "__main__":
    main()
