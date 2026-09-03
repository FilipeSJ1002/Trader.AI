# -*- coding: utf-8 -*-
"""
app/sobreposicao.py — usar o sinal fraco para PROTEGER, não para gerar
======================================================================

A pergunta que faltava (02/09/2026)
-----------------------------------
Todas as medições anteriores compararam o sistema contra comprar-e-segurar e
perderam. Mas elas partiam de uma premissa que nunca foi questionada: a de que
o sistema tem de gerar o retorno do zero, saindo do mercado e operando nos dois
sentidos.

Os números dizem outra coisa. O retorno do período está em SEGURAR os ativos
(+216% em 3,5 anos). O nosso sinal vale 3 pontos percentuais acima do acaso —
fraco demais para sustentar um sistema inteiro, mas talvez não para uma tarefa
menor: decidir quando NÃO estar exposto.

A sobreposição inverte o desenho:

  base        comprado nos 6 ativos, peso igual, o tempo todo
  sinal       quando o oráculo prevê BAIXA para os próximos dias, sai para caixa
  volta       quando prevê ALTA, recompra

Um sinal de 53,7% erra 46,3% das vezes, então ele vai tirar você do mercado em
altas (custo de oportunidade) e vai te salvar em quedas (ganho). A pergunta é se
o saldo entre os dois é positivo depois das taxas.

Por que isto pode funcionar onde o resto falhou
-----------------------------------------------
  1. o retorno de base não depende do sinal — ele vem do mercado
  2. são ~100 trocas em vez de ~1.000: um décimo do custo de transação
  3. um sinal fraco não precisa estar certo na média para reduzir o
     rebaixamento; basta acertar parte das quedas grandes

E por que pode não funcionar: em mercado de alta prolongada, cada saída errada
custa o movimento inteiro perdido. Mede-se.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

import numpy as np

from dados.fonte import FonteParquet, ler_config
from execucao.papel import TAXA_TOMADOR
from nucleo.tipos import Regime
from oraculo.ruidoso import OraculoLento

DE, ATE = datetime(2023, 1, 1), datetime(2026, 7, 25)
ACURACIAS = [0.500, 0.5369, 0.560, 0.600]
SEMENTES = [1, 2, 3, 4, 5]
PASSOS = [3, 7]


def serie_diaria(historicos):
    """Fechamento diário médio do universo, normalizado. A base comprada."""
    curvas, dias_ref = [], None
    for h in historicos.values():
        i0, i1 = h.indice_de(DE), h.indice_de(ATE)
        ts = h.instantes[i0:i1 + 1]
        fech = h.em(i1).serie("fechamento")[i0:i1 + 1]
        # Um ponto por dia: o último minuto de cada dia.
        dias = ts.astype("datetime64[D]")
        corte = np.concatenate([np.nonzero(np.diff(dias))[0], [len(dias) - 1]])
        curvas.append(fech[corte] / fech[corte][0])
        if dias_ref is None:
            dias_ref = dias[corte]
    n = min(len(c) for c in curvas)
    return dias_ref[:n], np.mean([c[:n] for c in curvas], axis=0)


def simular(dias, base, regimes, taxa):
    """
    Percorre os dias mantendo ou não a exposição.

    `regimes[i]` diz o que o oráculo previu para o dia i. Exposto = 1 quando a
    previsão é de alta; 0 quando é de baixa. A taxa é cobrada em cada MUDANÇA
    de exposição, nas duas pernas.
    """
    ret = np.diff(base) / base[:-1]
    capital, exposto, trocas = 1.0, True, 0
    curva = [1.0]

    for i, r in enumerate(ret):
        alvo = regimes[i] is not Regime.BEAR
        if alvo != exposto:
            capital *= (1 - taxa)      # sai ou entra: paga uma perna
            trocas += 1
            exposto = alvo
        if exposto:
            capital *= (1 + r)
        curva.append(capital)

    return np.array(curva), trocas


def metricas(curva):
    total = float(curva[-1] - 1)
    pico = np.maximum.accumulate(curva)
    return total, float((curva / pico - 1).min())


def main() -> None:
    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]
    print(f"Carregando {len(ativos)} ativos...", flush=True)
    h = FonteParquet(cfg=cfg).carregar(ativos)

    dias, base = serie_diaria(h)
    bh_total, bh_queda = metricas(base)
    print(f"\n{DE:%Y-%m-%d} a {ATE:%Y-%m-%d} | {len(dias)} dias")
    print(f"COMPRAR E SEGURAR: {bh_total*100:+.1f}% | "
          f"queda máxima {bh_queda*100:.1f}%\n", flush=True)

    print(f"  {'passo':>6} {'acurácia':>9} {'retorno':>11} {'erro':>7} "
          f"{'queda':>8} {'trocas':>7} {'vs segurar':>11}")
    print("  " + "-" * 66, flush=True)

    for passo in PASSOS:
        for acc in ACURACIAS:
            totais, quedas, trocas_l = [], [], []
            for semente in SEMENTES:
                o = OraculoLento(h, passo=passo, acuracia=acc, semente=semente)
                # O regime de cada dia, consultado uma vez por dia.
                regimes = []
                for d in dias[:-1]:
                    dia = d.astype("datetime64[us]").astype(datetime)
                    i = h[ativos[0]].indice_de(dia + timedelta(hours=12))
                    regimes.append(o.regime(h[ativos[0]].em(i)))
                curva, trocas = simular(dias, base, regimes, TAXA_TOMADOR)
                t, q = metricas(curva)
                totais.append(t)
                quedas.append(q)
                trocas_l.append(trocas)

            a = np.array(totais)
            erro = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
            dif = a.mean() - bh_total
            print(f"  {passo:>4}d  {acc:>8.1%} {a.mean()*100:>+10.1f}% "
                  f"{erro*100:>6.1f} {np.mean(quedas)*100:>7.1f}% "
                  f"{np.mean(trocas_l):>7.0f} {dif*100:>+10.1f}pp", flush=True)

    print("\n" + "=" * 72)
    print("  A SOBREPOSIÇÃO FUNCIONA?")
    print("=" * 72)
    print("\n  A linha de 50% é o controle: com previsão sem valor, sair e voltar")
    print("  do mercado ao acaso só paga taxa e perde parte das altas. Ela tem")
    print("  de ficar ABAIXO de comprar-e-segurar — se não ficar, há defeito.")
    print("\n  A linha de 53,7% é o que temos. Se ela superar comprar-e-segurar")
    print("  em retorno OU reduzir a queda máxima substancialmente sem perder")
    print("  muito retorno, há algo aproveitável aqui.")
    print("=" * 72)


if __name__ == "__main__":
    main()
