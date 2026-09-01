# -*- coding: utf-8 -*-
"""
avaliacao/metricas.py — o que uma rodada significa
===================================================

Duas medidas que nao podem faltar, porque a ausencia delas ja enganou este
projeto:

  rebaixamento maximo   uma estrategia com +2%/mes e queda de 60% e pior que
                        uma com +1%/mes e queda de 8%: a primeira quebra a
                        conta antes de chegar ao fim. Retorno sozinho mente.
  numero de operacoes   abaixo de ~50, o resultado e sorte. A metrica existe
                        para que ninguem ordene versoes por um numero apoiado
                        em doze operacoes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from avaliacao.replay import Resultado


@dataclass(frozen=True, slots=True)
class Metricas:
    retorno: float
    mensal: float
    rebaixamento: float
    operacoes: int
    acerto: float
    taxas: float
    expectancia: float
    saldo_final: float
    dias: int

    def __str__(self) -> str:
        return (f"{self.retorno*100:+.2f}% ({self.mensal*100:+.2f}%/mes) | "
                f"queda {self.rebaixamento*100:.1f}% | "
                f"{self.operacoes} ops | acerto {self.acerto*100:.0f}%")


def medir(res: Resultado) -> Metricas:
    """Reduz uma rodada de replay a numeros comparaveis."""
    curva = np.array([v for _, v in res.curva], dtype=float)
    instantes = [t for t, _ in res.curva]

    dias = ((instantes[-1] - instantes[0]).days if len(instantes) > 1 else 1)
    dias = max(dias, 1)
    retorno = res.retorno
    meses = max(dias / 30.44, 1e-9)
    mensal = (1 + retorno) ** (1 / meses) - 1 if retorno > -1 else -1.0

    if len(curva):
        pico = np.maximum.accumulate(curva)
        rebaixamento = float((curva / pico - 1.0).min())
    else:
        rebaixamento = 0.0

    liquidos = np.array([f.liquido for f in res.fechamentos], dtype=float)
    n = len(liquidos)

    return Metricas(
        retorno=retorno,
        mensal=mensal,
        rebaixamento=rebaixamento,
        operacoes=n,
        acerto=float((liquidos > 0).mean()) if n else 0.0,
        taxas=float(sum(f.taxas for f in res.fechamentos)),
        expectancia=float(liquidos.mean()) if n else 0.0,
        saldo_final=res.saldo_final,
        dias=dias,
    )


def comprar_e_segurar(historicos, de, ate, capital: float) -> Metricas:
    """
    A referencia obrigatoria: carteira de peso igual, comprada o periodo todo.

    Se a estrategia nao bate uma carteira parada, ela nao paga o trabalho nem o
    risco. Na V8, nenhuma das 14 configuracoes bateu.
    """
    curvas = []
    for h in historicos.values():
        i0, i1 = h.indice_de(de), h.indice_de(ate)
        serie = h.em(i1).serie("fechamento")[i0:i1 + 1]
        curvas.append(serie / serie[0])

    n = min(len(c) for c in curvas)
    carteira = np.mean([c[:n] for c in curvas], axis=0) * capital

    dias = max((ate - de).days, 1)
    retorno = float(carteira[-1] / capital - 1)
    meses = max(dias / 30.44, 1e-9)
    pico = np.maximum.accumulate(carteira)

    return Metricas(
        retorno=retorno,
        mensal=(1 + retorno) ** (1 / meses) - 1 if retorno > -1 else -1.0,
        rebaixamento=float((carteira / pico - 1.0).min()),
        operacoes=0,
        acerto=float("nan"),
        taxas=0.0,
        expectancia=0.0,
        saldo_final=float(carteira[-1]),
        dias=dias,
    )
