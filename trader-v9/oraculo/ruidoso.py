# -*- coding: utf-8 -*-
"""
oraculo/ruidoso.py — o oraculo com acuracia controlada
=======================================================

Por que este arquivo substitui o teto como criterio (26/08/2026)
---------------------------------------------------------------
A primeira medicao de teto usou um oraculo perfeito e devolveu +11.212.412.123%
em 5,5 anos. O numero esta aritmeticamente correto e e inutil: ninguem vai
acertar 100% dos dias, entao ele nao decide nada. Pior, o diagnostico mostrou
que 537 das 574 saidas de 2025 foram por VIRADA DE REGIME — a previsao perfeita
vazava para o momento de sair, e o que estava sendo medido era uma maquina de
adivinhar o futuro, nao a arquitetura de tres pilares.

A pergunta que decide a Sprint 2 nao e "quanto renderia se acertassemos sempre".
E: **a partir de qual acuracia isto passa a valer a pena?**

Se a resposta for 55%, ha o que disputar — e o alvo do classificador esta posto.
Se for 70%, o projeto morre aqui, porque prever a direcao diaria de cripto acima
de ~55% de forma sustentada nao e coisa que exista na literatura seria.

Como funciona
-------------
Sabe a resposta certa (herda do OraculoPerfeito) e a estraga de proposito: com
probabilidade `acuracia` devolve o regime verdadeiro, senao devolve o oposto. A
decisao e por DIA e memorizada, entao o mesmo dia nunca muda de resposta no meio.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from dados.visao import Historico
from nucleo.protocolos import VisaoDeMercado
from nucleo.tipos import Regime
from oraculo.teto import OraculoPerfeito


class OraculoRuidoso(OraculoPerfeito):
    """Acerta a direcao do dia com probabilidade `acuracia`."""

    def __init__(
        self,
        historicos: dict[str, Historico],
        acuracia: float = 0.55,
        semente: int = 0,
        limiar_fora: float = 0.0,
    ):
        if not 0.0 <= acuracia <= 1.0:
            raise ValueError(f"acuracia fora de [0,1]: {acuracia}")
        super().__init__(historicos, limiar_fora=limiar_fora)
        self.nome = f"acuracia {acuracia:.0%}"
        self.acuracia = float(acuracia)
        self._rng = np.random.default_rng(semente)
        self._decidido: dict[date, Regime] = {}

    def regime(self, visao: VisaoDeMercado) -> Regime:
        dia = visao.ts.date()
        if dia in self._decidido:
            return self._decidido[dia]

        verdade = self._mapa.get(dia, Regime.FORA)
        if verdade is Regime.FORA:
            escolha = Regime.FORA
        elif self._rng.random() < self.acuracia:
            escolha = verdade
        else:
            escolha = Regime.BEAR if verdade is Regime.BULL else Regime.BULL

        self._decidido[dia] = escolha
        return escolha

    def acuracia_realizada(self) -> float:
        """A acuracia que de fato saiu — confere se o sorteio fez o combinado."""
        pares = [(d, r) for d, r in self._decidido.items()
                 if self._mapa.get(d) not in (None, Regime.FORA)]
        if not pares:
            return float("nan")
        return float(np.mean([r is self._mapa[d] for d, r in pares]))
