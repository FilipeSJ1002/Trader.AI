# -*- coding: utf-8 -*-
"""
execucao/risco.py — de Sinal para Ordem
========================================

Os motores dizem "aqui vale entrar, com esta conviccao". Este arquivo decide
QUANTO e ONDE fica o stop. E a unica camada que conhece o saldo.

Duas escolhas de desenho que valem explicacao:

Stop e alvo em ATR, nao em porcentagem
--------------------------------------
Um stop de 1,5% significa coisas diferentes no BTC e no AVAX — no ativo mais
agitado ele e tocado por ruido normal, no mais calmo ele quase nunca dispara.
Em multiplos de ATR, "1,5 ATR" quer dizer a mesma coisa em todos: uma volta
tipica do proprio ativo. Foi um dos defeitos silenciosos das versoes anteriores,
que usavam o mesmo percentual para todo o universo.

...mas de QUAL prazo (medido em 26/08/2026)
-------------------------------------------
O ATR de 1 minuto do BTC e 0,073% do preco. Um stop de 1,5 desses ATRs fica a
0,11% da entrada — dentro do ruido, e a ida e volta de taxas custa 0,08%. Na
primeira rodada com dados reais isso produziu 9.773 operacoes em seis meses (54
por dia) e -61%, sem que a estrategia tivesse chance nenhuma.

O stop precisa estar na escala do tempo que a posicao pretende durar. Por isso
`prazo_atr` existe e vem como "diario": uma posicao que se pretende segurar por
horas ou dias tem de suportar a oscilacao normal de horas ou dias. O prazo de
1 minuto continua disponivel, mas so faz sentido para quem for sair em minutos.

Alavancagem começa em 1x
------------------------
Nao e timidez: e sequenciamento. A alavancagem multiplica ganho E perda pelo
mesmo fator, entao ela nao cria vantagem — so amplia a que existir. Medir o teto
com 1x mostra o tamanho da vantagem; escolher a alavancagem depois e uma decisao
separada, tomada com o numero na mao. Medir tudo junto e nao saber qual dos dois
produziu o resultado.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nucleo.protocolos import VisaoDeMercado
from nucleo.tipos import Lado, Sinal


@dataclass(frozen=True, slots=True)
class Ordem:
    """O que sai daqui: um Sinal ja dimensionado e com barreiras."""

    sinal: Sinal
    quantidade: float
    alavancagem: float
    stop: float
    alvo: float

    @property
    def notional(self) -> float:
        return self.quantidade * self.sinal.preco_ref

    @property
    def margem(self) -> float:
        return self.notional / self.alavancagem

    @property
    def risco_maximo(self) -> float:
        """Quanto se perde se o stop for tocado exatamente, sem deslizar."""
        return abs(self.sinal.preco_ref - self.stop) * self.quantidade


class Risco:
    """Transforma Sinal em Ordem. Nao decide SE opera — so quanto."""

    def __init__(
        self,
        fracao_por_operacao: float = 0.20,
        alavancagem: float = 1.0,
        atr_stop: float = 1.5,
        atr_alvo: float = 3.0,
        max_posicoes: int = 3,
        usar_forca: bool = True,
        prazo_atr: str = "diario",
    ):
        """
        `fracao_por_operacao` e do saldo, por operacao.
        `atr_alvo / atr_stop` e a razao alvo:stop — 3,0/1,5 e 2:1.
        `usar_forca` escala o tamanho pela conviccao do motor: um sinal de
        forca 1,0 usa a fracao inteira, um de 0,5 usa metade. Se a forca nao
        carregar informacao, isto e neutro; se carregar, aproveita.

        `prazo_atr` diz de qual prazo sai o ATR que dimensiona o stop:
        "diario", "h4" ou "minuto". Ver a nota no topo do arquivo — este
        parametro decide se o stop e uma barreira ou um sorteio.
        """
        if prazo_atr not in ("diario", "h4", "minuto"):
            raise ValueError(f"prazo_atr invalido: {prazo_atr}")
        if not 0 < fracao_por_operacao <= 1:
            raise ValueError(f"fracao invalida: {fracao_por_operacao}")
        if alavancagem < 1:
            raise ValueError(f"alavancagem minima e 1: {alavancagem}")
        if atr_stop <= 0 or atr_alvo <= 0:
            raise ValueError("multiplos de ATR precisam ser positivos")

        self.fracao = fracao_por_operacao
        self.alavancagem = alavancagem
        self.atr_stop = atr_stop
        self.atr_alvo = atr_alvo
        self.max_posicoes = max_posicoes
        self.usar_forca = usar_forca
        self.prazo_atr = prazo_atr

    def dimensionar(
        self, sinal: Sinal, visao: VisaoDeMercado, saldo: float
    ) -> Ordem | None:
        """
        Devolve a Ordem, ou None se ela nao for viavel.

        None acontece quando o ATR ainda nao existe ou o saldo acabou — casos
        em que uma ordem seria calculada sobre lixo.
        """
        atr = self._atr(visao)
        if not np.isfinite(atr) or atr <= 0 or saldo <= 0:
            return None

        preco = sinal.preco_ref
        escala = sinal.forca if self.usar_forca else 1.0
        margem = saldo * self.fracao * escala
        quantidade = (margem * self.alavancagem) / preco
        if quantidade <= 0:
            return None

        d_stop = self.atr_stop * atr
        d_alvo = self.atr_alvo * atr
        if sinal.lado is Lado.LONG:
            stop, alvo = preco - d_stop, preco + d_alvo
        else:
            stop, alvo = preco + d_stop, preco - d_alvo

        # Um stop mais largo que o proprio preco significa ATR absurdo em
        # relacao ao ativo — dado ruim, nao oportunidade.
        if stop <= 0:
            return None

        return Ordem(sinal=sinal, quantidade=quantidade,
                     alavancagem=self.alavancagem, stop=stop, alvo=alvo)

    def _atr(self, visao: VisaoDeMercado) -> float:
        """O ATR do prazo escolhido. Ver a nota no topo do arquivo."""
        if self.prazo_atr == "minuto":
            return visao.agora("atr")
        alvo = visao.diario if self.prazo_atr == "diario" else visao.h4
        return alvo.agora("atr")

    def tem_espaco(self, abertas: int) -> bool:
        return abertas < self.max_posicoes

    def __repr__(self) -> str:
        return (f"<Risco {self.fracao:.0%}/op x{self.alavancagem:g} "
                f"stop {self.atr_stop}ATR({self.prazo_atr}) "
                f"alvo {self.atr_alvo}ATR max {self.max_posicoes}>")
