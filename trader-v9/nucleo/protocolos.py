# -*- coding: utf-8 -*-
"""
nucleo/protocolos.py — as interfaces que todo o resto obedece
=============================================================

Sao `Protocol`, nao classes-base: nada precisa herdar de nada. Uma classe que
tem os metodos certos JA satisfaz o contrato, e o verificador de tipos reclama
antes de rodar. Isso mantem os pacotes desacoplados de verdade — `motores/` nao
importa `execucao/` nem o contrario.

Os quatro contratos e o defeito de versao anterior que cada um previne:

  VisaoDeMercado  a estrategia so recebe passado. Ver dados/visao.py.
  Motor           um especialista so; nao conhece saldo, corretora nem o outro.
  Oraculo         uma decisao por dia; nao escolhe ativo, hora nem preco.
  Corretora       UMA interface para a real e a de papel. Na V6 havia duas
                  implementacoes da mesma estrategia e elas divergiam.
"""
from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

import numpy as np

from nucleo.tipos import Lado, Posicao, Regime, Sinal


@runtime_checkable
class VisaoDeMercado(Protocol):
    """
    A janela pela qual uma estrategia enxerga o mercado.

    A garantia central do projeto: um objeto que satisfaz este protocolo contem
    APENAS dados ate `ts`. Nao e disciplina de quem escreve a estrategia — os
    arrays devolvidos por `serie()` terminam no instante atual porque nunca
    receberam o resto. Nao ha como olhar adiante, nem por engano.
    """

    symbol: str
    ts: datetime

    def serie(self, nome: str) -> np.ndarray:
        """A coluna inteira ate agora. O ultimo elemento e o instante atual."""
        ...

    def agora(self, nome: str) -> float:
        """Valor atual de uma coluna. NaN se ainda nao ha dados suficientes."""
        ...

    def antes(self, nome: str, passos: int = 1) -> float:
        """Valor de `passos` barras atras. Para detectar cruzamentos."""
        ...

    @property
    def fechamento(self) -> float:
        ...

    @property
    def diario(self) -> "VisaoDeMercado":
        """A mesma janela em barras de 1 dia, so com barras JA FECHADAS."""
        ...

    @property
    def h4(self) -> "VisaoDeMercado":
        """Idem, em barras de 4 horas."""
        ...


@runtime_checkable
class Motor(Protocol):
    """
    Um especialista. Ve o mercado, devolve um sinal ou nada.

    Nao conhece: o saldo, a corretora, o outro motor, o oraculo, nem quantas
    posicoes existem. Essa ignorancia e o que permite testa-lo isolado e o que
    impede que "agressividade" vire logica espalhada por varios arquivos.

    `lado` e fixo por motor e validado em motores/base.py: um Motor Bull que
    emita SHORT e um defeito, nao uma estrategia.
    """

    nome: str
    lado: Lado

    def avaliar(self, visao: VisaoDeMercado) -> Sinal | None:
        ...


@runtime_checkable
class Oraculo(Protocol):
    """
    De quem e o dia.

    Uma decisao por dia sobre contexto macro. Deliberadamente pobre em poder:
    nao escolhe ativo, nao escolhe hora, nao escolhe preco. Se o Oraculo pudesse
    fazer essas coisas, voltariamos a ter uma rede decidindo minuto a minuto —
    exatamente o que a V6 fez e que nao se sustentou na medicao.
    """

    nome: str

    def regime(self, visao: VisaoDeMercado) -> Regime:
        ...


@runtime_checkable
class Corretora(Protocol):
    """
    Real ou de papel — a MESMA interface.

    E o que garante que backtest e producao nao possam divergir: existe um
    caminho de codigo (avaliacao/replay.py) com dois adaptadores por baixo.

    `proteger` devolve bool de proposito. A de papel sempre confirma; a real so
    confirma depois de reler a ordem no sistema de ordens condicionais (algo
    orders), onde stop e alvo realmente vivem. Foi ali que a V6 deixou posicoes
    descobertas achando que estavam protegidas.
    """

    def abrir(self, sinal: Sinal, quantidade: float, alavancagem: float) -> Posicao:
        ...

    def proteger(self, posicao: Posicao, stop: float, alvo: float) -> bool:
        """True somente com stop E alvo confirmados na corretora."""
        ...

    def fechar(self, posicao: Posicao, causa: str) -> float:
        """Fecha a mercado e devolve o preco efetivo de saida."""
        ...

    def posicoes(self) -> list[Posicao]:
        ...

    @property
    def saldo(self) -> float:
        ...


@runtime_checkable
class Fonte(Protocol):
    """Origem dos dados historicos. Implementada em dados/fonte.py."""

    def simbolos(self) -> list[str]:
        ...

    def historico(self, symbol: str) -> object:
        """Devolve um dados.visao.Historico pronto, com indicadores calculados."""
        ...
