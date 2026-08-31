# -*- coding: utf-8 -*-
"""
nucleo/tipos.py — os dados que atravessam o sistema
====================================================

Camada mais baixa. Nao importa NADA do projeto e nao contem regra de negocio:
so as formas que todas as outras camadas concordam em usar.

Tudo aqui e congelado (`frozen=True`). Um Sinal que atravessa tres camadas nao
pode ser alterado no caminho — se o motor emitiu forca 0,8, e 0,8 que chega ao
risco. Bugs de mutacao a distancia foram caros nas versoes anteriores e aqui
sao impossiveis por construcao.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class Lado(str, Enum):
    """Direcao de uma operacao. Herda de str para serializar direto em TOML/JSON."""

    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sinal(self) -> int:
        """+1 para LONG, -1 para SHORT. Evita `if lado == ...` espalhado."""
        return 1 if self is Lado.LONG else -1

    @property
    def oposto(self) -> "Lado":
        return Lado.SHORT if self is Lado.LONG else Lado.LONG


class Regime(str, Enum):
    """
    De quem e o dia, segundo o Oraculo.

    FORA existe por uma razao estatistica, nao por cautela: um classificador
    binario obrigado a escolher um lado todo dia acerta ~50% por construcao e
    fica indistinguivel de moeda. Com o direito de dizer "hoje nenhum", o acerto
    passa a carregar informacao.
    """

    BULL = "BULL"
    BEAR = "BEAR"
    FORA = "FORA"

    @property
    def lado(self) -> Lado | None:
        """O lado que este regime autoriza, ou None se o dia e de ninguem."""
        if self is Regime.BULL:
            return Lado.LONG
        if self is Regime.BEAR:
            return Lado.SHORT
        return None


@dataclass(frozen=True, slots=True)
class Barra:
    """Uma vela. Os campos tem o nome que a corretora usa, sem traducao."""

    ts: datetime
    abertura: float
    maxima: float
    minima: float
    fechamento: float
    volume: float

    @property
    def amplitude(self) -> float:
        return self.maxima - self.minima

    @property
    def corpo(self) -> float:
        """Positivo em vela de alta, negativo em vela de baixa."""
        return self.fechamento - self.abertura


@dataclass(frozen=True, slots=True)
class Sinal:
    """
    O que um Motor emite. Uma INTENCAO, nao uma ordem.

    Repare no que NAO existe aqui: quantidade, margem, alavancagem. O motor nao
    conhece o tamanho da conta — ele diz "aqui vale entrar, com esta conviccao".
    Quem transforma isso em ordem e execucao/risco.py.

    Essa separacao e o que permite mudar a agressividade do sistema alterando um
    numero no config, sem tocar em nenhum motor.
    """

    ts: datetime
    symbol: str
    lado: Lado
    preco_ref: float
    forca: float          # 0,0 a 1,0 — conviccao relativa DENTRO deste motor
    motivo: str           # legivel por humano; aparece no log e na auditoria

    def __post_init__(self) -> None:
        if not 0.0 <= self.forca <= 1.0:
            raise ValueError(f"forca fora de [0,1]: {self.forca}")
        if self.preco_ref <= 0:
            raise ValueError(f"preco_ref invalido: {self.preco_ref}")


@dataclass(frozen=True, slots=True)
class Posicao:
    """Uma posicao viva na corretora — real ou de papel."""

    id: str
    symbol: str
    lado: Lado
    entrada: float
    quantidade: float
    stop: float
    alvo: float
    aberta_em: datetime
    alavancagem: float = 1.0

    @property
    def notional(self) -> float:
        return self.entrada * self.quantidade

    def resultado_em(self, preco: float) -> float:
        """PnL bruto em dolares a este preco, sem taxas."""
        return self.quantidade * self.lado.sinal * (preco - self.entrada)

    def bateu_stop(self, maxima: float, minima: float) -> bool:
        if self.lado is Lado.LONG:
            return minima <= self.stop
        return maxima >= self.stop

    def bateu_alvo(self, maxima: float, minima: float) -> bool:
        if self.lado is Lado.LONG:
            return maxima >= self.alvo
        return minima <= self.alvo


@dataclass(frozen=True, slots=True)
class Fechamento:
    """O registro de uma posicao encerrada. Entra direto nas metricas."""

    posicao: Posicao
    fechada_em: datetime
    preco_saida: float
    causa: str            # "ALVO", "STOP", "REGIME", "TEMPO", "FIM"
    bruto: float
    taxas: float

    @property
    def liquido(self) -> float:
        return self.bruto - self.taxas

    @property
    def minutos(self) -> float:
        return (self.fechada_em - self.posicao.aberta_em).total_seconds() / 60.0
