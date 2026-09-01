# -*- coding: utf-8 -*-
"""
execucao/papel.py — a corretora simulada
=========================================

Satisfaz nucleo.protocolos.Corretora, a MESMA interface da corretora real. E o
que garante que backtest e producao nao possam divergir: existe um caminho de
codigo (avaliacao/replay.py) com dois adaptadores por baixo. Na V6 havia duas
implementacoes da estrategia e ninguem as tinha comparado; quando comparamos,
divergiam.

Quem dispara o stop
-------------------
Aqui, como na Binance: a CORRETORA. A estrategia registra stop e alvo e some;
sao ordens em repouso que a corretora executa quando o preco encosta. Se o
replay checasse os gatilhos por fora, estariamos simulando um sistema diferente
do que roda — o mesmo erro de ter duas implementacoes.

As tres regras pessimistas
--------------------------
Sem dados de tick nao da para saber o que aconteceu dentro de um minuto, entao
toda ambiguidade e resolvida contra nos:

  1. mesma vela toca stop E alvo  -> conta como STOP
  2. gatilho executa no preco do gatilho, sem deslizamento a favor
  3. entrada e saida a mercado pagam taxa cheia de tomador

A regra 1 e a que mais importa: assumir o alvo nesses casos infla o resultado
de forma invisivel e sistematica.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from execucao.risco import Ordem
from nucleo.tipos import Fechamento, Lado, Posicao

TAXA_TOMADOR = 0.0004      # 0,04% por perna na Binance Futures — 0,08% ida e volta


class SemSaldo(Exception):
    """A conta zerou. O replay para."""


class CorretoraPapel:
    """Corretora de mentira com contabilidade de verdade."""

    def __init__(self, saldo_inicial: float = 5000.0,
                 taxa: float = TAXA_TOMADOR):
        self._saldo = float(saldo_inicial)
        self.saldo_inicial = float(saldo_inicial)
        self.taxa = float(taxa)

        self._posicoes: dict[str, Posicao] = {}       # symbol -> posicao
        self._barreiras: dict[str, tuple[float, float]] = {}   # symbol -> (stop, alvo)
        self._taxa_entrada: dict[str, float] = {}
        self.fechamentos: list[Fechamento] = []
        self._contador = 0

    # ── protocolo Corretora ────────────────────────────────────────────────
    @property
    def saldo(self) -> float:
        return self._saldo

    def posicoes(self) -> list[Posicao]:
        return list(self._posicoes.values())

    def abrir(self, ordem: Ordem) -> Posicao:
        """
        Executa a mercado no preco de referencia do sinal.

        Ao vivo, a ordem sai alguns segundos depois da decisao e e preenchida
        em pedacos — na auditoria real de 19/08 uma entrada de AVAX virou seis
        fills a precos diferentes. Isso empurra o resultado real para BAIXO do
        que sai daqui. Nao simulamos deslizamento; entao todo numero deste
        modulo e otimista, e e assim que deve ser lido.
        """
        s = ordem.sinal
        if s.symbol in self._posicoes:
            raise ValueError(f"ja ha posicao aberta em {s.symbol}")

        taxa = ordem.notional * self.taxa
        self._saldo -= taxa
        if self._saldo <= 0:
            raise SemSaldo(f"saldo zerou ao abrir {s.symbol}")

        self._contador += 1
        pos = Posicao(
            id=f"P{self._contador:05d}",
            symbol=s.symbol,
            lado=s.lado,
            entrada=s.preco_ref,
            quantidade=ordem.quantidade,
            stop=ordem.stop,
            alvo=ordem.alvo,
            aberta_em=s.ts,
            alavancagem=ordem.alavancagem,
        )
        self._posicoes[s.symbol] = pos
        self._taxa_entrada[s.symbol] = taxa
        return pos

    def proteger(self, posicao: Posicao, stop: float, alvo: float) -> bool:
        """
        Registra as barreiras. A de papel sempre confirma.

        A real so devolve True depois de reler a ordem no sistema de ordens
        condicionais — foi ali que a V6 deixou posicoes descobertas achando que
        estavam protegidas. A assinatura identica obriga o replay a tratar o
        False, entao o caminho de erro esta exercitado antes de existir dinheiro.
        """
        if posicao.symbol not in self._posicoes:
            return False
        self._barreiras[posicao.symbol] = (float(stop), float(alvo))
        return True

    def fechar(self, posicao: Posicao, causa: str,
               preco: float | None = None,
               quando: datetime | None = None) -> float:
        """Fecha a mercado e devolve o preco de saida."""
        symbol = posicao.symbol
        pos = self._posicoes.pop(symbol, None)
        if pos is None:
            raise ValueError(f"nao ha posicao aberta em {symbol}")
        self._barreiras.pop(symbol, None)

        saida = float(preco if preco is not None else pos.entrada)
        bruto = pos.resultado_em(saida)
        taxa_saida = saida * pos.quantidade * self.taxa
        taxa_total = self._taxa_entrada.pop(symbol, 0.0) + taxa_saida

        self._saldo += bruto - taxa_saida
        self.fechamentos.append(Fechamento(
            posicao=pos,
            fechada_em=quando or pos.aberta_em,
            preco_saida=saida,
            causa=causa,
            bruto=bruto,
            taxas=taxa_total,
        ))
        if self._saldo <= 0:
            raise SemSaldo(f"saldo zerou ao fechar {symbol}")
        return saida

    # ── gatilhos ───────────────────────────────────────────────────────────
    def varrer(
        self,
        symbol: str,
        instantes: np.ndarray,
        maximas: np.ndarray,
        minimas: np.ndarray,
    ) -> Fechamento | None:
        """
        Percorre minuto a minuto e dispara stop ou alvo, se algum for tocado.

        `instantes`, `maximas` e `minimas` sao as barras desde a ultima varredura.
        Devolve o Fechamento se disparou, ou None.
        """
        pos = self._posicoes.get(symbol)
        if pos is None or symbol not in self._barreiras:
            return None
        stop, alvo = self._barreiras[symbol]

        for k in range(len(instantes)):
            alta, baixa = float(maximas[k]), float(minimas[k])
            if pos.lado is Lado.LONG:
                bateu_stop, bateu_alvo = baixa <= stop, alta >= alvo
            else:
                bateu_stop, bateu_alvo = alta >= stop, baixa <= alvo

            # Regra pessimista: empate na mesma vela conta como stop.
            if bateu_stop:
                quando = instantes[k].astype("datetime64[us]").astype(datetime)
                self.fechar(pos, "STOP", preco=stop, quando=quando)
                return self.fechamentos[-1]
            if bateu_alvo:
                quando = instantes[k].astype("datetime64[us]").astype(datetime)
                self.fechar(pos, "ALVO", preco=alvo, quando=quando)
                return self.fechamentos[-1]
        return None

    # ── contabilidade ──────────────────────────────────────────────────────
    def patrimonio(self, precos: dict[str, float]) -> float:
        """Saldo mais o resultado nao realizado das posicoes abertas."""
        aberto = sum(p.resultado_em(precos[p.symbol])
                     for p in self._posicoes.values() if p.symbol in precos)
        return self._saldo + aberto

    def __repr__(self) -> str:
        return (f"<CorretoraPapel saldo ${self._saldo:,.2f} | "
                f"{len(self._posicoes)} aberta(s) | "
                f"{len(self.fechamentos)} fechada(s)>")
