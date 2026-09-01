# -*- coding: utf-8 -*-
"""
motores/base.py — o contrato comum dos especialistas
=====================================================

Um Motor e um especialista de UM lado. O Bull nunca vende, o Bear nunca compra.
Isso nao e convencao — e verificado aqui, em `emitir()`, e um motor que tente
violar levanta excecao em vez de mandar uma ordem errada para a corretora.

O que um motor NAO faz, por desenho:

  nao filtra regime    quem decide se o dia e de alta ou de baixa e o Oraculo.
                       Se o motor tambem filtrasse, haveria duas decisoes de
                       regime brigando, e o teto medido na Sprint 1 nao valeria
                       nada — o oraculo perfeito apareceria vetado pelo motor.
  nao dimensiona       o Sinal nao tem quantidade nem alavancagem. Quem
                       transforma sinal em ordem e execucao/risco.py.
  nao conhece o saldo  nem quantas posicoes existem, nem o outro motor.

Essa pobreza deliberada e o que permite testar um motor isolado e mudar a
agressividade do sistema mexendo num numero do config.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from nucleo.protocolos import VisaoDeMercado
from nucleo.tipos import Lado, Sinal


class MotorBase(ABC):
    """Base dos especialistas. Satisfaz nucleo.protocolos.Motor."""

    nome: str = "base"
    lado: Lado

    def __init__(self, forca_min: float = 0.35):
        """
        `forca_min` e o botao de agressividade: quanto menor, mais o motor
        dispara. 0,35 e agressivo de proposito — a V9 nasce para operar, e
        e o Oraculo que segura a mao, nao o motor.
        """
        if not 0.0 <= forca_min <= 1.0:
            raise ValueError(f"forca_min fora de [0,1]: {forca_min}")
        self.forca_min = forca_min

    # ── o que cada especialista implementa ─────────────────────────────────
    @abstractmethod
    def _pontuar(self, v: VisaoDeMercado) -> tuple[float, list[str]]:
        """
        Devolve (forca em [0,1], razoes legiveis).

        Forca e conviccao DENTRO deste motor — nao e comparavel com a do outro.
        Comparar Bull com Bear e trabalho do Oraculo, e ele nao usa este numero.
        """

    # ── o caminho unico de saida ───────────────────────────────────────────
    def avaliar(self, v: VisaoDeMercado) -> Sinal | None:
        """
        Ve o mercado, devolve um sinal ou nada.

        A validacao de lado mora AQUI, e nao em `emitir()`, por um motivo que um
        teste descobriu: dentro de `emitir` a checagem seria vazia — ele mesmo
        constroi o Sinal com `lado=self.lado`, entao comparar os dois e sempre
        verdadeiro e nada seria protegido. Validando o sinal DEVOLVIDO, uma
        subclasse que reescreva `emitir` tambem e pega.

        `avaliar` e o metodo do protocolo: e por ele que o replay chama. Uma
        subclasse que reescreva `avaliar` escapa da checagem, mas nesse ponto
        ela nao esta mais usando esta base — esta escrevendo outro motor.
        """
        if not v.pronta:
            return None

        forca, razoes = self._pontuar(v)
        if not np.isfinite(forca) or forca < self.forca_min:
            return None

        sinal = self.emitir(v, forca, " + ".join(razoes))
        if sinal is not None and sinal.lado is not self.lado:
            raise AssertionError(
                f"{self.nome} e especialista em {self.lado.value} mas emitiu "
                f"{sinal.lado.value}. Isto e defeito, nao estrategia."
            )
        return sinal

    def emitir(self, v: VisaoDeMercado, forca: float, motivo: str) -> Sinal:
        """Constroi o Sinal. Quem valida o lado e `avaliar`."""
        return Sinal(
            ts=v.ts,
            symbol=v.symbol,
            lado=self.lado,
            preco_ref=v.fechamento,
            forca=float(min(max(forca, 0.0), 1.0)),
            motivo=f"[{self.nome}] {motivo}",
        )

    def __repr__(self) -> str:
        return f"<{self.nome} {self.lado.value} forca_min={self.forca_min}>"


# ── Ajudantes de pontuacao ──────────────────────────────────────────────────
#
# Ambos os motores usam as mesmas formas, com os sinais trocados. Ficam aqui
# para que Bull e Bear nao possam divergir em como medem a mesma coisa.

def rampa(x: float, comeco: float, fim: float) -> float:
    """
    Interpola 0 -> 1 conforme x vai de `comeco` a `fim`.

    Funciona nos dois sentidos: rampa(rsi, 45, 20) cresce enquanto o RSI cai.
    Devolve 0 para NaN, entao um indicador ainda sem valor nunca pontua.
    """
    if not np.isfinite(x):
        return 0.0
    if comeco == fim:
        return 1.0 if x == comeco else 0.0
    t = (x - comeco) / (fim - comeco)
    return float(min(max(t, 0.0), 1.0))


def virando_para_cima(v: VisaoDeMercado, coluna: str = "macd_hist") -> bool:
    """O histograma subiu em relacao a barra anterior."""
    ant, ago = v.antes(coluna), v.agora(coluna)
    return bool(np.isfinite(ant) and np.isfinite(ago) and ago > ant)


def virando_para_baixo(v: VisaoDeMercado, coluna: str = "macd_hist") -> bool:
    ant, ago = v.antes(coluna), v.agora(coluna)
    return bool(np.isfinite(ant) and np.isfinite(ago) and ago < ant)
