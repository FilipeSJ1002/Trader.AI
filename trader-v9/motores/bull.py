# -*- coding: utf-8 -*-
"""
motores/bull.py — o especialista de alta
=========================================

A logica raiz da V1: comprar a correcao. Quando o Oraculo diz que o dia e de
alta, este motor procura o momento em que o preco recuou o bastante para a
entrada valer mais a pena, sem esperar confirmacao demais.

Tres leituras, somadas com peso:

  RSI baixo       o preco recuou em relacao a si mesmo
  banda inferior  recuou em relacao a propria volatilidade (comparavel entre
                  ativos, ao contrario de uma queda em porcentagem)
  MACD virando    o recuo esta perdendo forca — sem isto, compra-se faca caindo

Nenhuma sozinha basta, e nenhuma e obrigatoria: a forca e a soma ponderada, e
o motor dispara ao passar de `forca_min`. Esse desenho e o que torna a
agressividade um numero, e nao uma reescrita.

Este motor NAO checa tendencia. Quem faz isso e o Oraculo — ver motores/base.py.
"""
from __future__ import annotations

from motores.base import MotorBase, rampa, virando_para_cima
from nucleo.protocolos import VisaoDeMercado
from nucleo.tipos import Lado


class MotorBull(MotorBase):
    """Compra correcoes. Nunca vende."""

    nome = "bull"
    lado = Lado.LONG

    def __init__(
        self,
        forca_min: float = 0.35,
        rsi_de: float = 48.0,
        rsi_ate: float = 22.0,
        bb_de: float = 0.42,
        bb_ate: float = 0.02,
        peso_rsi: float = 0.35,
        peso_bb: float = 0.35,
        peso_macd: float = 0.30,
    ):
        """
        Os limiares sao FAIXAS, nao portas. Um RSI de 47 ja pontua um pouco;
        22 pontua cheio. Limiar duro faz o sistema ligar e desligar num tick de
        diferenca — e foi assim que a V1 gerou sinais que nao se repetiam.

        Padroes agressivos de proposito: RSI comeca a contar em 48, nao em 30.
        """
        super().__init__(forca_min)
        self.rsi_de, self.rsi_ate = rsi_de, rsi_ate
        self.bb_de, self.bb_ate = bb_de, bb_ate
        self.peso_rsi, self.peso_bb, self.peso_macd = peso_rsi, peso_bb, peso_macd

        soma = peso_rsi + peso_bb + peso_macd
        if abs(soma - 1.0) > 1e-9:
            raise ValueError(f"os pesos precisam somar 1,0 — somam {soma}")

    def _pontuar(self, v: VisaoDeMercado) -> tuple[float, list[str]]:
        rsi = v.agora("rsi")
        bb = v.agora("bb_pos")

        p_rsi = rampa(rsi, self.rsi_de, self.rsi_ate)
        p_bb = rampa(bb, self.bb_de, self.bb_ate)
        p_macd = 1.0 if virando_para_cima(v) else 0.0

        forca = (self.peso_rsi * p_rsi
                 + self.peso_bb * p_bb
                 + self.peso_macd * p_macd)

        razoes = []
        if p_rsi > 0:
            razoes.append(f"RSI {rsi:.0f}")
        if p_bb > 0:
            razoes.append(f"banda {bb:.2f}")
        if p_macd > 0:
            razoes.append("MACD virando pra cima")
        if not razoes:
            razoes.append("sem leitura favoravel")

        return forca, razoes
