# -*- coding: utf-8 -*-
"""
motores/bear.py — o especialista de baixa
==========================================

O espelho do Bull: vende o repique. Quando o Oraculo diz que o dia e de baixa,
este motor procura o momento em que o preco subiu o bastante para a venda valer
mais a pena.

  RSI alto        subiu em relacao a si mesmo
  banda superior  subiu em relacao a propria volatilidade
  MACD virando    o repique esta perdendo forca

Por que os limiares NAO sao simetricos aos do Bull
--------------------------------------------------
Seria elegante espelhar 48/22 em 52/78 e pronto. Mas os mercados que operamos
sobem devagar e caem rapido: a distribuicao do RSI e assimetrica, e um RSI de
78 e mais raro que um de 22. Espelhar os numeros faria o Bear disparar menos
que o Bull, e a comparacao entre os dois — que e todo o trabalho do Oraculo —
ficaria enviesada por um detalhe de calibragem, nao por qualidade.

Os padroes daqui sao um pouco mais frouxos para compensar. Se estao certos e
questao empirica, e a Sprint 1 responde: se o Bear quase nunca disparar nos
dias que o oraculo perfeito deu a ele, o problema esta aqui.
"""
from __future__ import annotations

from motores.base import MotorBase, rampa, virando_para_baixo
from nucleo.protocolos import VisaoDeMercado
from nucleo.tipos import Lado


class MotorBear(MotorBase):
    """Vende repiques. Nunca compra."""

    nome = "bear"
    lado = Lado.SHORT

    def __init__(
        self,
        forca_min: float = 0.35,
        rsi_de: float = 50.0,
        rsi_ate: float = 74.0,
        bb_de: float = 0.56,
        bb_ate: float = 0.96,
        peso_rsi: float = 0.35,
        peso_bb: float = 0.35,
        peso_macd: float = 0.30,
    ):
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
        p_macd = 1.0 if virando_para_baixo(v) else 0.0

        forca = (self.peso_rsi * p_rsi
                 + self.peso_bb * p_bb
                 + self.peso_macd * p_macd)

        razoes = []
        if p_rsi > 0:
            razoes.append(f"RSI {rsi:.0f}")
        if p_bb > 0:
            razoes.append(f"banda {bb:.2f}")
        if p_macd > 0:
            razoes.append("MACD virando pra baixo")
        if not razoes:
            razoes.append("sem leitura favoravel")

        return forca, razoes
