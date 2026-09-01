# -*- coding: utf-8 -*-
"""
avaliacao/robustez.py — nenhum resultado e um numero so
========================================================

Principio III da arquitetura, e o que mais custou caro para ser aprendido.

Em 25/08/2026 descobrimos que deslocar o inicio de uma simulacao da V8 em sete
minutos mudava o resultado de -4,90% para +8,24%. O desvio padrao entre rodadas
que deveriam ser identicas era de 5 a 16 pontos percentuais — maior que a
diferenca entre as configuracoes que o ranking pretendia ordenar. Aquele
ranking, e todos os anteriores, eram ordenacao de sorte.

O que este arquivo faz
----------------------
Roda a mesma configuracao em TODAS as fases do ciclo. Com a_cada=15, sao 15
rodadas: uma avaliando :00/:15/:30/:45, outra :01/:16/:31/:46, e assim por
diante. A fase e uma escolha sem significado economico nenhum — um sistema real
adota uma por acaso. Se o resultado depender dela, o resultado e sorte.

Diferente da varredura da V8, esta enumera o espaco INTEIRO da escolha
arbitraria, em vez de amostrar sete pontos dele. A media e o desvio saem
completos, nao estimados.

Como ler a saida
----------------
Uma configuracao so merece atencao se |t| >= 2, ou seja, se a media estiver a
mais de dois erros padrao de zero. E se varias forem comparadas, o limiar tem
de ser corrigido pelo numero de comparacoes — testar cinco coisas quase garante
que uma pareca boa por acaso.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np

from avaliacao.metricas import Metricas, medir
from avaliacao.replay import replay


@dataclass(frozen=True, slots=True)
class Robustez:
    """O resultado de uma configuracao, com a incerteza junto."""

    nome: str
    media: float
    erro: float
    pior: float
    melhor: float
    rebaixamento_medio: float
    operacoes_medio: float
    rodadas: int

    @property
    def t(self) -> float:
        return self.media / self.erro if self.erro > 0 else 0.0

    @property
    def distinguivel(self) -> bool:
        """Longe de zero o bastante para nao ser acaso (limiar bruto)."""
        return abs(self.t) >= 2.0

    @property
    def amplitude(self) -> float:
        """A distancia entre a pior e a melhor rodada: o tamanho da sorte."""
        return self.melhor - self.pior

    def __str__(self) -> str:
        marca = " *" if self.distinguivel else ""
        return (f"{self.nome:<28} {self.media*100:>+7.2f}% "
                f"± {self.erro*100:>5.2f}  t={self.t:>+5.2f}  "
                f"[{self.pior*100:>+6.1f}% .. {self.melhor*100:>+6.1f}%]"
                f"{marca}")


def varrer_fases(
    nome: str,
    montar: Callable[[], tuple],
    de: datetime,
    ate: datetime,
    a_cada: int = 15,
    fases: list[int] | None = None,
    aoprogresso: Callable[[int, int], None] | None = None,
) -> tuple[Robustez, list[Metricas]]:
    """
    Roda a mesma configuracao em todas as fases do ciclo.

    `montar` devolve, a cada chamada, uma tupla pronta para o replay:
        (historicos, motores, oraculo, corretora, risco, referencia)
    Precisa ser uma FUNCAO, e nao objetos prontos, porque a corretora guarda
    saldo e posicoes: reaproveitar a mesma entre rodadas contaminaria uma com
    o estado da outra.
    """
    fases = list(range(a_cada)) if fases is None else fases
    medidas: list[Metricas] = []

    for k, fase in enumerate(fases, 1):
        historicos, motores, oraculo, corretora, risco, referencia = montar()
        res = replay(historicos, motores, oraculo, corretora, risco,
                     de, ate, a_cada=a_cada, referencia=referencia, fase=fase)
        medidas.append(medir(res))
        if aoprogresso:
            aoprogresso(k, len(fases))

    retornos = np.array([m.retorno for m in medidas], dtype=float)
    erro = (float(retornos.std(ddof=1) / np.sqrt(len(retornos)))
            if len(retornos) > 1 else 0.0)

    return Robustez(
        nome=nome,
        media=float(retornos.mean()),
        erro=erro,
        pior=float(retornos.min()),
        melhor=float(retornos.max()),
        rebaixamento_medio=float(np.mean([m.rebaixamento for m in medidas])),
        operacoes_medio=float(np.mean([m.operacoes for m in medidas])),
        rodadas=len(medidas),
    ), medidas


def limiar_corrigido(n_comparacoes: int, alfa: float = 0.05) -> float:
    """
    Bonferroni: o limiar de significancia dividido pelo numero de comparacoes.

    Sem isto, testar cinco configuracoes faz uma parecer boa por acaso com
    probabilidade de ~23%. Foi o que quase aconteceu no ranking da V8, onde
    'B razao 1:1' cruzou o limiar bruto e nao sobreviveu a correcao.
    """
    return alfa / max(n_comparacoes, 1)


def tabela(resultados: list[Robustez], referencia: Metricas | None = None,
           titulo: str = "ROBUSTEZ") -> str:
    """Monta a tabela de saida, ordenada por media."""
    linhas = sorted(resultados, key=lambda r: r.media, reverse=True)
    largura = 96
    out = ["", "=" * largura, f"  {titulo}", "=" * largura, "",
           f"  {'Configuracao':<28} {'media':>9} {'erro':>7} {'t':>7} "
           f"{'pior':>9} {'melhor':>9} {'queda':>8} {'ops':>7}",
           "  " + "-" * (largura - 4)]

    for r in linhas:
        marca = " *" if r.distinguivel else ""
        out.append(
            f"  {r.nome:<28} {r.media*100:>+8.2f}% {r.erro*100:>6.2f} "
            f"{r.t:>+7.2f} {r.pior*100:>+8.1f}% {r.melhor*100:>+8.1f}% "
            f"{r.rebaixamento_medio*100:>7.1f}% {r.operacoes_medio:>7.0f}{marca}"
        )

    if referencia is not None:
        out.append("  " + "-" * (largura - 4))
        out.append(
            f"  {'* COMPRAR E SEGURAR':<28} {referencia.retorno*100:>+8.2f}% "
            f"{'—':>6} {'—':>7} {'—':>9} {'—':>9} "
            f"{referencia.rebaixamento*100:>7.1f}% {0:>7}"
        )

    out.append("  " + "-" * (largura - 4))
    reais = [r for r in linhas if r.distinguivel]
    if reais:
        out.append("  Distinguiveis de zero (|t| >= 2): "
                   + ", ".join(r.nome for r in reais))
    else:
        out.append("  NENHUMA e distinguivel de zero.")
    out.append("")
    out.append("  'pior' e 'melhor' sao rodadas da MESMA configuracao no MESMO")
    out.append("  periodo, mudando so a fase do ciclo — uma escolha sem")
    out.append("  significado economico. A distancia entre elas e a sorte.")
    out.append("=" * largura)
    return "\n".join(out)
