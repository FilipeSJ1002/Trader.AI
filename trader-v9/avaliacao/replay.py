# -*- coding: utf-8 -*-
"""
avaliacao/replay.py — o unico loop
===================================

Este arquivo e o coracao do desenho. Ele nao e "o backtest": e o loop que a
producao tambem vai usar. A diferenca entre medir e operar e QUAL Corretora
esta por baixo — papel ou Binance — e mais nada.

Enquanto houver um caminho de codigo so, backtest e producao nao podem divergir.
Na V6 havia dois e ninguem os tinha comparado; quando comparamos, discordavam.

A ordem dos passos importa
--------------------------
1. gatilhos primeiro    a corretora dispara stop/alvo com as barras que
                        passaram desde o ciclo anterior. Se a entrada viesse
                        antes, uma posicao ja liquidada ainda ocuparia vaga.
2. regime               o Oraculo diz de quem e o dia.
3. saida por regime     virou a chave, fecha o que esta do lado errado.
4. entrada              um sinal por ciclo, o de maior forca.

O passo 3 e o unico ponto onde o Oraculo mexe em posicao aberta, e e de
proposito: sem ele, uma virada de regime deixaria posicoes do motor errado
vivas ate baterem stop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from dados.visao import Historico
from execucao.papel import SemSaldo
from execucao.risco import Risco
from nucleo.protocolos import Corretora, Motor, Oraculo
from nucleo.tipos import Fechamento, Regime


@dataclass
class Resultado:
    """O que uma rodada de replay produz."""

    saldo_inicial: float
    saldo_final: float
    fechamentos: list[Fechamento]
    curva: list[tuple[datetime, float]] = field(default_factory=list)
    regimes: list[tuple[datetime, Regime]] = field(default_factory=list)
    parou_por: str = "fim do periodo"

    @property
    def retorno(self) -> float:
        return self.saldo_final / self.saldo_inicial - 1.0

    def __repr__(self) -> str:
        return (f"<Resultado {self.retorno*100:+.2f}% | "
                f"{len(self.fechamentos)} ops | {self.parou_por}>")


def replay(
    historicos: dict[str, Historico],
    motores: dict[Regime, Motor],
    oraculo: Oraculo,
    corretora: Corretora,
    risco: Risco,
    de: datetime,
    ate: datetime,
    a_cada: int = 15,
    referencia: str | None = None,
    fase: int = 0,
) -> Resultado:
    """
    Percorre o tempo chamando Oraculo, Motor e Corretora.

    `motores` mapeia Regime.BULL e Regime.BEAR aos especialistas.
    `referencia` e o ativo cujo contexto o Oraculo le (padrao: o primeiro).
    `fase` desloca a grade de avaliacao dentro do ciclo: com a_cada=15, a fase
    0 avalia :00/:15/:30/:45 e a fase 7 avalia :07/:22/:37/:52. E uma escolha
    sem significado economico nenhum — por isso avaliacao/robustez.py percorre
    todas elas: um resultado que dependa da fase e sorte, nao estrategia.
    """
    if not 0 <= fase < a_cada:
        raise ValueError(f"fase precisa estar em [0, {a_cada}): {fase}")
    simbolos = sorted(historicos)
    if not simbolos:
        raise ValueError("nenhum historico fornecido")
    referencia = referencia or simbolos[0]
    if referencia not in historicos:
        raise ValueError(f"referencia '{referencia}' nao esta nos historicos")

    # Linha do tempo comum: so instantes que existem em TODOS os ativos.
    passos = _linha_do_tempo(historicos, de, ate, a_cada, fase)
    if len(passos) < 2:
        raise ValueError(f"periodo curto demais: {len(passos)} passos")

    res = Resultado(saldo_inicial=corretora.saldo, saldo_final=corretora.saldo,
                    fechamentos=[])
    ultimo_indice: dict[str, int] = {}

    try:
        for ts in passos:
            indices = {s: historicos[s].indice_de(ts) for s in simbolos}

            # 1. A CORRETORA dispara stop e alvo — nao o replay.
            for pos in list(corretora.posicoes()):
                s = pos.symbol
                h = historicos[s]
                ini = ultimo_indice.get(s, indices[s]) + 1
                fim = indices[s]
                if fim >= ini:
                    corretora.varrer(s, *h.barras_entre(ini, fim))
            for s in simbolos:
                ultimo_indice[s] = indices[s]

            visoes = {s: historicos[s].em(indices[s]) for s in simbolos}

            # 2. De quem e o dia.
            regime = oraculo.regime(visoes[referencia])
            res.regimes.append((ts, regime))
            motor = motores.get(regime)

            # 3. Virou a chave: fecha o que ficou do lado errado.
            lado_certo = regime.lado
            for pos in list(corretora.posicoes()):
                if lado_certo is None or pos.lado is not lado_certo:
                    corretora.fechar(pos, "REGIME",
                                     preco=visoes[pos.symbol].fechamento,
                                     quando=ts)

            # 4. Entrada — um sinal por ciclo, o de maior forca.
            if motor is not None and risco.tem_espaco(len(corretora.posicoes())):
                ocupados = {p.symbol for p in corretora.posicoes()}
                melhor = None
                for s in simbolos:
                    if s in ocupados:
                        continue
                    sinal = motor.avaliar(visoes[s])
                    if sinal is not None and (melhor is None
                                              or sinal.forca > melhor.forca):
                        melhor = sinal

                if melhor is not None:
                    ordem = risco.dimensionar(melhor, visoes[melhor.symbol],
                                              corretora.saldo)
                    if ordem is not None:
                        pos = corretora.abrir(ordem)
                        # Se a protecao nao confirmar, a posicao nao pode ficar
                        # de pe. Foi exatamente este caminho que faltou na V6.
                        if not corretora.proteger(pos, ordem.stop, ordem.alvo):
                            corretora.fechar(pos, "SEM_PROTECAO",
                                             preco=melhor.preco_ref, quando=ts)

            res.curva.append(
                (ts, corretora.patrimonio({s: visoes[s].fechamento
                                           for s in simbolos}))
            )

        # Fecha o que sobrou, ao ultimo preco conhecido.
        ultimo = passos[-1]
        for pos in list(corretora.posicoes()):
            i = historicos[pos.symbol].indice_de(ultimo)
            corretora.fechar(pos, "FIM",
                             preco=historicos[pos.symbol].em(i).fechamento,
                             quando=ultimo)
        res.saldo_final = corretora.saldo

    except SemSaldo as e:
        # A conta acabou. Uma corretora real liquida a posicao nesse instante;
        # ela nao sobrevive para correr ate o fim do periodo.
        #
        # Medido em 02/09/2026: sem este corte, o fechamento final era
        # executado mesmo apos a quebra, e uma posicao vencedora aberta
        # RESSUSCITAVA a conta. A varredura de alavancagem chegou a reportar
        # "6 de 6 contas zeradas" com melhor rodada de +168,3% — impossivel.
        # Quebrar e o unico desfecho irreversivel do sistema, e o codigo tem
        # de trata-lo como tal.
        res.parou_por = f"conta zerada: {e}"
        res.saldo_final = 0.0

    res.fechamentos = list(corretora.fechamentos)
    return res


def _linha_do_tempo(
    historicos: dict[str, Historico], de: datetime, ate: datetime,
    a_cada: int, fase: int = 0
) -> list[datetime]:
    """
    Instantes de ciclo presentes em TODOS os ativos.

    A amostragem e feita sobre o RELOGIO, nao sobre a posicao no indice. Fatiar
    por posicao (`indice[::15]`) faz o passo depender de onde a serie comeca —
    e foi assim que, na V8, deslocar o inicio em sete minutos mudou o resultado
    de -4,90% para +8,24%. O relogio nao tem esse problema.
    """
    comum = None
    for h in historicos.values():
        idx = h.instantes
        recorte = idx[(idx >= np.datetime64(de, "us"))
                      & (idx <= np.datetime64(ate, "us"))]
        comum = recorte if comum is None else np.intersect1d(comum, recorte)
    if comum is None or len(comum) == 0:
        return []

    # Ancora na EPOCA, nao no inicio do periodo. Ancorar em comum[0] teria o
    # mesmo defeito da V8: os instantes de ciclo mudariam conforme a data de
    # inicio, e o resultado dependeria de um detalhe sem significado. Com a
    # epoca como ancora, pedir 2021-2026 ou 2025-2026 avalia exatamente os
    # mesmos relogios no trecho em comum.
    minutos = comum.astype("datetime64[m]").astype(np.int64)
    alinhados = comum[minutos % a_cada == fase % a_cada]
    return [t.astype("datetime64[us]").astype(datetime) for t in alinhados]
