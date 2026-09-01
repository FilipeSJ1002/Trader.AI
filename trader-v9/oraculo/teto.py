# -*- coding: utf-8 -*-
"""
oraculo/teto.py — o oraculo que trapaceia, e os controles que o interpretam
============================================================================

O sistema inteiro e limitado pelo Oraculo: ele nunca renderia mais do que
renderia se acertasse SEMPRE. Este arquivo constroi esse oraculo perfeito para
descobrir o numero antes de gastar meses treinando um classificador.

E o unico componente do projeto autorizado a ver o futuro, e e deliberado.

Tres oraculos, e por que os tres importam
-----------------------------------------
  OraculoPerfeito   ve o futuro e sempre acerta -> o TETO
  OraculoMoeda      cara ou coroa                -> o CHAO
  OraculoFixo       sempre o mesmo lado          -> o controle sem previsao

Sozinho, o teto nao diz nada. Se o perfeito render +40% e a moeda render +38%,
o que produziu o resultado foram os motores, nao a previsao de regime — e o
Oraculo real, que sera pior que o perfeito, nao tem espaco para agregar valor.
A distancia entre o teto e o chao e o tamanho do premio que a Sprint 2 disputa.

O que "perfeito" quer dizer aqui
--------------------------------
Saber, no comeco do dia, se o mercado vai fechar em alta ou em baixa. Nao e
onisciencia total (nao escolhe ativo, nem hora, nem preco) — e exatamente o
poder que o Oraculo da arquitetura tem, levado ao acerto de 100%. Um teto medido
com poder ALEM do que o oraculo real tera seria um teto inalcancavel por
construcao, e nao serviria para decidir nada.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import polars as pl

from dados.visao import Historico
from nucleo.protocolos import VisaoDeMercado
from nucleo.tipos import Regime


class OraculoPerfeito:
    """
    Sabe de antemao a direcao do dia. O limite superior do projeto.

    Constroi um mapa dia -> regime a partir do retorno diario MEDIO do universo
    (peso igual). Se o mercado subiu no dia, o dia era do Bull; se caiu, do Bear.
    """

    def __init__(
        self,
        historicos: dict[str, Historico],
        limiar_fora: float = 0.0,
    ):
        """
        `limiar_fora` e o movimento minimo, em fracao, para valer operar. Com
        0,0 o oraculo sempre escolhe um lado. Com 0,01, dias que andaram menos
        de 1% viram FORA — e um teto ainda mais alto, porque poder pular os
        dias mornos so ajuda.
        """
        if not historicos:
            raise ValueError("sem historicos, nao ha o que prever")
        self.nome = f"perfeito(fora<{limiar_fora:.1%})"
        self.limiar_fora = float(limiar_fora)
        self._mapa = self._mapear(historicos, self.limiar_fora)

    @staticmethod
    def _mapear(historicos: dict[str, Historico],
                limiar: float) -> dict[date, Regime]:
        """Retorno diario medio do universo -> regime de cada dia."""
        series = []
        for sym, h in historicos.items():
            ts, _, _ = h.barras_entre(0, len(h) - 1)
            fech = h.em(len(h) - 1).serie("fechamento")
            diario = (
                pl.DataFrame({"ts": ts, "fechamento": fech})
                .group_by_dynamic("ts", every="1d", closed="left", label="left")
                .agg(pl.col("fechamento").last())
                .drop_nulls()
            )
            r = diario["fechamento"].pct_change()
            series.append(pl.DataFrame({"dia": diario["ts"],
                                        sym: r}).drop_nulls())

        junto = series[0]
        for s in series[1:]:
            junto = junto.join(s, on="dia", how="inner")

        colunas = [c for c in junto.columns if c != "dia"]
        medio = junto.select(colunas).mean_horizontal().to_numpy()
        dias = junto["dia"].to_numpy()

        mapa: dict[date, Regime] = {}
        for d, r in zip(dias, medio):
            dia = d.astype("datetime64[us]").astype(datetime).date()
            if not np.isfinite(r) or abs(r) < limiar:
                mapa[dia] = Regime.FORA
            else:
                mapa[dia] = Regime.BULL if r > 0 else Regime.BEAR
        return mapa

    def regime(self, visao: VisaoDeMercado) -> Regime:
        """
        O regime do dia corrente — sabido de antemao.

        Dias fora do mapa (o primeiro da serie, feriados de dados) viram FORA:
        na duvida, nao operar, para o teto nao ser inflado por adivinhacao.
        """
        return self._mapa.get(visao.ts.date(), Regime.FORA)

    @property
    def dias_mapeados(self) -> int:
        return len(self._mapa)

    def distribuicao(self) -> dict[str, int]:
        """Quantos dias de cada regime — util para ver se o mapa faz sentido."""
        d: dict[str, int] = {}
        for r in self._mapa.values():
            d[r.value] = d.get(r.value, 0) + 1
        return d


class OraculoMoeda:
    """
    Cara ou coroa, uma vez por dia. O CHAO.

    Existe para responder a pergunta que o teto sozinho nao responde: o
    resultado veio da previsao de regime ou dos motores? Se o perfeito nao
    superar a moeda com folga, prever regime nao vale nada e a Sprint 2 morre.
    """

    def __init__(self, semente: int = 0, permitir_fora: bool = False):
        self.nome = f"moeda(s={semente})"
        self._rng = np.random.default_rng(semente)
        self._opcoes = ([Regime.BULL, Regime.BEAR, Regime.FORA]
                        if permitir_fora else [Regime.BULL, Regime.BEAR])
        self._cache: dict[date, Regime] = {}

    def regime(self, visao: VisaoDeMercado) -> Regime:
        """Uma decisao por DIA — nao por ciclo, senao viraria ruido puro."""
        dia = visao.ts.date()
        if dia not in self._cache:
            self._cache[dia] = self._opcoes[self._rng.integers(len(self._opcoes))]
        return self._cache[dia]


class OraculoFixo:
    """Sempre o mesmo regime. O controle sem previsao nenhuma."""

    def __init__(self, regime: Regime):
        self.nome = f"sempre-{regime.value}"
        self._r = regime

    def regime(self, visao: VisaoDeMercado) -> Regime:
        return self._r
