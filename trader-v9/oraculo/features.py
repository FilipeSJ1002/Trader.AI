# -*- coding: utf-8 -*-
"""
oraculo/features.py — o contexto macro que o Oraculo enxerga
=============================================================

O Oraculo decide UMA vez por dia, olhando o quadro grande. Estas sao as
features, e todas saem de barras JA FECHADAS.

A garantia de nao-vazamento e reaproveitada, nao reescrita
----------------------------------------------------------
As features do dia D sao lidas de uma `VisaoDeMercado` construida no primeiro
minuto de D. Por construcao, essa visao so contem barras diarias fechadas ate
D-1 e barras de 4h fechadas ate as 00:00 de D — exatamente a informacao que
existiria de verdade na hora de decidir. Nao ha `shift` para lembrar de fazer,
nem janela para alinhar na mao: se a visao nao vaza, as features nao vazam.

O que entra
-----------
  tendencia    onde o preco esta em relacao as medias, medido em ATR (e nao em
               porcentagem, para ser comparavel entre ativos e entre epocas)
  momento      RSI e MACD diarios e de 4h
  volatilidade ATR sobre preco, e se ela esta subindo ou caindo
  posicao      onde o preco esta dentro das bandas de Bollinger
  mercado      a media e a dispersao entre os ativos — quando todos andam
               juntos, o regime e mais claro que quando cada um vai para um lado
  calendario   dia da semana em seno/cosseno (segunda e domingo sao vizinhos)

O que NAO entra, de proposito
-----------------------------
Nada de 1 minuto. O Oraculo nao decide entrada; se ele enxergasse o minuto,
voltariamos a ter uma rede decidindo operacao — que foi o que a V6 fez e nao se
sustentou na medicao.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from dados.visao import Historico
from nucleo.protocolos import VisaoDeMercado


def _dist_em_atr(visao: VisaoDeMercado, janela: int) -> float:
    """Distancia do preco a media de `janela` barras, em multiplos de ATR."""
    fech = visao.serie("fechamento")
    atr = visao.agora("atr")
    if len(fech) < janela or not np.isfinite(atr) or atr <= 0:
        return np.nan
    media = float(np.mean(fech[-janela:]))
    return float((fech[-1] - media) / atr)


def _inclinacao(visao: VisaoDeMercado, coluna: str, janela: int) -> float:
    """Variacao da coluna nas ultimas `janela` barras, normalizada."""
    s = visao.serie(coluna)
    if len(s) < janela + 1:
        return np.nan
    ini, fim = float(s[-janela - 1]), float(s[-1])
    if not (np.isfinite(ini) and np.isfinite(fim)) or ini == 0:
        return np.nan
    return float((fim - ini) / abs(ini))


def features_de_um_ativo(visao: VisaoDeMercado) -> dict[str, float]:
    """As features macro de um unico ativo, no instante da visao."""
    d, h4 = visao.diario, visao.h4
    fech_d = d.serie("fechamento")

    saida: dict[str, float] = {}

    # Tendencia diaria, em ATR
    for janela in (5, 20, 50, 200):
        saida[f"d_dist_sma{janela}"] = _dist_em_atr(d, janela)

    # Momento
    saida["d_rsi"] = d.agora("rsi")
    saida["d_macd_hist"] = (d.agora("macd_hist") / d.agora("atr")
                            if d.agora("atr") else np.nan)
    saida["d_bb_pos"] = d.agora("bb_pos")
    saida["h4_rsi"] = h4.agora("rsi")
    saida["h4_bb_pos"] = h4.agora("bb_pos")
    saida["h4_macd_hist"] = (h4.agora("macd_hist") / h4.agora("atr")
                             if h4.agora("atr") else np.nan)

    # Volatilidade e sua direcao
    preco = float(fech_d[-1]) if len(fech_d) else np.nan
    atr_d = d.agora("atr")
    saida["d_atr_rel"] = atr_d / preco if preco else np.nan
    saida["d_atr_incl"] = _inclinacao(d, "atr", 10)
    saida["d_bb_largura"] = d.agora("bb_largura")

    # Retornos recentes, em ATR (comparavel entre ativos)
    for janela in (1, 3, 7):
        if len(fech_d) > janela and atr_d and np.isfinite(atr_d):
            saida[f"d_ret{janela}"] = float(
                (fech_d[-1] - fech_d[-1 - janela]) / atr_d)
        else:
            saida[f"d_ret{janela}"] = np.nan

    return saida


def montar_tabela(
    historicos: dict[str, Historico],
    de: datetime,
    ate: datetime,
    referencia: str | None = None,
) -> pl.DataFrame:
    """
    Uma linha por dia, com as features do universo inteiro.

    As features de cada dia sao lidas no primeiro minuto dele — ou seja, com
    barras fechadas ate a vespera. E o que existiria de verdade na hora de
    decidir.
    """
    simbolos = sorted(historicos)
    referencia = referencia or simbolos[0]
    linhas = []

    dia = datetime(de.year, de.month, de.day)
    while dia <= ate:
        try:
            visoes = {s: historicos[s].em(historicos[s].indice_de(dia))
                      for s in simbolos}
        except ValueError:
            dia += timedelta(days=1)
            continue

        # O ativo de referencia entra com todas as features.
        linha: dict[str, float | datetime] = {"dia": dia}
        linha.update({f"ref_{k}": v
                      for k, v in features_de_um_ativo(visoes[referencia]).items()})

        # O universo entra resumido: media e dispersao. Quando todos os ativos
        # apontam para o mesmo lado, o regime e mais claro do que quando cada
        # um vai para um lado — e a dispersao captura isso.
        por_ativo = [features_de_um_ativo(visoes[s]) for s in simbolos]
        for chave in ("d_dist_sma20", "d_dist_sma50", "d_rsi", "d_ret1",
                      "d_ret7", "d_atr_rel"):
            valores = np.array([f[chave] for f in por_ativo], dtype=float)
            validos = valores[np.isfinite(valores)]
            linha[f"uni_{chave}_media"] = (float(validos.mean())
                                           if len(validos) else np.nan)
            linha[f"uni_{chave}_disp"] = (float(validos.std())
                                          if len(validos) > 1 else np.nan)

        # Calendario circular: segunda e domingo sao vizinhos, 0 e 6 nao sao.
        ang = 2 * np.pi * dia.weekday() / 7
        linha["cal_sin"] = float(np.sin(ang))
        linha["cal_cos"] = float(np.cos(ang))

        linhas.append(linha)
        dia += timedelta(days=1)

    return pl.DataFrame(linhas)


COLUNAS_NAO_FEATURE = ("dia", "regime", "y")


def colunas_de_feature(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in COLUNAS_NAO_FEATURE]
