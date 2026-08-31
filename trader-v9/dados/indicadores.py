# -*- coding: utf-8 -*-
"""
dados/indicadores.py — RSI, MACD, Bollinger e ATR, escritos aqui
=================================================================

Por que nao usar biblioteca
---------------------------
A V6 usou `pandas-ta` numa versao beta cujo Bollinger estava errado. O defeito
passou meses sem ser notado porque ninguem conferia o indicador — so o
resultado final, que ja vinha misturado com tudo. Cada funcao aqui tem quinze
linhas e um teste contra valores calculados na mao. E barato, e nunca mais
dependemos de uma convencao que alguem mudou entre versoes.

Convencoes, explicitas de proposito
-----------------------------------
Toda ambiguidade abaixo ja causou divergencia entre duas implementacoes da
mesma coisa em algum lugar deste projeto:

  RSI        suavizacao de Wilder (EWM com alpha=1/n, sem reajuste), que e a
             definicao original. Nao e media simples.
  Bollinger  desvio padrao AMOSTRAL (ddof=1). Varias bibliotecas usam
             populacional (ddof=0) e as bandas saem mais estreitas.
  ATR        Wilder sobre o True Range, mesma suavizacao do RSI.
  MACD       EMAs com reajuste (adjust=True), que e o padrao de mercado.

Causalidade
-----------
Toda funcao aqui olha SO para tras. `rolling_*` e `ewm_mean` do Polars sao
causais por definicao, e testes/test_indicadores.py prova isso comparando o
calculo sobre a serie inteira com o calculo sobre cada prefixo.
"""
from __future__ import annotations

import numpy as np
import polars as pl


# ── Blocos basicos ──────────────────────────────────────────────────────────
def sma(s: pl.Series, n: int) -> pl.Series:
    """Media movel simples. Null nas primeiras n-1 posicoes."""
    return s.rolling_mean(window_size=n)


def ema(s: pl.Series, n: int) -> pl.Series:
    """
    Media movel exponencial, convencao de mercado (adjust=True).

    Com adjust=True os primeiros valores sao corrigidos pelo peso acumulado,
    entao a serie nao comeca enviesada para o primeiro ponto.
    """
    return s.ewm_mean(span=n, adjust=True, min_samples=n)


def wilder(s: pl.Series, n: int) -> pl.Series:
    """
    Suavizacao de Wilder: EWM com alpha = 1/n e SEM reajuste.

    E o que Wilder descreveu em 1978 e o que RSI e ATR usam. Trocar isto por
    `ema(s, n)` muda os valores de forma sutil — o suficiente para dois codigos
    discordarem sem que ninguem entenda por que.
    """
    return s.ewm_mean(alpha=1.0 / n, adjust=False, min_samples=n)


# ── Indicadores ─────────────────────────────────────────────────────────────
def rsi(fechamento: pl.Series, n: int = 14) -> pl.Series:
    """
    Indice de Forca Relativa (Wilder), de 0 a 100.

    Acima de 70 costuma ser lido como esticado para cima; abaixo de 30, para
    baixo. Os limiares vivem no config, nao aqui.

    Quando nao ha perda alguma na janela, a divisao seria por zero — o RSI
    correto nesse caso e 100, e e isso que devolvemos.
    """
    delta = fechamento.diff()
    ganho = pl.Series(np.where(delta.to_numpy() > 0, delta.to_numpy(), 0.0))
    perda = pl.Series(np.where(delta.to_numpy() < 0, -delta.to_numpy(), 0.0))

    media_ganho = wilder(ganho, n).to_numpy()
    media_perda = wilder(perda, n).to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = media_ganho / media_perda
        valores = 100.0 - (100.0 / (1.0 + rs))

    # Sem perdas na janela -> RSI 100. Sem ganhos -> RSI 0.
    valores = np.where((media_perda == 0) & (media_ganho > 0), 100.0, valores)
    valores = np.where((media_ganho == 0) & (media_perda > 0), 0.0, valores)
    # Serie totalmente parada: nem ganho nem perda. Indefinido -> 50 (neutro).
    valores = np.where((media_ganho == 0) & (media_perda == 0), 50.0, valores)
    # A primeira posicao nunca tem delta.
    valores[0] = np.nan
    valores[np.isnan(media_ganho)] = np.nan

    return pl.Series("rsi", valores)


def macd(
    fechamento: pl.Series,
    rapida: int = 12,
    lenta: int = 26,
    sinal: int = 9,
) -> tuple[pl.Series, pl.Series, pl.Series]:
    """
    MACD, linha de sinal e histograma.

    Devolve (macd, sinal, histograma). O cruzamento do histograma por zero e o
    gatilho classico: negativo -> positivo indica virada para cima.
    """
    linha = ema(fechamento, rapida) - ema(fechamento, lenta)
    linha_sinal = ema(linha, sinal)
    histograma = linha - linha_sinal
    return (
        linha.rename("macd"),
        linha_sinal.rename("macd_sinal"),
        histograma.rename("macd_hist"),
    )


def bollinger(
    fechamento: pl.Series,
    n: int = 20,
    k: float = 2.0,
) -> tuple[pl.Series, pl.Series, pl.Series]:
    """
    Bandas de Bollinger: (inferior, media, superior).

    Desvio padrao AMOSTRAL (ddof=1) — a convencao de Bollinger. Bibliotecas que
    usam ddof=0 produzem bandas mais estreitas e, portanto, mais toques. Foi
    exatamente esse tipo de discrepancia que corrompeu as features da V6.
    """
    media = sma(fechamento, n)
    desvio = fechamento.rolling_std(window_size=n, ddof=1)
    return (
        (media - k * desvio).rename("bb_inf"),
        media.rename("bb_medio"),
        (media + k * desvio).rename("bb_sup"),
    )


def largura_bollinger(
    inferior: pl.Series, medio: pl.Series, superior: pl.Series
) -> pl.Series:
    """
    (superior - inferior) / medio. Mede compressao de volatilidade.

    Util para o Oraculo: faixa estreita costuma preceder movimento amplo, e o
    valor e comparavel entre ativos de precos muito diferentes.
    """
    return ((superior - inferior) / medio).rename("bb_largura")


def true_range(
    maxima: pl.Series, minima: pl.Series, fechamento: pl.Series
) -> pl.Series:
    """
    True Range: o maior entre a amplitude da vela e os saltos desde o
    fechamento anterior. Captura gaps, que a amplitude simples ignora.
    """
    fech_ant = fechamento.shift(1)
    a = (maxima - minima).to_numpy()
    b = np.abs((maxima - fech_ant).to_numpy())
    c = np.abs((minima - fech_ant).to_numpy())
    return pl.Series("tr", np.nanmax(np.vstack([a, b, c]), axis=0))


def atr(
    maxima: pl.Series, minima: pl.Series, fechamento: pl.Series, n: int = 14
) -> pl.Series:
    """
    Average True Range (Wilder). A unidade natural para dimensionar stop e alvo:
    um stop de 1,5 ATR significa a mesma coisa no BTC e no XRP, o que uma
    porcentagem fixa nao consegue.
    """
    return wilder(true_range(maxima, minima, fechamento), n).rename("atr")


# ── Aplicacao em lote ───────────────────────────────────────────────────────
#
# Uma unica funcao monta TODAS as colunas derivadas. Motores e Oraculo leem
# dessas colunas pelo nome e nunca recalculam nada — se dois componentes
# calculassem o mesmo indicador por conta propria, voltariamos ao problema de
# duas implementacoes que divergem.

COLUNAS_BASE = ("abertura", "maxima", "minima", "fechamento", "volume")


def enriquecer(df: pl.DataFrame, cfg: dict | None = None) -> pl.DataFrame:
    """
    Acrescenta as colunas de indicadores a um frame OHLCV.

    Espera as colunas de COLUNAS_BASE mais `ts`. Devolve um frame novo — o
    original nao e tocado.
    """
    c = {"rsi": 14, "macd_rapida": 12, "macd_lenta": 26, "macd_sinal": 9,
         "bb_n": 20, "bb_k": 2.0, "atr": 14, **(cfg or {})}

    faltando = [x for x in COLUNAS_BASE if x not in df.columns]
    if faltando:
        raise ValueError(f"colunas ausentes no frame: {faltando}")

    fech, maxi, mini = df["fechamento"], df["maxima"], df["minima"]
    linha, sinal_l, hist = macd(fech, c["macd_rapida"], c["macd_lenta"],
                                c["macd_sinal"])
    bb_i, bb_m, bb_s = bollinger(fech, c["bb_n"], c["bb_k"])

    return df.with_columns([
        rsi(fech, c["rsi"]),
        linha, sinal_l, hist,
        bb_i, bb_m, bb_s,
        largura_bollinger(bb_i, bb_m, bb_s),
        atr(maxi, mini, fech, c["atr"]),
        # Posicao do preco dentro das bandas: 0 na inferior, 1 na superior.
        # Comparavel entre ativos, ao contrario da distancia em dolares.
        (((fech - bb_i) / (bb_s - bb_i)).rename("bb_pos")),
    ])
