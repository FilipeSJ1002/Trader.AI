# -*- coding: utf-8 -*-
"""
dados/atualizar.py — completa o histórico local com o que já aconteceu
=======================================================================

Os parquets em data/ são um retrato: eles terminam no dia em que foram
baixados. Um sistema que decide hoje precisa de dados de hoje.

Este módulo busca na corretora as barras entre o fim do arquivo e agora, e as
devolve concatenadas em memória. Não reescreve os parquets — o histórico de
pesquisa permanece congelado e reproduzível, e a atualização vale só para a
execução corrente.

A checagem que este arquivo faz por você
-----------------------------------------
Se a defasagem for grande demais para ser coberta, ou se a corretora devolver
menos do que o pedido, `atualizar` avisa e devolve o que conseguiu. Quem chama
decide se opera. O que NÃO pode acontecer é decidir com dados velhos sem saber
disso — foi o que o primeiro teste ao vivo quase fez, com 40 dias de defasagem
passando despercebidos.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import numpy as np
import polars as pl

LIMITE_POR_CHAMADA = 1500        # maximo de candles por requisicao na Binance
MINUTOS_POR_INTERVALO = {"1m": 1, "5m": 5, "15m": 15, "1h": 60,
                         "4h": 240, "1d": 1440}
PAUSA = 0.25                     # respeita o limite de requisicoes


def _agora() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def defasagem(historico) -> timedelta:
    """Quanto tempo separa o fim do arquivo do instante atual."""
    return _agora() - historico.fim


def baixar_desde(api, symbol: str, inicio: datetime, intervalo: str = "1m",
                 log=print) -> pl.DataFrame | None:
    """
    Candles entre `inicio` e agora, paginados.

    O intervalo padrao e 1m por compatibilidade com os parquets de
    pesquisa. Para operacao use "4h": as features do oraculo saem so das
    visoes diaria e de 4h, e o resultado e identico com 240x menos dados.
    """
    passo_ms = MINUTOS_POR_INTERVALO[intervalo] * 60_000
    cursor = int(inicio.replace(tzinfo=timezone.utc).timestamp() * 1000)
    fim = int(_agora().replace(tzinfo=timezone.utc).timestamp() * 1000)
    linhas = []

    while cursor < fim:
        try:
            lote = api.futures_klines(symbol=symbol, interval=intervalo,
                                      startTime=cursor,
                                      limit=LIMITE_POR_CHAMADA)
        except Exception as e:
            log(f"    [erro] {symbol}: {e}")
            break
        if not lote:
            break
        linhas += lote
        proximo = int(lote[-1][0]) + passo_ms
        if proximo <= cursor:
            break
        cursor = proximo
        if len(lote) < LIMITE_POR_CHAMADA:
            break
        time.sleep(PAUSA)

    if not linhas:
        return None

    return pl.DataFrame({
        "ts": [datetime.utcfromtimestamp(int(l[0]) / 1000) for l in linhas],
        "abertura": [float(l[1]) for l in linhas],
        "maxima": [float(l[2]) for l in linhas],
        "minima": [float(l[3]) for l in linhas],
        "fechamento": [float(l[4]) for l in linhas],
        "volume": [float(l[5]) for l in linhas],
    }).unique(subset="ts", keep="last").sort("ts")


def passo_do_historico(historico) -> str:
    """
    Descobre o intervalo das barras pelo espacamento mediano.

    O mesmo codigo serve para os parquets de pesquisa (1 minuto) e para os
    de operacao (4 horas), sem que ninguem precise configurar isso.
    """
    ts = historico.instantes
    if len(ts) < 3:
        return "1m"
    minutos = int(np.median(np.diff(ts[-200:]).astype("timedelta64[m]")
                            .astype(int)))
    for nome, m in MINUTOS_POR_INTERVALO.items():
        if m == minutos:
            return nome
    return "1m"


def atualizar(api, fonte, ativos: list[str], cfg: dict | None = None,
              tolerancia_min: int | None = None,
              log=print) -> tuple[dict, bool]:
    """
    Devolve (historicos atualizados, tudo_em_dia).

    `tolerancia_min` sai do proprio intervalo das barras quando nao e dado:
    numa base de 4h, uma barra de ate 4h de idade e o normal, nao atraso.
    """
    from dados.visao import Historico

    saida, em_dia = {}, True
    for symbol in ativos:
        h = fonte.historico(symbol)
        intervalo = passo_do_historico(h)
        tol = timedelta(minutes=(tolerancia_min if tolerancia_min is not None
                                 else MINUTOS_POR_INTERVALO[intervalo] + 30))
        atraso = defasagem(h)

        if atraso <= tol:
            log(f"  {symbol}: em dia ({intervalo}, defasagem "
                f"{atraso.total_seconds()/60:.0f} min)")
            saida[symbol] = h
            continue

        log(f"  {symbol}: defasado {atraso.days}d {atraso.seconds//3600}h — "
            f"buscando na corretora...")
        novo = baixar_desde(api, symbol, h.fim + timedelta(minutes=1),
                            intervalo=intervalo, log=log)
        if novo is None or novo.is_empty():
            log(f"    [aviso] {symbol}: nada retornado; segue com o arquivo")
            saida[symbol] = h
            em_dia = False
            continue

        antigo = pl.DataFrame({
            "ts": h.instantes,
            "abertura": h.em(len(h) - 1).serie("abertura"),
            "maxima": h.em(len(h) - 1).serie("maxima"),
            "minima": h.em(len(h) - 1).serie("minima"),
            "fechamento": h.em(len(h) - 1).serie("fechamento"),
            "volume": h.em(len(h) - 1).serie("volume"),
        })
        # O parquet historico e o download podem usar precisoes de tempo
        # diferentes (ns e us). Concatenar sem alinhar levanta SchemaError.
        novo = novo.with_columns(pl.col("ts").cast(antigo["ts"].dtype))
        junto = (pl.concat([antigo, novo])
                 .unique(subset="ts", keep="last").sort("ts"))
        atualizado = Historico(symbol, junto, (cfg or {}).get("indicadores"))
        saida[symbol] = atualizado

        resto = defasagem(atualizado)
        if resto > tol:
            log(f"    [aviso] {symbol}: ainda defasado "
                f"{resto.total_seconds()/60:.0f} min apos a atualizacao")
            em_dia = False
        else:
            log(f"    [ok] {symbol}: +{len(novo):,} barras, agora ate "
                f"{atualizado.fim:%Y-%m-%d %H:%M}")

    return saida, em_dia
