# -*- coding: utf-8 -*-
"""
Testes dos indicadores: valores conferidos na mao + causalidade.

O teste de causalidade e o mais importante do projeto. Ele recalcula cada
indicador usando SO os dados ate um instante e compara com o calculo feito
sobre a serie inteira. Se algum indicador olhasse adiante, os dois valores
divergiriam e o teste quebra.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from dados.indicadores import (atr, bollinger, ema, enriquecer, macd, rsi, sma,
                               true_range, wilder)


@pytest.fixture
def serie_simples() -> pl.Series:
    return pl.Series("x", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


@pytest.fixture
def ohlcv() -> pl.DataFrame:
    """Serie sintetica com tendencia, ruido e um gap — 600 barras."""
    rng = np.random.default_rng(42)
    n = 600
    passo = rng.normal(0.0002, 0.006, n)
    passo[300] = 0.05                       # gap, para o ATR ter o que capturar
    fech = 100.0 * np.exp(np.cumsum(passo))
    corpo = np.abs(rng.normal(0, 0.003, n)) * fech
    return pl.DataFrame({
        "ts": pl.datetime_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1) + timedelta(minutes=n),
            interval="1m", eager=True, closed="left",
        ),
        "abertura": fech - rng.normal(0, 0.002, n) * fech,
        "maxima": fech + corpo,
        "minima": fech - corpo,
        "fechamento": fech,
        "volume": rng.uniform(10, 1000, n),
    })


# ── Valores conferidos na mao ───────────────────────────────────────────────
def test_sma_valores(serie_simples):
    r = sma(serie_simples, 3).to_list()
    assert r[0] is None and r[1] is None
    assert r[2] == pytest.approx(2.0)       # (1+2+3)/3
    assert r[5] == pytest.approx(5.0)       # (4+5+6)/3


def test_wilder_e_recursao_de_1978():
    """Wilder: media[t] = media[t-1] + (x[t] - media[t-1]) / n."""
    s = pl.Series("x", [1.0, 2.0, 3.0, 4.0])
    r = wilder(s, 2).to_list()
    # min_samples=2, entao a primeira posicao e nula.
    # Com adjust=False e alpha=0.5, o EWM parte de x[0]=1:
    #   t1: 1 + (2-1)*0.5 = 1.5
    #   t2: 1.5 + (3-1.5)*0.5 = 2.25
    #   t3: 2.25 + (4-2.25)*0.5 = 3.125
    assert r[0] is None
    assert r[1] == pytest.approx(1.5)
    assert r[2] == pytest.approx(2.25)
    assert r[3] == pytest.approx(3.125)


def test_rsi_serie_so_de_alta_vale_100():
    """Sem nenhuma perda, o RSI correto e 100 — nao NaN nem divisao por zero."""
    s = pl.Series("x", [float(i) for i in range(1, 40)])
    r = rsi(s, 14).to_numpy()
    assert np.all(r[20:] == pytest.approx(100.0))


def test_rsi_serie_so_de_baixa_vale_zero():
    s = pl.Series("x", [float(i) for i in range(40, 1, -1)])
    r = rsi(s, 14).to_numpy()
    assert np.all(r[20:] == pytest.approx(0.0))


def test_rsi_serie_parada_e_neutro():
    """Preco travado: nem ganho nem perda. Indefinido pela formula; usamos 50."""
    s = pl.Series("x", [10.0] * 30)
    assert rsi(s, 14).to_numpy()[-1] == pytest.approx(50.0)


def test_rsi_fica_na_faixa(ohlcv):
    r = rsi(ohlcv["fechamento"], 14).to_numpy()
    validos = r[~np.isnan(r)]
    assert len(validos) > 500
    assert validos.min() >= 0.0 and validos.max() <= 100.0


def test_bollinger_usa_desvio_amostral(ohlcv):
    """
    ddof=1, nao 0. Esta e a convencao de Bollinger; bibliotecas que usam ddof=0
    entregam bandas mais estreitas e, portanto, mais toques falsos.
    """
    n = 20
    inf, med, sup = bollinger(ohlcv["fechamento"], n, 2.0)
    janela = ohlcv["fechamento"].to_numpy()[:n]
    esperado_med = janela.mean()
    esperado_desv = janela.std(ddof=1)
    assert med.to_numpy()[n - 1] == pytest.approx(esperado_med)
    assert sup.to_numpy()[n - 1] == pytest.approx(esperado_med + 2 * esperado_desv)
    assert inf.to_numpy()[n - 1] == pytest.approx(esperado_med - 2 * esperado_desv)
    # E o erro classico: ddof=0 daria outro numero.
    assert sup.to_numpy()[n - 1] != pytest.approx(
        esperado_med + 2 * janela.std(ddof=0)
    )


def test_bollinger_ordem_das_bandas(ohlcv):
    inf, med, sup = bollinger(ohlcv["fechamento"], 20, 2.0)
    i, m, s = inf.to_numpy(), med.to_numpy(), sup.to_numpy()
    ok = ~np.isnan(i)
    assert np.all(i[ok] <= m[ok]) and np.all(m[ok] <= s[ok])


def test_true_range_captura_gap(ohlcv):
    """No gap da barra 300 o TR tem de ser maior que a amplitude da vela."""
    tr = true_range(ohlcv["maxima"], ohlcv["minima"],
                    ohlcv["fechamento"]).to_numpy()
    amplitude = (ohlcv["maxima"] - ohlcv["minima"]).to_numpy()
    assert tr[300] > amplitude[300]


def test_atr_positivo(ohlcv):
    a = atr(ohlcv["maxima"], ohlcv["minima"], ohlcv["fechamento"], 14).to_numpy()
    validos = a[~np.isnan(a)]
    assert len(validos) > 500 and np.all(validos > 0)


def test_macd_histograma_e_a_diferenca(ohlcv):
    linha, sinal, hist = macd(ohlcv["fechamento"])
    d = (linha - sinal).to_numpy()
    h = hist.to_numpy()
    ok = ~np.isnan(h)
    assert np.allclose(d[ok], h[ok])


# ── Causalidade: o teste que importa ────────────────────────────────────────
@pytest.mark.parametrize("corte", [200, 350, 480, 599])
def test_nenhum_indicador_ve_o_futuro(ohlcv, corte):
    """
    Calcula sobre a serie inteira e sobre o prefixo ate `corte`. Os valores em
    `corte` tem de ser identicos: se algum indicador usasse dados posteriores,
    a versao completa saberia algo que a truncada nao sabe.
    """
    completo = enriquecer(ohlcv)
    truncado = enriquecer(ohlcv.head(corte + 1))

    derivadas = [c for c in completo.columns
                 if c not in ("ts", "abertura", "maxima", "minima",
                              "fechamento", "volume")]
    assert derivadas, "nenhuma coluna derivada — o teste nao testaria nada"

    for col in derivadas:
        a = completo[col].to_numpy()[corte]
        b = truncado[col].to_numpy()[corte]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12), (
            f"'{col}' difere em {corte}: inteiro={a!r} truncado={b!r}. "
            f"O indicador esta olhando para frente."
        )


def test_enriquecer_exige_as_colunas_base():
    with pytest.raises(ValueError, match="colunas ausentes"):
        enriquecer(pl.DataFrame({"ts": [1], "fechamento": [1.0]}))
