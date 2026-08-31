# -*- coding: utf-8 -*-
"""
Testes da VisaoDeMercado.

O que precisa ficar provado:
  1. a visao contem SO passado — nem por acidente ha dado posterior nela
  2. as barras de 4h e 1D so aparecem depois de FECHADAS
  3. Historico.em(i) devolve exatamente o que enriquecer() daria no prefixo
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from dados.visao import Historico, VisaoDeMercado
from nucleo.protocolos import VisaoDeMercado as ProtocoloVisao


@pytest.fixture
def historico() -> Historico:
    """Cinco dias de barras de 1 minuto — o bastante para ter diarias fechadas."""
    rng = np.random.default_rng(7)
    n = 5 * 24 * 60
    fech = 100.0 * np.exp(np.cumsum(rng.normal(0.00005, 0.0015, n)))
    corpo = np.abs(rng.normal(0, 0.001, n)) * fech
    df = pl.DataFrame({
        "ts": pl.datetime_range(
            datetime(2024, 3, 1), datetime(2024, 3, 1) + timedelta(minutes=n),
            interval="1m", eager=True, closed="left",
        ),
        "abertura": fech,
        "maxima": fech + corpo,
        "minima": fech - corpo,
        "fechamento": fech,
        "volume": rng.uniform(1, 100, n),
    })
    return Historico("TESTE", df)


def test_satisfaz_o_protocolo(historico):
    assert isinstance(historico.em(500), ProtocoloVisao)


def test_visao_termina_no_instante_atual(historico):
    for i in (200, 1500, 4000):
        v = historico.em(i)
        assert v.barras == i + 1
        assert len(v.serie("rsi")) == i + 1


def test_visao_nao_carrega_dado_do_futuro(historico):
    """
    A prova estrutural: a visao em i e a visao em i+500 tem de coincidir em
    todo o trecho comum. Se a primeira guardasse algo do futuro, algum valor
    ja calculado mudaria quando o futuro chegasse.
    """
    cedo, tarde = historico.em(2000), historico.em(2500)
    for col in ("fechamento", "rsi", "macd_hist", "bb_pos", "atr"):
        a = cedo.serie(col)
        b = tarde.serie(col)[: len(a)]
        ok = ~(np.isnan(a) | np.isnan(b))
        assert np.allclose(a[ok], b[ok], rtol=1e-12, atol=1e-12), (
            f"'{col}' mudou retroativamente — ha vazamento de futuro"
        )


def test_agora_e_o_ultimo_valor(historico):
    v = historico.em(3000)
    assert v.agora("fechamento") == pytest.approx(v.serie("fechamento")[-1])
    assert v.antes("fechamento") == pytest.approx(v.serie("fechamento")[-2])
    assert v.antes("fechamento", 5) == pytest.approx(v.serie("fechamento")[-6])


def test_diaria_so_mostra_barras_fechadas(historico):
    """
    As 14h do dia 3, a diaria do dia 3 ainda esta se formando. Usa-la seria
    vazamento. So podem aparecer as barras dos dias 1 e 2.
    """
    i = historico.indice_de(datetime(2024, 3, 3, 14, 0))
    v = historico.em(i)
    ultimo_dia = v.diario.ts

    assert v.diario.barras == 2, (
        f"esperava 2 diarias fechadas (dias 1 e 2), veio {v.diario.barras}"
    )
    assert ultimo_dia == datetime(2024, 3, 3, 14, 0)   # o relogio, nao a barra


def test_h4_so_mostra_barras_fechadas(historico):
    """As 09h30, as barras de 4h fechadas sao 00-04 e 04-08. A de 08-12, nao."""
    i = historico.indice_de(datetime(2024, 3, 2, 9, 30))
    v = historico.em(i)
    esperado = 6 + 2          # 6 do dia 1 (24h/4h) + 2 do dia 2
    assert v.h4.barras == esperado


def test_barra_de_4h_aparece_exatamente_ao_fechar(historico):
    """Na fronteira: as 07h59 a barra 04-08 nao fechou; as 08h00, fechou."""
    antes = historico.em(historico.indice_de(datetime(2024, 3, 2, 7, 59)))
    depois = historico.em(historico.indice_de(datetime(2024, 3, 2, 8, 0)))
    assert depois.h4.barras == antes.h4.barras + 1


def test_coluna_inexistente_da_erro_util(historico):
    with pytest.raises(KeyError, match="nao existe"):
        historico.em(100).serie("indicador_que_nao_existe")


def test_pronta_fica_falsa_no_comeco(historico):
    assert not historico.em(5).pronta
    assert historico.em(3000).pronta


def test_indice_de_e_a_ultima_barra_ate_o_instante(historico):
    quando = datetime(2024, 3, 2, 12, 0)
    i = historico.indice_de(quando)
    assert historico.em(i).ts <= quando
    assert historico.em(i + 1).ts > quando


def test_rejeita_timestamps_duplicados():
    df = pl.DataFrame({
        "ts": [datetime(2024, 1, 1), datetime(2024, 1, 1)],
        "abertura": [1.0, 1.0], "maxima": [1.0, 1.0],
        "minima": [1.0, 1.0], "fechamento": [1.0, 1.0], "volume": [1.0, 1.0],
    })
    with pytest.raises(ValueError, match="duplicados"):
        Historico("X", df)


def test_visao_e_imutavel(historico):
    v = historico.em(100)
    with pytest.raises(Exception):
        v.ts = datetime(2030, 1, 1)          # type: ignore[misc]
