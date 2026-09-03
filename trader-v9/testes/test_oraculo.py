# -*- coding: utf-8 -*-
"""
Testes do oraculo que trapaceia e da varredura de robustez.

O oraculo perfeito e o UNICO componente autorizado a ver o futuro. Estes testes
provam que ele de fato acerta — se ele errasse, o teto medido seria menor que o
teto real e a decisao da Sprint 2 sairia errada por defeito, nao por evidencia.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from avaliacao.metricas import comprar_e_segurar, medir
from avaliacao.replay import replay
from avaliacao.robustez import limiar_corrigido, varrer_fases
from dados.visao import Historico
from execucao.papel import CorretoraPapel
from execucao.risco import Risco
from motores.bear import MotorBear
from motores.bull import MotorBull
from nucleo.protocolos import Oraculo
from nucleo.tipos import Regime
from oraculo.teto import OraculoFixo, OraculoMoeda, OraculoPerfeito


def _historico(passos: np.ndarray, symbol: str) -> Historico:
    n = len(passos)
    fech = 100.0 * np.exp(np.cumsum(passos))
    corpo = np.abs(passos) * fech * 0.6 + fech * 2e-4
    df = pl.DataFrame({
        "ts": pl.datetime_range(datetime(2024, 1, 1),
                                datetime(2024, 1, 1) + timedelta(minutes=n),
                                interval="1m", eager=True, closed="left"),
        "abertura": fech, "maxima": fech + corpo, "minima": fech - corpo,
        "fechamento": fech, "volume": np.full(n, 100.0),
    })
    return Historico(symbol, df)


@pytest.fixture
def mercado() -> dict[str, Historico]:
    rng = np.random.default_rng(21)
    n = 40 * 24 * 60
    return {
        "AAA": _historico(rng.normal(0.00005, 0.0012, n), "AAA"),
        "BBB": _historico(rng.normal(0.00003, 0.0014, n), "BBB"),
    }


@pytest.fixture
def alternado() -> dict[str, Historico]:
    """Dias que sobem e caem alternadamente — direcao conhecida de antemao."""
    n_dias, por_dia = 40, 24 * 60
    passos = np.concatenate([
        np.full(por_dia, (0.0004 if d % 2 == 0 else -0.0004))
        for d in range(n_dias)
    ])
    return {"AAA": _historico(passos, "AAA")}


# ── O oraculo perfeito ──────────────────────────────────────────────────────
def test_satisfaz_o_protocolo(mercado):
    assert isinstance(OraculoPerfeito(mercado), Oraculo)
    assert isinstance(OraculoMoeda(), Oraculo)
    assert isinstance(OraculoFixo(Regime.BULL), Oraculo)


def test_perfeito_acerta_a_direcao_do_dia(alternado):
    """
    Num mercado que sobe nos dias pares e cai nos impares, o oraculo perfeito
    tem de dizer BULL nos pares e BEAR nos impares. Se errar, o teto sai menor
    que o teto real e a decisao da Sprint 2 sai errada por defeito.
    """
    h = alternado["AAA"]
    o = OraculoPerfeito(alternado)
    acertos = total = 0
    for dia in range(2, 39):
        i = h.indice_de(datetime(2024, 1, 1) + timedelta(days=dia, hours=12))
        v = h.em(i)
        esperado = Regime.BULL if dia % 2 == 0 else Regime.BEAR
        total += 1
        acertos += (o.regime(v) is esperado)
    assert acertos == total, f"errou {total - acertos} de {total} dias"


def test_perfeito_e_estavel_dentro_do_dia(mercado):
    """O regime e do DIA: nao pode mudar entre um ciclo e outro do mesmo dia."""
    h = mercado["AAA"]
    o = OraculoPerfeito(mercado)
    base = datetime(2024, 1, 20)
    regimes = {o.regime(h.em(h.indice_de(base + timedelta(hours=hh))))
               for hh in (1, 6, 12, 18, 23)}
    assert len(regimes) == 1


def test_limiar_fora_cria_dias_de_ninguem(mercado):
    sem = OraculoPerfeito(mercado, limiar_fora=0.0)
    com = OraculoPerfeito(mercado, limiar_fora=0.05)
    assert sem.distribuicao().get("FORA", 0) < com.distribuicao().get("FORA", 0)


def test_dia_desconhecido_vira_fora(mercado):
    """Na duvida, nao operar — para o teto nao ser inflado por adivinhacao."""
    h = mercado["AAA"]
    o = OraculoPerfeito(mercado)
    o._mapa.clear()
    assert o.regime(h.em(30_000)) is Regime.FORA


def test_moeda_decide_uma_vez_por_dia(mercado):
    h = mercado["AAA"]
    o = OraculoMoeda(semente=3)
    base = datetime(2024, 1, 20)
    regimes = {o.regime(h.em(h.indice_de(base + timedelta(hours=hh))))
               for hh in (1, 6, 12, 18, 23)}
    assert len(regimes) == 1


def test_moeda_com_a_mesma_semente_repete(mercado):
    h = mercado["AAA"]
    a, b = OraculoMoeda(semente=5), OraculoMoeda(semente=5)
    for i in (20_000, 30_000, 40_000):
        assert a.regime(h.em(i)) is b.regime(h.em(i))


def test_perfeito_rende_mais_que_a_moeda(alternado):
    """
    O teste que da sentido ao teto. Num mercado de direcao conhecida, saber a
    direcao TEM de valer mais que sortear. Se nao valesse aqui, o desenho
    inteiro dos tres pilares estaria errado.
    """
    def rodar(oraculo):
        c = CorretoraPapel(5000.0)
        r = replay(alternado,
                   {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
                   oraculo, c, Risco(prazo_atr="h4"),
                   datetime(2024, 1, 20), datetime(2024, 2, 8), a_cada=15)
        return medir(r)

    perfeito = rodar(OraculoPerfeito(alternado))
    moeda = rodar(OraculoMoeda(semente=1))
    assert perfeito.operacoes > 0, "o teto nao operou — o teste nao provou nada"
    assert perfeito.retorno > moeda.retorno, (
        f"perfeito {perfeito.retorno:.4f} nao superou moeda {moeda.retorno:.4f}"
    )


# ── Robustez ────────────────────────────────────────────────────────────────
def test_varredura_roda_todas_as_fases(mercado):
    def montar():
        return (mercado,
                {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
                OraculoFixo(Regime.BULL),
                CorretoraPapel(5000.0),
                Risco(prazo_atr="h4"),
                "AAA")

    r, medidas = varrer_fases("teste", montar,
                              datetime(2024, 1, 20), datetime(2024, 2, 8),
                              a_cada=5)
    assert r.rodadas == 5 and len(medidas) == 5
    assert r.pior <= r.media <= r.melhor
    assert r.erro >= 0


def test_cada_fase_comeca_com_corretora_limpa(mercado):
    """
    `montar` tem de ser uma FUNCAO. Se as rodadas partilhassem a corretora, o
    saldo de uma contaminaria a outra e a varredura mediria acumulo, nao fase.
    """
    def montar():
        return (mercado,
                {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
                OraculoFixo(Regime.BULL),
                CorretoraPapel(5000.0),
                Risco(prazo_atr="h4"),
                "AAA")

    _, medidas = varrer_fases("teste", montar,
                              datetime(2024, 1, 20), datetime(2024, 2, 8),
                              a_cada=3)
    # Todas partiram de 5000: nenhum saldo final pode ser identico ao de outra
    # rodada por acumulo, e todos tem de ser da mesma ordem de grandeza.
    for m in medidas:
        assert 0 < m.saldo_final < 20_000


def test_fase_fora_da_faixa_e_recusada(mercado):
    with pytest.raises(ValueError, match="fase precisa estar"):
        replay(mercado, {Regime.BULL: MotorBull()}, OraculoFixo(Regime.BULL),
               CorretoraPapel(5000.0), Risco(),
               datetime(2024, 1, 20), datetime(2024, 2, 8),
               a_cada=15, fase=15)


def test_limiar_bonferroni():
    assert limiar_corrigido(1) == pytest.approx(0.05)
    assert limiar_corrigido(5) == pytest.approx(0.01)


# ── Metricas ────────────────────────────────────────────────────────────────
def test_rebaixamento_e_negativo_ou_zero(mercado):
    c = CorretoraPapel(5000.0)
    r = replay(mercado, {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
               OraculoFixo(Regime.BULL), c, Risco(prazo_atr="h4"),
               datetime(2024, 1, 20), datetime(2024, 2, 8))
    assert medir(r).rebaixamento <= 0


def test_comprar_e_segurar_bate_com_a_variacao_do_preco(mercado):
    m = comprar_e_segurar(mercado, datetime(2024, 1, 20),
                          datetime(2024, 2, 8), 5000.0)
    esperado = np.mean([
        h.em(h.indice_de(datetime(2024, 2, 8))).fechamento
        / h.em(h.indice_de(datetime(2024, 1, 20))).fechamento
        for h in mercado.values()
    ]) - 1
    assert m.retorno == pytest.approx(esperado, rel=1e-6)
    assert m.operacoes == 0


def test_rebaixamento_nunca_passa_de_cem_por_cento():
    """
    Perder mais que tudo nao existe. No instante da quebra o patrimonio chega a
    ser marcado levemente negativo, e sem o piso a varredura de alavancagem
    reportava queda de -100,1%.
    """
    from avaliacao.replay import Resultado

    r = Resultado(saldo_inicial=1000.0, saldo_final=0.0, fechamentos=[],
                  curva=[(datetime(2024, 1, 1), 1000.0),
                         (datetime(2024, 1, 2), 500.0),
                         (datetime(2024, 1, 3), -5.0)])
    assert medir(r).rebaixamento == pytest.approx(-1.0)
