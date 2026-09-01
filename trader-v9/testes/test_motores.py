# -*- coding: utf-8 -*-
"""
Testes dos motores.

O teste que da nome ao arquivo e `test_bull_nunca_vende`: um especialista que
opere o lado errado e defeito, e o codigo tem de gritar em vez de mandar a
ordem. Os demais garantem que a pontuacao e monotonica — mais recuo, mais
forca — porque uma pontuacao que nao ordena direito e pior que nenhuma.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from dados.visao import Historico
from motores.base import MotorBase, rampa
from motores.bear import MotorBear
from motores.bull import MotorBull
from nucleo.protocolos import Motor as ProtocoloMotor
from nucleo.tipos import Lado, Sinal


def _historico(passos: np.ndarray, symbol: str = "TESTE") -> Historico:
    """Monta um Historico a partir de uma sequencia de retornos."""
    n = len(passos)
    fech = 100.0 * np.exp(np.cumsum(passos))
    corpo = np.abs(passos) * fech * 0.5 + fech * 1e-4
    df = pl.DataFrame({
        "ts": pl.datetime_range(
            datetime(2024, 1, 1), datetime(2024, 1, 1) + timedelta(minutes=n),
            interval="1m", eager=True, closed="left",
        ),
        "abertura": fech,
        "maxima": fech + corpo,
        "minima": fech - corpo,
        "fechamento": fech,
        "volume": np.full(n, 100.0),
    })
    return Historico(symbol, df)


@pytest.fixture
def queda_forte() -> Historico:
    """Alta longa e depois uma queda brusca — o cenario do Bull."""
    rng = np.random.default_rng(3)
    subida = rng.normal(0.0004, 0.0008, 400)
    tombo = rng.normal(-0.004, 0.001, 60)
    return _historico(np.concatenate([subida, tombo]))


@pytest.fixture
def alta_forte() -> Historico:
    """Queda longa e depois um repique brusco — o cenario do Bear."""
    rng = np.random.default_rng(4)
    descida = rng.normal(-0.0004, 0.0008, 400)
    repique = rng.normal(0.004, 0.001, 60)
    return _historico(np.concatenate([descida, repique]))


# ── Contrato ────────────────────────────────────────────────────────────────
def test_motores_satisfazem_o_protocolo():
    assert isinstance(MotorBull(), ProtocoloMotor)
    assert isinstance(MotorBear(), ProtocoloMotor)


def test_lados_sao_fixos():
    assert MotorBull().lado is Lado.LONG
    assert MotorBear().lado is Lado.SHORT


def test_bull_nunca_vende(queda_forte, alta_forte):
    """
    O teste central. Varre os dois cenarios inteiros: em nenhum instante, sob
    nenhuma leitura de mercado, o Bull pode emitir SHORT.
    """
    motor = MotorBull()
    emitidos = 0
    for h in (queda_forte, alta_forte):
        for i in range(50, len(h)):
            s = motor.avaliar(h.em(i))
            if s is not None:
                emitidos += 1
                assert s.lado is Lado.LONG, f"Bull emitiu {s.lado} em {s.ts}"
    assert emitidos > 0, "o Bull nao emitiu nada — o teste nao provou nada"


def test_bear_nunca_compra(queda_forte, alta_forte):
    motor = MotorBear()
    emitidos = 0
    for h in (queda_forte, alta_forte):
        for i in range(50, len(h)):
            s = motor.avaliar(h.em(i))
            if s is not None:
                emitidos += 1
                assert s.lado is Lado.SHORT, f"Bear emitiu {s.lado} em {s.ts}"
    assert emitidos > 0, "o Bear nao emitiu nada — o teste nao provou nada"


def test_motor_que_tenta_trocar_de_lado_explode():
    """
    Um especialista que opere o lado errado tem de quebrar, nao mandar a ordem.

    A subclasse abaixo reescreve `emitir` para devolver o lado oposto — que e
    exatamente como o defeito apareceria na vida real: alguem mexe no metodo de
    construcao do sinal e inverte um sinal por engano. `avaliar` tem de pegar.
    """

    class MotorTorto(MotorBase):
        nome = "torto"
        lado = Lado.LONG            # se diz especialista de alta...

        def _pontuar(self, v):
            return 1.0, ["sempre"]

        def emitir(self, v, forca, motivo):
            # ...mas emite venda. Defeito, nao estrategia.
            return Sinal(ts=v.ts, symbol=v.symbol, lado=Lado.SHORT,
                         preco_ref=v.fechamento, forca=forca, motivo=motivo)

    h = _historico(np.random.default_rng(1).normal(0, 0.001, 300))
    with pytest.raises(AssertionError, match="defeito, nao estrategia"):
        MotorTorto().avaliar(h.em(250))


# ── Pontuacao ───────────────────────────────────────────────────────────────
def test_rampa_nos_dois_sentidos():
    assert rampa(48, 48, 22) == pytest.approx(0.0)
    assert rampa(22, 48, 22) == pytest.approx(1.0)
    assert rampa(35, 48, 22) == pytest.approx(0.5)
    assert rampa(10, 48, 22) == pytest.approx(1.0)      # satura
    assert rampa(60, 48, 22) == pytest.approx(0.0)
    assert rampa(float("nan"), 48, 22) == 0.0           # NaN nunca pontua


def test_bull_pontua_mais_quanto_mais_fundo(queda_forte):
    """Mais recuo tem de dar mais forca. Uma pontuacao que nao ordena e inutil."""
    motor = MotorBull()
    meio_da_alta = motor._pontuar(queda_forte.em(380))[0]
    fundo_do_tombo = motor._pontuar(queda_forte.em(len(queda_forte) - 2))[0]
    assert fundo_do_tombo > meio_da_alta


def test_bear_pontua_mais_quanto_mais_esticado(alta_forte):
    motor = MotorBear()
    meio_da_queda = motor._pontuar(alta_forte.em(380))[0]
    topo_do_repique = motor._pontuar(alta_forte.em(len(alta_forte) - 2))[0]
    assert topo_do_repique > meio_da_queda


def test_forca_fica_na_faixa(queda_forte, alta_forte):
    for motor in (MotorBull(), MotorBear()):
        for h in (queda_forte, alta_forte):
            for i in range(50, len(h), 7):
                f, _ = motor._pontuar(h.em(i))
                assert 0.0 <= f <= 1.0 or np.isnan(f)


def test_forca_min_controla_a_agressividade(queda_forte):
    """O botao de agressividade tem de funcionar: menor limiar, mais sinais."""
    contagens = []
    for limiar in (0.20, 0.50, 0.80):
        m = MotorBull(forca_min=limiar)
        contagens.append(sum(m.avaliar(queda_forte.em(i)) is not None
                             for i in range(50, len(queda_forte))))
    assert contagens[0] > contagens[1] > contagens[2], contagens


def test_nao_emite_antes_dos_indicadores_existirem():
    h = _historico(np.random.default_rng(9).normal(0, 0.001, 300))
    assert MotorBull().avaliar(h.em(3)) is None
    assert MotorBear().avaliar(h.em(3)) is None


def test_pesos_precisam_somar_um():
    with pytest.raises(ValueError, match="somam"):
        MotorBull(peso_rsi=0.5, peso_bb=0.5, peso_macd=0.5)


def test_sinal_carrega_motivo_legivel(queda_forte):
    motor = MotorBull(forca_min=0.2)
    sinais = [s for i in range(50, len(queda_forte))
              if (s := motor.avaliar(queda_forte.em(i))) is not None]
    assert sinais
    assert sinais[-1].motivo.startswith("[bull]")
    assert "RSI" in sinais[-1].motivo or "banda" in sinais[-1].motivo


def test_preco_ref_e_o_fechamento_do_instante(queda_forte):
    motor = MotorBull(forca_min=0.2)
    for i in range(50, len(queda_forte)):
        v = queda_forte.em(i)
        s = motor.avaliar(v)
        if s is not None:
            assert s.preco_ref == pytest.approx(v.fechamento)
            assert s.ts == v.ts
            return
    pytest.fail("nenhum sinal emitido")
