# -*- coding: utf-8 -*-
"""
Testes do replay, da corretora de papel e do risco.

O teste que da nome ao arquivo e `test_janela_nao_depende_da_data_de_inicio`:
e a defesa contra o defeito que destruiu todos os rankings da V8, onde deslocar
o inicio em sete minutos mudava o resultado de -4,90% para +8,24%.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from avaliacao.replay import _linha_do_tempo, replay
from dados.visao import Historico
from execucao.papel import CorretoraPapel, SemSaldo
from execucao.risco import Risco
from motores.bear import MotorBear
from motores.bull import MotorBull
from nucleo.protocolos import Corretora
from nucleo.tipos import Lado, Regime, Sinal


class OraculoFixo:
    """Sempre o mesmo regime. Para testar o replay sem depender do Oraculo."""

    def __init__(self, r: Regime):
        self.nome = f"fixo-{r.value}"
        self._r = r

    def regime(self, visao) -> Regime:
        return self._r


def _historico(passos: np.ndarray, symbol: str = "AAA",
               inicio: datetime = datetime(2024, 1, 1)) -> Historico:
    n = len(passos)
    fech = 100.0 * np.exp(np.cumsum(passos))
    corpo = np.abs(passos) * fech * 0.6 + fech * 2e-4
    df = pl.DataFrame({
        "ts": pl.datetime_range(inicio, inicio + timedelta(minutes=n),
                                interval="1m", eager=True, closed="left"),
        "abertura": fech,
        "maxima": fech + corpo,
        "minima": fech - corpo,
        "fechamento": fech,
        "volume": np.full(n, 100.0),
    })
    return Historico(symbol, df)


@pytest.fixture
def mercado() -> dict[str, Historico]:
    """
    Trinta dias. Nao e exagero: com `prazo_atr="diario"` o sistema so consegue
    dimensionar depois de 14 barras diarias FECHADAS, entao qualquer fixture
    mais curta faz o replay nao operar e o teste passa sem testar nada.
    """
    rng = np.random.default_rng(11)
    n = 30 * 24 * 60
    return {
        "AAA": _historico(rng.normal(0.00008, 0.0012, n), "AAA"),
        "BBB": _historico(rng.normal(-0.00005, 0.0015, n), "BBB"),
    }


# ── Corretora de papel ──────────────────────────────────────────────────────
def test_satisfaz_o_protocolo():
    assert isinstance(CorretoraPapel(), Corretora)


def _ordem(lado=Lado.LONG, preco=100.0, qtd=10.0, stop=97.0, alvo=106.0):
    from execucao.risco import Ordem
    s = Sinal(ts=datetime(2024, 1, 1), symbol="AAA", lado=lado,
              preco_ref=preco, forca=1.0, motivo="teste")
    return Ordem(sinal=s, quantidade=qtd, alavancagem=1.0, stop=stop, alvo=alvo)


def test_taxa_cobrada_nas_duas_pernas():
    c = CorretoraPapel(saldo_inicial=10_000.0, taxa=0.0004)
    pos = c.abrir(_ordem())                    # notional 1000 -> taxa 0,40
    assert c.saldo == pytest.approx(10_000 - 0.40)
    c.fechar(pos, "TESTE", preco=100.0)        # sem PnL, so a taxa de saida
    assert c.saldo == pytest.approx(10_000 - 0.80)
    assert c.fechamentos[-1].taxas == pytest.approx(0.80)


def test_lucro_e_prejuizo_batem_com_a_conta():
    c = CorretoraPapel(saldo_inicial=10_000.0, taxa=0.0)
    pos = c.abrir(_ordem(qtd=10.0, preco=100.0))
    c.fechar(pos, "TESTE", preco=110.0)
    assert c.saldo == pytest.approx(10_100.0)   # 10 unidades x $10

    c2 = CorretoraPapel(saldo_inicial=10_000.0, taxa=0.0)
    p2 = c2.abrir(_ordem(lado=Lado.SHORT, qtd=10.0, preco=100.0,
                         stop=103.0, alvo=94.0))
    c2.fechar(p2, "TESTE", preco=90.0)
    assert c2.saldo == pytest.approx(10_100.0)  # short ganha na queda


def test_empate_na_mesma_vela_conta_como_stop():
    """A regra pessimista. Assumir o alvo infla o resultado sistematicamente."""
    c = CorretoraPapel(saldo_inicial=10_000.0, taxa=0.0)
    pos = c.abrir(_ordem(preco=100.0, stop=97.0, alvo=106.0))
    c.proteger(pos, 97.0, 106.0)

    ts = np.array([np.datetime64("2024-01-01T00:01", "us")])
    f = c.varrer("AAA", ts, np.array([107.0]), np.array([96.0]))  # toca os dois
    assert f is not None and f.causa == "STOP"


def test_alvo_dispara_quando_so_ele_e_tocado():
    c = CorretoraPapel(saldo_inicial=10_000.0, taxa=0.0)
    pos = c.abrir(_ordem(preco=100.0, stop=97.0, alvo=106.0))
    c.proteger(pos, 97.0, 106.0)
    ts = np.array([np.datetime64("2024-01-01T00:01", "us")])
    f = c.varrer("AAA", ts, np.array([107.0]), np.array([99.0]))
    assert f is not None and f.causa == "ALVO"
    assert f.preco_saida == pytest.approx(106.0)


def test_posicao_sem_protecao_nao_e_varrida():
    """Sem proteger(), nao ha barreira — espelha a corretora real."""
    c = CorretoraPapel(saldo_inicial=10_000.0, taxa=0.0)
    c.abrir(_ordem(preco=100.0, stop=97.0, alvo=106.0))
    ts = np.array([np.datetime64("2024-01-01T00:01", "us")])
    assert c.varrer("AAA", ts, np.array([200.0]), np.array([1.0])) is None


def test_nao_abre_duas_posicoes_no_mesmo_ativo():
    c = CorretoraPapel(saldo_inicial=10_000.0)
    c.abrir(_ordem())
    with pytest.raises(ValueError, match="ja ha posicao aberta"):
        c.abrir(_ordem())


def test_saldo_zerado_levanta():
    c = CorretoraPapel(saldo_inicial=100.0, taxa=0.0)
    pos = c.abrir(_ordem(qtd=10.0, preco=100.0))
    with pytest.raises(SemSaldo):
        c.fechar(pos, "TESTE", preco=50.0)      # perde 500 num saldo de 100


def test_prazo_atr_invalido_e_recusado():
    with pytest.raises(ValueError, match="prazo_atr invalido"):
        Risco(prazo_atr="semanal")


# ── Risco ───────────────────────────────────────────────────────────────────
def test_stop_e_alvo_saem_em_multiplos_de_atr(mercado):
    v = mercado["AAA"].em(30_000)
    for prazo, fonte in [("minuto", lambda x: x),
                         ("h4", lambda x: x.h4),
                         ("diario", lambda x: x.diario)]:
        r = Risco(atr_stop=1.5, atr_alvo=3.0, usar_forca=False,
                  prazo_atr=prazo)
        s = Sinal(ts=v.ts, symbol="AAA", lado=Lado.LONG,
                  preco_ref=v.fechamento, forca=1.0, motivo="t")
        o = r.dimensionar(s, v, 10_000.0)
        atr = fonte(v).agora("atr")
        assert o is not None, f"nao dimensionou com prazo {prazo}"
        assert o.stop == pytest.approx(v.fechamento - 1.5 * atr)
        assert o.alvo == pytest.approx(v.fechamento + 3.0 * atr)


def test_atr_diario_e_muito_maior_que_o_de_minuto(mercado):
    """
    A razao de `prazo_atr` existir. Com o ATR de 1 minuto o stop fica dentro do
    ruido e a taxa de 0,08% come a operacao antes de ela ter chance.
    """
    v = mercado["AAA"].em(30_000)
    assert v.diario.agora("atr") > 10 * v.agora("atr")


def test_short_inverte_stop_e_alvo(mercado):
    v = mercado["AAA"].em(30_000)
    r = Risco(usar_forca=False)
    s = Sinal(ts=v.ts, symbol="AAA", lado=Lado.SHORT, preco_ref=v.fechamento,
              forca=1.0, motivo="t")
    o = r.dimensionar(s, v, 10_000.0)
    assert o is not None and o.stop > v.fechamento > o.alvo


def test_forca_escala_o_tamanho(mercado):
    v = mercado["AAA"].em(30_000)
    r = Risco(usar_forca=True)
    def qtd(f):
        s = Sinal(ts=v.ts, symbol="AAA", lado=Lado.LONG,
                  preco_ref=v.fechamento, forca=f, motivo="t")
        return r.dimensionar(s, v, 10_000.0).quantidade
    assert qtd(0.5) == pytest.approx(qtd(1.0) / 2)


def test_sem_atr_nao_dimensiona(mercado):
    v = mercado["AAA"].em(5)                    # cedo demais, ATR ainda e NaN
    s = Sinal(ts=v.ts, symbol="AAA", lado=Lado.LONG, preco_ref=v.fechamento,
              forca=1.0, motivo="t")
    assert Risco().dimensionar(s, v, 10_000.0) is None


# ── Linha do tempo ──────────────────────────────────────────────────────────
def test_janela_nao_depende_da_data_de_inicio(mercado):
    """
    A defesa contra o defeito da V8.

    Os instantes de ciclo sao ancorados na epoca, nao no comeco do periodo.
    Pedir uma janela larga ou uma estreita tem de avaliar EXATAMENTE os mesmos
    relogios no trecho comum — senao o resultado depende de onde se comecou a
    olhar, que foi o que invalidou todos os rankings anteriores.
    """
    largo = _linha_do_tempo(mercado, datetime(2024, 1, 1),
                            datetime(2024, 1, 20), 15)
    estreito = _linha_do_tempo(mercado, datetime(2024, 1, 12, 0, 7),
                               datetime(2024, 1, 20), 15)
    comuns = [t for t in largo if t >= datetime(2024, 1, 12, 0, 7)]
    assert comuns == estreito


def test_passos_caem_em_minutos_redondos(mercado):
    for t in _linha_do_tempo(mercado, datetime(2024, 1, 1),
                             datetime(2024, 1, 3), 15):
        assert t.minute % 15 == 0


# ── Replay ──────────────────────────────────────────────────────────────────
def test_replay_roda_e_fecha_tudo(mercado):
    c = CorretoraPapel(saldo_inicial=5000.0)
    res = replay(
        historicos=mercado,
        motores={Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
        oraculo=OraculoFixo(Regime.BULL),
        corretora=c,
        risco=Risco(),
        de=datetime(2024, 1, 20), ate=datetime(2024, 1, 30),
    )
    assert res.fechamentos, "nenhuma operacao — o teste nao provou nada"
    assert not c.posicoes(), "sobrou posicao aberta no fim"
    assert len(res.curva) > 100


def test_regime_bull_so_gera_long(mercado):
    c = CorretoraPapel(saldo_inicial=5000.0)
    res = replay(mercado, {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
                 OraculoFixo(Regime.BULL), c, Risco(),
                 datetime(2024, 1, 20), datetime(2024, 1, 30))
    assert all(f.posicao.lado is Lado.LONG for f in res.fechamentos)


def test_regime_bear_so_gera_short(mercado):
    c = CorretoraPapel(saldo_inicial=5000.0)
    res = replay(mercado, {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
                 OraculoFixo(Regime.BEAR), c, Risco(),
                 datetime(2024, 1, 20), datetime(2024, 1, 30))
    assert res.fechamentos
    assert all(f.posicao.lado is Lado.SHORT for f in res.fechamentos)


def test_regime_fora_nao_opera(mercado):
    c = CorretoraPapel(saldo_inicial=5000.0)
    res = replay(mercado, {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
                 OraculoFixo(Regime.FORA), c, Risco(),
                 datetime(2024, 1, 20), datetime(2024, 1, 30))
    assert not res.fechamentos
    assert res.saldo_final == pytest.approx(5000.0)


def test_limite_de_posicoes_respeitado(mercado):
    c = CorretoraPapel(saldo_inicial=5000.0)
    replay(mercado, {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
           OraculoFixo(Regime.BULL), c, Risco(max_posicoes=1),
           datetime(2024, 1, 20), datetime(2024, 1, 30))
    # Com 2 ativos e teto 1, nunca pode ter havido 2 simultaneas: se houvesse,
    # o fechamento de uma se sobreporia no tempo ao da outra.
    janelas = sorted((f.posicao.aberta_em, f.fechada_em)
                     for f in c.fechamentos)
    for (_, fim_a), (ini_b, _) in zip(janelas, janelas[1:]):
        assert ini_b >= fim_a, "houve duas posicoes abertas ao mesmo tempo"


def test_toda_posicao_e_protegida(mercado):
    """Nenhum fechamento pode ter causa SEM_PROTECAO com a corretora de papel."""
    c = CorretoraPapel(saldo_inicial=5000.0)
    res = replay(mercado, {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
                 OraculoFixo(Regime.BULL), c, Risco(),
                 datetime(2024, 1, 20), datetime(2024, 1, 30))
    assert not [f for f in res.fechamentos if f.causa == "SEM_PROTECAO"]


def test_corretora_que_nao_protege_faz_o_replay_fechar_na_hora(mercado):
    """Se proteger() falhar, a posicao nao pode ficar de pe. O buraco da V6."""

    class CorretoraSemProtecao(CorretoraPapel):
        def proteger(self, posicao, stop, alvo):
            return False

    c = CorretoraSemProtecao(saldo_inicial=5000.0)
    res = replay(mercado, {Regime.BULL: MotorBull(), Regime.BEAR: MotorBear()},
                 OraculoFixo(Regime.BULL), c, Risco(),
                 datetime(2024, 1, 20), datetime(2024, 1, 30))
    assert res.fechamentos
    assert all(f.causa == "SEM_PROTECAO" for f in res.fechamentos)
    assert not c.posicoes()
