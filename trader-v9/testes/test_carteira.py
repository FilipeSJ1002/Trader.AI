# -*- coding: utf-8 -*-
"""
Testes do arredondamento de quantidade.

Motivo (05/09/2026): a primeira ordem armada foi rejeitada com APIError -1111,
"Precision is over the maximum defined for this asset". A causa era
math.floor(0.0096/0.001)*0.001 == 0.009000000000000001 — ponto flutuante
produzindo 18 casas decimais onde a corretora aceita 3.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from execucao.carteira import CarteiraBinance


def _carteira(passo: str, min_qtd: float = 0.0, min_notional: float = 0.0):
    c = CarteiraBinance(["XXXUSDT"])
    c._filtros = {"XXXUSDT": {"passo": passo, "min_qtd": min_qtd,
                              "min_notional": min_notional}}
    return c


def casas(x: float) -> int:
    """Quantas casas decimais o numero tem quando escrito sem notacao cientifica."""
    s = f"{Decimal(str(x)):f}"
    return len(s.split(".")[1].rstrip("0")) if "." in s else 0


@pytest.mark.parametrize("passo,entrada,esperado", [
    ("0.001", 0.0096, 0.009),        # o caso exato que quebrou em producao
    ("0.001", 0.0090, 0.009),
    ("0.01",  0.3123, 0.31),
    ("0.1",   7.5399, 7.5),
    ("1",     103.87, 103.0),
    ("0.1",   549.04, 549.0),
])
def test_trunca_para_o_passo(passo, entrada, esperado):
    assert _carteira(passo)._arredondar("XXXUSDT", entrada) == pytest.approx(esperado)


@pytest.mark.parametrize("passo", ["0.001", "0.01", "0.1", "1"])
@pytest.mark.parametrize("valor", [0.0096, 0.3123, 7.5399, 103.87, 549.04])
def test_nunca_excede_a_precisao_do_passo(passo, valor):
    """
    A garantia que faltava: o resultado nao pode ter mais casas decimais que o
    proprio passo. E exatamente isto que o APIError -1111 reclama.
    """
    r = _carteira(passo)._arredondar("XXXUSDT", valor)
    assert casas(r) <= casas(float(passo)), f"{r!r} tem casas demais para {passo}"


def test_preserva_o_sinal():
    c = _carteira("0.001")
    assert c._arredondar("XXXUSDT", -0.0096) == pytest.approx(-0.009)


def test_nunca_arredonda_para_cima():
    """Truncar, nunca arredondar: sobrar margem e seguro, faltar nao e."""
    c = _carteira("0.001")
    for v in (0.0019, 0.0011, 0.0099):
        assert c._arredondar("XXXUSDT", v) <= v


def test_quantidade_abaixo_do_minimo_nao_vira_ordem():
    c = _carteira("0.001", min_qtd=0.01)
    c.armado = True
    enviou = []
    c.api = type("FalsaAPI", (), {
        "futures_create_order": lambda *a, **k: enviou.append(k) or {}
    })()
    assert c._ordem("XXXUSDT", "BUY", 0.005) is True
    assert not enviou, "mandou ordem abaixo da quantidade minima"


def test_valor_abaixo_do_notional_minimo_nao_vira_ordem():
    c = _carteira("0.001", min_notional=100.0)
    c.armado = True
    enviou = []
    c.api = type("FalsaAPI", (), {
        "futures_create_order": lambda *a, **k: enviou.append(k) or {}
    })()
    assert c._ordem("XXXUSDT", "BUY", 0.005, preco=1000.0) is True
    assert not enviou, "mandou ordem abaixo do notional minimo"
