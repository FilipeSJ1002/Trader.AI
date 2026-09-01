# -*- coding: utf-8 -*-
"""
dados/fonte.py — carrega o historico do disco
==============================================

A unica coisa que a V9 reaproveita das versoes anteriores sao os arquivos de
dados. Nenhum codigo antigo e importado — so os parquets, que sao fatos do
mercado e nao carregam decisao nossa nenhuma.

Os parquets vem no formato das V1..V8 (colunas em ingles, `date` como tempo).
A traducao para o vocabulario da V9 acontece aqui, num lugar so.
"""
from __future__ import annotations

import glob
import os
import tomllib

import polars as pl

from dados.visao import Historico

# Como as colunas se chamam no disco -> como se chamam na V9.
DE_PARA = {
    "date": "ts",
    "open": "abertura",
    "high": "maxima",
    "low": "minima",
    "close": "fechamento",
    "volume": "volume",
}

RAIZ_PADRAO = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "data")


class FonteParquet:
    """Le `data/SYMBOL_1m.parquet` e devolve Historico pronto."""

    def __init__(self, raiz: str = RAIZ_PADRAO, cfg: dict | None = None):
        self.raiz = raiz
        self.cfg = cfg or {}
        self._cache: dict[str, Historico] = {}

    def simbolos(self) -> list[str]:
        padrao = os.path.join(self.raiz, "*_1m.parquet")
        return sorted(os.path.basename(p).replace("_1m.parquet", "")
                      for p in glob.glob(padrao))

    def historico(self, symbol: str) -> Historico:
        """Carrega (e memoriza) o historico de um ativo."""
        if symbol in self._cache:
            return self._cache[symbol]

        caminho = os.path.join(self.raiz, f"{symbol}_1m.parquet")
        if not os.path.exists(caminho):
            raise FileNotFoundError(
                f"{caminho} nao existe. Disponiveis: {self.simbolos()}"
            )

        bruto = pl.read_parquet(caminho)
        faltando = [c for c in DE_PARA if c not in bruto.columns]
        if faltando:
            raise ValueError(f"{symbol}: colunas ausentes no parquet: {faltando}")

        df = (bruto.rename({k: v for k, v in DE_PARA.items() if k != v})
                   .select(list(DE_PARA.values()))
                   .sort("ts")
                   .unique(subset="ts", keep="last"))

        h = Historico(symbol, df, self.cfg.get("indicadores"))
        self._cache[symbol] = h
        return h

    def carregar(self, simbolos: list[str]) -> dict[str, Historico]:
        return {s: self.historico(s) for s in simbolos}


def ler_config(caminho: str | None = None) -> dict:
    """Le config/v9.toml. Todo numero ajustavel do projeto mora la."""
    if caminho is None:
        caminho = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config", "v9.toml",
        )
    with open(caminho, "rb") as f:
        return tomllib.load(f)
