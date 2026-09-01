# -*- coding: utf-8 -*-
"""
dados/visao.py — a janela pela qual a estrategia ve o mercado
==============================================================

O problema que este arquivo resolve
-----------------------------------
Todo backtest confia que o autor da estrategia nao vai espiar o futuro. Essa
confianca falha em silencio: um `shift` esquecido, um indicador centrado, um
`fillna(method="bfill")`, e o resultado fica otimo por um motivo que nunca
existira ao vivo.

Aqui a garantia e ESTRUTURAL. Ha dois objetos:

  Historico        tem os dados todos. E o arquivista. Estrategia nenhuma
                   recebe um destes.
  VisaoDeMercado   e o que a estrategia recebe. Contem FATIAS que terminam no
                   instante atual. Nao ha metodo para olhar adiante porque nao
                   ha dado adiante dentro do objeto.

As fatias sao views do NumPy: criar uma e O(1) e nao copia memoria, entao dá
para gerar uma visao nova a cada passo do replay sem custo.

Barras fechadas
---------------
As visoes de 4h e 1D so expoem barras JA ENCERRADAS. As 14h de uma terça, a
barra diaria de terça ainda esta sendo formada — seu fechamento so existe a
meia-noite. Usa-la seria vazamento de futuro, sutil e devastador, e e o erro
mais comum em sistemas que misturam prazos. Aqui a barra so aparece depois de
`inicio + duracao <= agora`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from dados.indicadores import enriquecer


@dataclass(frozen=True, slots=True)
class VisaoDeMercado:
    """
    O mercado ate `ts`, e nada alem.

    Satisfaz nucleo.protocolos.VisaoDeMercado. Construida por Historico.em();
    nao instancie diretamente.
    """

    symbol: str
    ts: datetime
    _colunas: dict[str, np.ndarray]        # todas terminam no instante atual
    _grosseiras: dict[str, "VisaoDeMercado"]

    # ── leitura ────────────────────────────────────────────────────────────
    def serie(self, nome: str) -> np.ndarray:
        """A coluna inteira ate agora. Somente leitura."""
        try:
            return self._colunas[nome]
        except KeyError:
            raise KeyError(
                f"coluna '{nome}' nao existe. Disponiveis: "
                f"{sorted(self._colunas)}"
            ) from None

    def agora(self, nome: str) -> float:
        """Valor atual. NaN enquanto o indicador nao tem barras suficientes."""
        s = self.serie(nome)
        return float(s[-1]) if len(s) else float("nan")

    def antes(self, nome: str, passos: int = 1) -> float:
        """Valor de `passos` barras atras — para detectar cruzamentos."""
        s = self.serie(nome)
        i = len(s) - 1 - passos
        return float(s[i]) if i >= 0 else float("nan")

    def cruzou_para_cima(self, nome: str, nivel: float = 0.0) -> bool:
        """Estava abaixo do nivel na barra anterior e esta acima agora."""
        ant, ago = self.antes(nome), self.agora(nome)
        return bool(ant <= nivel < ago) if not (np.isnan(ant) or np.isnan(ago)) else False

    def cruzou_para_baixo(self, nome: str, nivel: float = 0.0) -> bool:
        ant, ago = self.antes(nome), self.agora(nome)
        return bool(ant >= nivel > ago) if not (np.isnan(ant) or np.isnan(ago)) else False

    @property
    def fechamento(self) -> float:
        return self.agora("fechamento")

    @property
    def barras(self) -> int:
        return len(self.serie("fechamento"))

    @property
    def pronta(self) -> bool:
        """
        True quando os indicadores ja tem valor. Um motor deve checar isto
        antes de decidir — no comeco da serie, `agora()` devolve NaN e toda
        comparacao com NaN e False, o que silenciaria a estrategia sem aviso.
        """
        return not np.isnan(self.agora("atr"))

    # ── prazos maiores ─────────────────────────────────────────────────────
    @property
    def diario(self) -> "VisaoDeMercado":
        return self._grosseiras["1d"]

    @property
    def h4(self) -> "VisaoDeMercado":
        return self._grosseiras["4h"]

    def __repr__(self) -> str:
        return (f"<Visao {self.symbol} @ {self.ts:%Y-%m-%d %H:%M} "
                f"| {self.barras} barras | fech {self.fechamento:.4f}>")


class Historico:
    """
    O arquivista: tem a serie inteira e distribui janelas.

    Calcula os indicadores UMA vez sobre todo o historico. Isso e legitimo
    porque todos eles olham so para tras — testes/test_visao.py prova, e o
    teste roda no CI, nao quando alguem lembra.
    """

    PERIODOS = {"4h": timedelta(hours=4), "1d": timedelta(days=1)}

    def __init__(self, symbol: str, df: pl.DataFrame, cfg: dict | None = None):
        """
        `df` precisa ter ts, abertura, maxima, minima, fechamento, volume,
        ordenado por ts e sem duplicatas.
        """
        if df["ts"].is_duplicated().any():
            raise ValueError(f"{symbol}: ha timestamps duplicados")
        if not df["ts"].is_sorted():
            df = df.sort("ts")

        self.symbol = symbol
        self.cfg = cfg or {}
        self._fino = enriquecer(df, self.cfg)
        self._ts = self._fino["ts"].to_numpy()
        self._cols = {c: self._fino[c].to_numpy().astype(np.float64)
                      for c in self._fino.columns if c != "ts"}

        self._grosso: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
        for nome, periodo in self.PERIODOS.items():
            g = enriquecer(self._reamostrar(df, periodo), self.cfg)
            self._grosso[nome] = (
                g["ts"].to_numpy(),
                {c: g[c].to_numpy().astype(np.float64)
                 for c in g.columns if c != "ts"},
            )

    @staticmethod
    def _reamostrar(df: pl.DataFrame, periodo: timedelta) -> pl.DataFrame:
        """
        Agrega para um prazo maior. `ts` da barra e o INICIO dela — quem
        garante que ela ja fechou e o corte em `em()`.
        """
        return (
            df.sort("ts")
            .group_by_dynamic("ts", every=periodo, closed="left", label="left")
            .agg([
                pl.col("abertura").first(),
                pl.col("maxima").max(),
                pl.col("minima").min(),
                pl.col("fechamento").last(),
                pl.col("volume").sum(),
            ])
            .drop_nulls()
        )

    # ── a unica porta de saida ─────────────────────────────────────────────
    def em(self, i: int) -> VisaoDeMercado:
        """
        A visao no indice `i` da serie de 1 minuto.

        Todas as fatias sao `[: i + 1]`, entao a visao devolvida nao carrega
        um unico ponto posterior a este instante.
        """
        if not 0 <= i < len(self._ts):
            raise IndexError(f"indice {i} fora de [0, {len(self._ts)})")
        agora = self._ts[i].astype("datetime64[us]").astype(datetime)

        grosseiras = {}
        for nome, (ts_g, cols_g) in self._grosso.items():
            # SO barras encerradas: inicio + duracao <= agora.
            fim = np.searchsorted(
                ts_g,
                np.datetime64(agora - self.PERIODOS[nome], "us"),
                side="right",
            )
            grosseiras[nome] = VisaoDeMercado(
                symbol=self.symbol,
                ts=agora,
                _colunas={c: v[:fim] for c, v in cols_g.items()},
                _grosseiras={},
            )

        return VisaoDeMercado(
            symbol=self.symbol,
            ts=agora,
            _colunas={c: v[: i + 1] for c, v in self._cols.items()},
            _grosseiras=grosseiras,
        )

    def barras_entre(self, i_ini: int, i_fim: int) -> tuple[np.ndarray, ...]:
        """
        (instantes, maximas, minimas) do trecho [i_ini, i_fim].

        Existe para a corretora varrer gatilhos de stop e alvo minuto a minuto
        entre dois ciclos. E o unico jeito legitimo de alguem de fora ler barras
        cruas daqui — sem isto, o replay acabaria mexendo nos atributos privados
        e o encapsulamento viraria enfeite.
        """
        i_ini, i_fim = max(i_ini, 0), min(i_fim, len(self._ts) - 1)
        if i_fim < i_ini:
            vazio = np.empty(0)
            return vazio, vazio, vazio
        fatia = slice(i_ini, i_fim + 1)
        return (self._ts[fatia], self._cols["maxima"][fatia],
                self._cols["minima"][fatia])

    @property
    def instantes(self) -> np.ndarray:
        """A linha do tempo completa. Somente leitura."""
        return self._ts

    def indice_de(self, quando: datetime) -> int:
        """Indice da ultima barra em ou antes de `quando`."""
        i = int(np.searchsorted(self._ts, np.datetime64(quando, "us"),
                                side="right")) - 1
        if i < 0:
            raise ValueError(f"{quando} e anterior ao inicio do historico")
        return i

    def passos(self, de: datetime, ate: datetime, a_cada: int = 1):
        """Gera os indices de replay entre duas datas, de `a_cada` minutos."""
        return range(self.indice_de(de), self.indice_de(ate) + 1, a_cada)

    @property
    def inicio(self) -> datetime:
        return self._ts[0].astype("datetime64[us]").astype(datetime)

    @property
    def fim(self) -> datetime:
        return self._ts[-1].astype("datetime64[us]").astype(datetime)

    def __len__(self) -> int:
        return len(self._ts)

    def __repr__(self) -> str:
        return (f"<Historico {self.symbol} | {len(self):,} barras | "
                f"{self.inicio:%Y-%m-%d} -> {self.fim:%Y-%m-%d}>")
