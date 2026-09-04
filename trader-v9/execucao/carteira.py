# -*- coding: utf-8 -*-
"""
execucao/carteira.py — comprado ou em caixa, e nada mais
=========================================================

O executor da sobreposição. Ele sabe fazer exatamente duas coisas:

    EXPOSTO   comprado nos ativos do universo, peso igual, 1x
    CAIXA     tudo vendido

Não há alavancagem, não há posição vendida, não há stop, não há alvo. Essa
pobreza é deliberada e é o que torna este arquivo seguro: a maior parte dos
incidentes do projeto veio de mecanismos que aqui não existem.

O que este arquivo NÃO tem, e por quê
--------------------------------------
  alavancagem    medido em 02/09/2026: de 1x para 2x o retorno CAI e o
                 rebaixamento dobra; em 20x as seis rodadas zeraram a conta
  posição vendida  o motor Bear nunca superou o custo em nenhuma medição
  stop/alvo      a sobreposição sai por decisão de regime, não por preço;
                 sem barreiras não há o que falhar em chegar à corretora,
                 que foi o incidente de 19/08/2026

As travas
---------
  1. DRY-RUN por padrão. Só envia ordem com --armar explícito.
  2. Idempotente: lê o estado real antes de agir e só negocia a diferença.
     Rodar duas vezes seguidas não duplica nada.
  3. Toda ordem é reconferida na corretora depois de enviada.
  4. Se o saldo não puder ser lido, não opera. Não há caminho que negocie
     sobre um número que não foi confirmado.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from binance.client import Client          # noqa: E402

TAXA_TOMADOR = 0.0004
MARGEM_DE_FOLGA = 0.98      # usa 98% do saldo; o resto cobre taxa e variação


@dataclass(frozen=True, slots=True)
class Posicao:
    symbol: str
    quantidade: float
    preco: float

    @property
    def valor(self) -> float:
        return self.quantidade * self.preco


class CarteiraBinance:
    """Adaptador da Binance para a sobreposição. Futuros, 1x, somente comprado."""

    def __init__(self, ativos: list[str], testnet: bool = True,
                 armado: bool = False, log=print):
        self.ativos = list(ativos)
        self.testnet = testnet
        self.armado = armado
        self.log = log
        self.api: Client | None = None
        self._filtros: dict[str, tuple[float, float]] = {}

    # ── ligação ────────────────────────────────────────────────────────────
    def conectar(self) -> None:
        # A Binance mantem testnets separadas para spot e futuros, com chaves
        # distintas. Estas sao as de FUTUROS; usar as de spot aqui produz
        # APIError -2015, que foi um dos tropecos da V4.
        chave = (os.getenv("BINANCE_FUTURES_API_KEY")
                 or os.getenv("BINANCE_API_KEY"))
        segredo = (os.getenv("BINANCE_FUTURES_SECRET_KEY")
                   or os.getenv("BINANCE_SECRET_KEY"))
        if not (chave and segredo):
            raise SystemExit(
                "Chaves ausentes. Esperado no .env: BINANCE_FUTURES_API_KEY e "
                "BINANCE_FUTURES_SECRET_KEY (ou as variantes sem _FUTURES)."
            )
        self.api = Client(chave, segredo, testnet=self.testnet)
        self.api.futures_ping()

        info = self.api.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] in self.ativos:
                passo = tick = 0.0
                for f in s["filters"]:
                    if f["filterType"] == "LOT_SIZE":
                        passo = float(f["stepSize"])
                    elif f["filterType"] == "PRICE_FILTER":
                        tick = float(f["tickSize"])
                self._filtros[s["symbol"]] = (passo, tick)

        faltando = [a for a in self.ativos if a not in self._filtros]
        if faltando:
            raise SystemExit(f"ativos sem filtros na corretora: {faltando}")

        modo = "TESTNET" if self.testnet else "REAL"
        estado = "ARMADO (envia ordens)" if self.armado else "DRY-RUN (nada e enviado)"
        self.log(f"Conectado — {modo} | {estado}")

    # ── leitura ────────────────────────────────────────────────────────────
    @property
    def saldo(self) -> float:
        """Patrimônio total em USDT. Levanta se não conseguir ler."""
        c = self.api.futures_account()
        return float(c["totalMarginBalance"])

    def posicoes(self) -> list[Posicao]:
        """Só as posições realmente abertas, com preço corrente."""
        saida = []
        for p in self.api.futures_account()["positions"]:
            qtd = float(p.get("positionAmt", 0))
            if abs(qtd) > 0 and p["symbol"] in self.ativos:
                preco = float(self.api.futures_symbol_ticker(
                    symbol=p["symbol"])["price"])
                saida.append(Posicao(p["symbol"], qtd, preco))
        return saida

    def _arredondar(self, symbol: str, qtd: float) -> float:
        passo, _ = self._filtros[symbol]
        if passo <= 0:
            return qtd
        return math.floor(abs(qtd) / passo) * passo * (1 if qtd >= 0 else -1)

    # ── escrita ────────────────────────────────────────────────────────────
    def _ordem(self, symbol: str, lado: str, qtd: float) -> bool:
        """Envia uma ordem a mercado e CONFERE que ela existe na corretora."""
        qtd = abs(self._arredondar(symbol, qtd))
        if qtd <= 0:
            return True

        if not self.armado:
            self.log(f"    [dry-run] {lado} {qtd} {symbol}")
            return True

        r = self.api.futures_create_order(
            symbol=symbol, side=lado, type="MARKET", quantity=qtd)
        ident = r.get("orderId")
        if ident is None:
            self.log(f"    [ERRO] {symbol}: resposta sem orderId — NAO criada")
            return False

        # Reconferir: a resposta pode chegar antes de a ordem existir de fato.
        for _ in range(3):
            time.sleep(0.4)
            try:
                conf = self.api.futures_get_order(symbol=symbol, orderId=ident)
            except Exception:
                continue
            if conf.get("status") in ("FILLED", "PARTIALLY_FILLED"):
                exec_qtd = float(conf.get("executedQty", 0))
                self.log(f"    [ok] {lado} {exec_qtd} {symbol} "
                         f"(ordem {ident}, {conf['status']})")
                return True
        self.log(f"    [ERRO] {symbol}: ordem {ident} nao confirmou execucao")
        return False

    # ── a única operação que existe ────────────────────────────────────────
    def ajustar(self, exposto: bool) -> bool:
        """
        Leva a carteira ao estado desejado. Idempotente.

        Lê o que existe, calcula a diferença e negocia só ela. Chamar duas
        vezes seguidas com o mesmo alvo não gera nenhuma ordem na segunda.
        """
        saldo = self.saldo
        abertas = {p.symbol: p for p in self.posicoes()}
        self.log(f"  Saldo ${saldo:,.2f} | {len(abertas)} posicao(oes) aberta(s)")

        if not exposto:
            if not abertas:
                self.log("  Ja em CAIXA — nada a fazer.")
                return True
            self.log("  Alvo: CAIXA. Vendendo tudo.")
            ok = True
            for p in abertas.values():
                ok &= self._ordem(p.symbol, "SELL" if p.quantidade > 0 else "BUY",
                                  p.quantidade)
            return ok

        # Exposto: peso igual entre os ativos.
        alvo_por_ativo = (saldo * MARGEM_DE_FOLGA) / len(self.ativos)
        self.log(f"  Alvo: EXPOSTO, ${alvo_por_ativo:,.2f} por ativo "
                 f"({len(self.ativos)} ativos, 1x)")

        ok = True
        for symbol in self.ativos:
            preco = float(self.api.futures_symbol_ticker(symbol=symbol)["price"])
            atual = abertas.get(symbol)
            valor_atual = atual.valor if atual else 0.0
            diferenca = alvo_por_ativo - valor_atual

            # Só mexe se a diferença for relevante — evita moer taxa em ajuste
            # de centavos a cada ciclo.
            if abs(diferenca) < alvo_por_ativo * 0.10:
                self.log(f"    {symbol}: ja proximo do alvo "
                         f"(${valor_atual:,.2f}), sem ordem")
                continue

            qtd = diferenca / preco
            ok &= self._ordem(symbol, "BUY" if qtd > 0 else "SELL", qtd)
        return ok

    def configurar_alavancagem(self) -> None:
        """Fixa 1x e margem isolada. Sem alavancagem, por medição."""
        if not self.armado:
            return
        for symbol in self.ativos:
            for chamada, kwargs in (
                (self.api.futures_change_margin_type,
                 {"symbol": symbol, "marginType": "ISOLATED"}),
                (self.api.futures_change_leverage,
                 {"symbol": symbol, "leverage": 1}),
            ):
                try:
                    chamada(**kwargs)
                except Exception as e:
                    # "No need to change" e resposta normal quando ja esta certo.
                    if "No need to change" not in str(e):
                        self.log(f"    [aviso] {symbol}: {e}")
