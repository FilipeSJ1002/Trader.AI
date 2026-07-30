# -*- coding: utf-8 -*-
"""
v6_executor.py — Trader.AI V6: ponte para execucao real (Binance Futures)
==========================================================================

Traduz as decisoes da estrategia hibrida em ORDENS REAIS na Binance Futures.
E o componente que faltava entre a estrategia validada (v5_backtest/v5_live)
e a operacao efetiva.

SEGURANCA EM PRIMEIRO LUGAR — tres travas independentes:
  1. DRY-RUN por padrao      : registra o que faria, NAO envia nada.
                               Enviar exige --armar explicitamente.
  2. TESTNET por padrao      : dinheiro ficticio. Produçao exige --real,
                               que por sua vez exige --armar.
  3. Limites rigidos         : exposicao maxima, posicoes simultaneas,
                               alavancagem teto — verificados a cada ordem.

PROTECAO NA CORRETORA (decisao de projeto):
  Ao abrir posicao, o executor registra TP e SL como ordens condicionais
  reduce-only NA PROPRIA BINANCE. Assim, se o bot cair, travar ou o PC
  desligar, o capital continua protegido — a corretora executa a saida.
  Um bot que depende de estar vivo para respeitar o stop e um bot perigoso.

Uso:
  python v6_executor.py --status                  # so inspeciona a conta
  python v6_executor.py --once                    # 1 ciclo em DRY-RUN (testnet)
  python v6_executor.py --once --armar            # 1 ciclo enviando ordens (testnet)
  python v6_executor.py --armar                   # loop continuo (testnet)
  python v6_executor.py --armar --real            # PRODUCAO (pede confirmacao)
"""
import os
import sys
import json
import time
import math
import argparse
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:                       # so para o editor/linter — nao roda
    from binance.client import Client

for _s in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_s, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

BASE = os.path.dirname(os.path.abspath(__file__))
REL  = os.path.join(BASE, "relatorios")
os.makedirs(REL, exist_ok=True)
LOG_PATH   = os.path.join(REL, "v6_executor.log")
STATE_PATH = os.path.join(BASE, "v6_executor_state.json")

# ── Limites de seguranca (verificados a cada ordem) ─────────────────────────
MAX_POSICOES_ABERTAS = 3        # nunca mais que isso simultaneamente
MAX_ALAVANCAGEM      = 5        # teto absoluto, mesmo que a estrategia peca mais
MARGEM_PCT           = 0.20     # 20% do saldo por operacao
MIN_NOTIONAL_USD     = 20.0     # abaixo disso a Binance rejeita
MAX_EXPOSICAO_PCT    = 0.60     # soma das margens nao passa de 60% do saldo


AJUDA_CHAVES = """
================================================================================
  COMO OBTER AS CHAVES DA BINANCE FUTURES TESTNET
================================================================================

A Binance mantem DUAS testnets independentes, com contas e chaves separadas:

    SPOT     ->  https://testnet.binance.vision       (a que voce ja usa)
    FUTUROS  ->  https://testnet.binancefuture.com    (necessaria para SHORT)

Usar a chave de spot num endpoint de futuros retorna APIError -2015.

PASSO A PASSO
-------------
 1. Acesse  https://testnet.binancefuture.com
 2. Entre com GitHub ou Google (a conta e independente da Binance real)
 3. Role ate o rodape -> aba "API Key"
 4. Copie a API Key e a Secret Key geradas
 5. Adicione ao arquivo .env do projeto:

        BINANCE_FUTURES_API_KEY=sua_chave_de_futuros
        BINANCE_FUTURES_SECRET_KEY=sua_secret_de_futuros

    (mantenha as chaves de spot como estao — o bot V4 continua usando-as)

 6. A conta ja vem com saldo ficticio de USDT para testes
 7. Verifique a conexao:

        python v6_executor.py --status

OBSERVACOES
-----------
 * A testnet de futuros e zerada periodicamente pela Binance; se o saldo
   sumir, basta gerar chaves novas.
 * Nenhum valor real esta envolvido enquanto --real nao for usado.
 * O executor so envia ordens com a flag --armar; sem ela, apenas simula.
================================================================================
"""


def log(msg=""):
    linha = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(linha, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(linha + "\n")


class FuturesExecutor:
    """Encapsula toda a interacao com a Binance Futures."""

    def __init__(self, testnet=True, dry_run=True):
        self.testnet = testnet
        self.dry_run = dry_run
        self.client: Optional["Client"] = None
        self.filtros = {}      # symbol -> {stepSize, tickSize, minNotional}

    @property
    def api(self) -> "Client":
        """
        Cliente da Binance ja conectado. Usar esta propriedade (em vez de
        self.client) garante erro claro se alguem chamar um metodo antes de
        conectar() — em vez de um AttributeError obscuro em 'None'.
        """
        if self.client is None:
            raise RuntimeError("Executor nao conectado — chame conectar() antes.")
        return self.client

    # ── Conexao ─────────────────────────────────────────────────────────────
    def conectar(self):
        """
        IMPORTANTE: a Binance mantem TESTNETS SEPARADAS para spot e futuros,
        com contas e chaves distintas:
            spot    -> testnet.binance.vision
            futuros -> testnet.binancefuture.com
        Usar chave de spot em endpoint de futuros retorna APIError -2015.
        Por isso procuramos primeiro as chaves especificas de futuros.
        """
        from binance.client import Client
        api_key = os.getenv("BINANCE_FUTURES_API_KEY") or os.getenv("BINANCE_API_KEY")
        secret  = os.getenv("BINANCE_FUTURES_SECRET_KEY") or os.getenv("BINANCE_SECRET_KEY")
        usando_especifica = bool(os.getenv("BINANCE_FUTURES_API_KEY"))

        if not api_key or not secret:
            raise RuntimeError("Chaves ausentes no .env "
                               "(BINANCE_FUTURES_API_KEY / BINANCE_FUTURES_SECRET_KEY)")
        if not usando_especifica:
            log("  [AVISO] usando BINANCE_API_KEY (spot). Se der APIError -2015, "
                "crie chaves de FUTUROS — ver instrucoes com --ajuda-chaves")

        self.client = Client(api_key, secret, testnet=self.testnet)

        # A Binance rejeita requisicoes cujo timestamp diverge do servidor dela
        # (APIError -1021). O relogio do Windows costuma ficar ~1s adiantado.
        # Medimos o desvio e compensamos, alem de ampliar a janela aceita.
        self._sincronizar_relogio()

        ambiente = "TESTNET (dinheiro ficticio)" if self.testnet else "PRODUCAO (dinheiro real)"
        modo = "DRY-RUN (nada e enviado)" if self.dry_run else "ARMADO (envia ordens)"
        log(f"Conectado — {ambiente} | {modo}")
        self._carregar_filtros()

    def _sincronizar_relogio(self):
        """Compensa a diferenca entre o relogio local e o da Binance."""
        try:
            servidor = self.api.futures_time()["serverTime"]
            local = int(time.time() * 1000)
            offset = servidor - local
            self.api.timestamp_offset = offset
            # Janela de tolerancia maior (padrao 5000ms) para variacoes de latencia
            self.api.REQUEST_TIMEOUT = 20
            log(f"Relogio sincronizado: desvio de {offset}ms compensado")
        except Exception as e:
            log(f"[AVISO] nao foi possivel sincronizar o relogio: {e}")

    def _carregar_filtros(self):
        """Cada simbolo tem regras de precisao; violar = ordem rejeitada."""
        info = self.api.futures_exchange_info()
        for s in info["symbols"]:
            f = {x["filterType"]: x for x in s["filters"]}
            self.filtros[s["symbol"]] = {
                "step":     float(f.get("LOT_SIZE", {}).get("stepSize", 0.001)),
                "tick":     float(f.get("PRICE_FILTER", {}).get("tickSize", 0.01)),
                "min_notional": float(f.get("MIN_NOTIONAL", {}).get("notional", MIN_NOTIONAL_USD)),
            }
        log(f"Filtros de precisao carregados para {len(self.filtros)} simbolos")

    # ── Arredondamentos exigidos pela corretora ─────────────────────────────
    def _ajusta_qtd(self, symbol, qtd):
        step = self.filtros.get(symbol, {}).get("step", 0.001)
        casas = max(0, int(round(-math.log10(step)))) if step < 1 else 0
        return round(math.floor(qtd / step) * step, casas)

    def _ajusta_preco(self, symbol, preco):
        tick = self.filtros.get(symbol, {}).get("tick", 0.01)
        casas = max(0, int(round(-math.log10(tick)))) if tick < 1 else 0
        return round(math.floor(preco / tick) * tick, casas)

    # ── Leitura de estado ───────────────────────────────────────────────────
    def saldo_usdt(self):
        for a in self.api.futures_account_balance():
            if a["asset"] == "USDT":
                return float(a["balance"])
        return 0.0

    def posicoes_abertas(self):
        """Le da CORRETORA — fonte da verdade, nao o arquivo local."""
        out = {}
        for p in self.api.futures_position_information():
            amt = float(p["positionAmt"])
            if abs(amt) > 0:
                out[p["symbol"]] = {
                    "qtd": amt,
                    "direcao": "LONG" if amt > 0 else "SHORT",
                    "entrada": float(p["entryPrice"]),
                    "pnl_nao_realizado": float(p.get("unRealizedProfit", 0)),
                    "alavancagem": int(float(p.get("leverage", 1))),
                }
        return out

    def ordens_abertas(self, symbol=None):
        if symbol:
            return self.api.futures_get_open_orders(symbol=symbol)
        return self.api.futures_get_open_orders()

    # ── Escrita (protegida pelo dry_run) ────────────────────────────────────
    def _enviar(self, descricao, fn, **kwargs):
        if self.dry_run:
            log(f"  [DRY-RUN] {descricao} | {kwargs}")
            return {"dry_run": True}
        try:
            r = fn(**kwargs)
            log(f"  [ENVIADO] {descricao} | id={r.get('orderId')}")
            return r
        except Exception as e:
            log(f"  [ERRO] {descricao} falhou: {e}")
            return None

    def abrir_posicao(self, symbol, direcao, margem_usd, alavancagem,
                      sl_price, tp_price, preco_ref):
        """
        Abre posicao a mercado e registra TP/SL como ordens condicionais
        reduce-only NA CORRETORA (protecao independente do bot estar vivo).
        """
        alavancagem = min(int(alavancagem), MAX_ALAVANCAGEM)
        notional = margem_usd * alavancagem

        # Cada simbolo tem seu proprio minimo (BTC exige $50; XRP, $5) e sua
        # granularidade de quantidade (AVAX so aceita inteiros). Violar
        # qualquer um resulta em ordem rejeitada pela corretora.
        min_not = self.filtros.get(symbol, {}).get("min_notional", MIN_NOTIONAL_USD)
        if notional < min_not:
            log(f"  [BLOQUEADO] {symbol}: notional ${notional:.2f} < minimo ${min_not:.0f} "
                f"exigido pela corretora")
            return None

        qtd = self._ajusta_qtd(symbol, notional / preco_ref)
        if qtd <= 0:
            log(f"  [BLOQUEADO] {symbol}: quantidade arredonda para zero "
                f"(step={self.filtros.get(symbol, {}).get('step')})")
            return None

        # O arredondamento para baixo pode derrubar o notional abaixo do minimo
        notional_real = qtd * preco_ref
        if notional_real < min_not:
            log(f"  [BLOQUEADO] {symbol}: apos arredondar, notional ${notional_real:.2f} "
                f"< minimo ${min_not:.0f}")
            return None

        lado      = "BUY" if direcao == "LONG" else "SELL"
        lado_saida = "SELL" if direcao == "LONG" else "BUY"
        sl_price = self._ajusta_preco(symbol, sl_price)
        tp_price = self._ajusta_preco(symbol, tp_price)

        log(f"  ABRIR {direcao} {symbol}: qtd={qtd} @ ~${preco_ref:.4f} | "
            f"margem ${margem_usd:.2f} x{alavancagem} = ${notional:.2f} | "
            f"SL ${sl_price} TP ${tp_price}")

        # 1. Configura margem isolada e alavancagem
        if not self.dry_run:
            try:
                self.api.futures_change_margin_type(symbol=symbol, marginType="ISOLATED")
            except Exception:
                pass  # ja esta isolada
            try:
                self.api.futures_change_leverage(symbol=symbol, leverage=alavancagem)
            except Exception as e:
                log(f"  [AVISO] nao ajustou alavancagem de {symbol}: {e}")

        # 2. Entrada a mercado
        entrada = self._enviar(
            f"entrada {lado} {symbol}",
            self.api.futures_create_order,
            symbol=symbol, side=lado, type="MARKET", quantity=qtd,
        )
        if entrada is None:
            return None

        # 3. Stop loss (reduce-only, fica na corretora)
        self._enviar(
            f"STOP LOSS {symbol} @ {sl_price}",
            self.api.futures_create_order,
            symbol=symbol, side=lado_saida, type="STOP_MARKET",
            stopPrice=sl_price, closePosition=True, timeInForce="GTE_GTC",
        )

        # 4. Take profit (reduce-only, fica na corretora)
        self._enviar(
            f"TAKE PROFIT {symbol} @ {tp_price}",
            self.api.futures_create_order,
            symbol=symbol, side=lado_saida, type="TAKE_PROFIT_MARKET",
            stopPrice=tp_price, closePosition=True, timeInForce="GTE_GTC",
        )

        return {"symbol": symbol, "direcao": direcao, "qtd": qtd,
                "alavancagem": alavancagem, "sl": sl_price, "tp": tp_price,
                "aberta_em": datetime.now(timezone.utc).isoformat()}

    def fechar_posicao(self, symbol, motivo=""):
        pos = self.posicoes_abertas().get(symbol)
        if not pos:
            log(f"  [INFO] {symbol} nao tem posicao aberta")
            return None
        lado = "SELL" if pos["direcao"] == "LONG" else "BUY"
        qtd  = abs(pos["qtd"])
        log(f"  FECHAR {pos['direcao']} {symbol}: qtd={qtd} | motivo: {motivo}")
        r = self._enviar(
            f"fechamento {symbol}",
            self.api.futures_create_order,
            symbol=symbol, side=lado, type="MARKET", quantity=qtd, reduceOnly=True,
        )
        self.cancelar_ordens(symbol)
        return r

    def cancelar_ordens(self, symbol):
        abertas = self.ordens_abertas(symbol)
        if not abertas:
            return
        if self.dry_run:
            log(f"  [DRY-RUN] cancelaria {len(abertas)} ordem(ns) de {symbol}")
            return
        try:
            self.api.futures_cancel_all_open_orders(symbol=symbol)
            log(f"  [ENVIADO] canceladas {len(abertas)} ordem(ns) de {symbol}")
        except Exception as e:
            log(f"  [ERRO] cancelamento de {symbol}: {e}")

    # ── Verificacoes de seguranca ───────────────────────────────────────────
    def pode_abrir(self, saldo):
        pos = self.posicoes_abertas()
        if len(pos) >= MAX_POSICOES_ABERTAS:
            return False, f"limite de {MAX_POSICOES_ABERTAS} posicoes atingido"
        exposicao = sum(abs(p["qtd"]) * p["entrada"] / max(p["alavancagem"], 1)
                        for p in pos.values())
        if saldo > 0 and exposicao / saldo > MAX_EXPOSICAO_PCT:
            return False, f"exposicao {exposicao/saldo*100:.0f}% > teto {MAX_EXPOSICAO_PCT*100:.0f}%"
        return True, "ok"


# ── Relatorio de conta ──────────────────────────────────────────────────────
def mostrar_status(ex):
    saldo = ex.saldo_usdt()
    pos   = ex.posicoes_abertas()
    log("=" * 68)
    log(f"  SALDO USDT (futuros): ${saldo:,.2f}")
    log(f"  POSICOES ABERTAS: {len(pos)}")
    for s, p in pos.items():
        log(f"    {p['direcao']:<5} {s:<10} qtd {p['qtd']:>12} @ ${p['entrada']:>10,.4f} "
            f"| x{p['alavancagem']} | PnL ${p['pnl_nao_realizado']:>+8.2f}")
    ordens = ex.ordens_abertas()
    log(f"  ORDENS PENDENTES: {len(ordens)}")
    for o in ordens[:10]:
        log(f"    {o['symbol']:<10} {o['type']:<20} stop ${o.get('stopPrice','-')}")
    ok, motivo = ex.pode_abrir(saldo)
    log(f"  PODE ABRIR NOVA POSICAO: {'sim' if ok else 'NAO — ' + motivo}")
    log("=" * 68)
    return saldo, pos


def main():
    ap = argparse.ArgumentParser(description="Trader.AI V6 — executor Binance Futures")
    ap.add_argument("--armar", action="store_true",
                    help="ENVIA ordens de verdade (sem esta flag e simulacao)")
    ap.add_argument("--real", action="store_true",
                    help="PRODUCAO com dinheiro real (padrao: testnet)")
    ap.add_argument("--status", action="store_true", help="So mostra o estado da conta")
    ap.add_argument("--once", action="store_true", help="Um ciclo e encerra")
    ap.add_argument("--fechar", metavar="SYMBOL", help="Fecha a posicao de um simbolo")
    ap.add_argument("--fechar-tudo", action="store_true", help="Fecha TODAS as posicoes")
    ap.add_argument("--intervalo", type=int, default=15, help="Minutos entre ciclos")
    ap.add_argument("--model", default="v5_model_b.pth")
    ap.add_argument("--ajuda-chaves", action="store_true",
                    help="Explica como obter as chaves da Futures Testnet")
    a = ap.parse_args()

    if a.ajuda_chaves:
        print(AJUDA_CHAVES)
        return

    testnet = not a.real
    dry_run = not a.armar

    # Confirmacao explicita para dinheiro real
    if a.real and a.armar:
        log("!" * 68)
        log("  ATENCAO: modo PRODUCAO com ordens REAIS e dinheiro REAL.")
        log("!" * 68)
        resp = input("  Digite exatamente 'CONFIRMO DINHEIRO REAL' para prosseguir: ")
        if resp.strip() != "CONFIRMO DINHEIRO REAL":
            log("  Cancelado pelo usuario.")
            return

    ex = FuturesExecutor(testnet=testnet, dry_run=dry_run)
    try:
        ex.conectar()
    except Exception as e:
        log(f"Falha ao conectar: {e}")
        log("Verifique BINANCE_API_KEY/BINANCE_SECRET_KEY no .env e se a conta")
        log("de FUTUROS esta habilitada (spot e futuros usam chaves distintas na testnet).")
        return

    if a.fechar:
        mostrar_status(ex)
        ex.fechar_posicao(a.fechar.upper(), motivo="comando manual")
        return

    if a.fechar_tudo:
        _, pos = mostrar_status(ex)
        for s in pos:
            ex.fechar_posicao(s, motivo="fechar-tudo manual")
        return

    if a.status:
        mostrar_status(ex)
        return

    # Ciclo de operacao: usa a MESMA estrategia do backtest/paper
    from v6_ciclo import executar_ciclo   # separado para manter este arquivo focado
    while True:
        try:
            saldo, _ = mostrar_status(ex)
            executar_ciclo(ex, saldo, model_path=a.model,
                           margem_pct=MARGEM_PCT, log=log)
        except KeyboardInterrupt:
            log("Interrompido pelo usuario.")
            break
        except Exception as e:
            log(f"[ERRO] ciclo falhou: {e} — tentara de novo")

        if a.once:
            break
        agora = time.time()
        passo = a.intervalo * 60
        time.sleep(passo - (agora % passo) + 2)


if __name__ == "__main__":
    main()
