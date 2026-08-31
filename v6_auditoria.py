# -*- coding: utf-8 -*-
"""
v6_auditoria.py — Trader.AI: o que REALMENTE aconteceu na corretora
====================================================================

O log do bot conta a intencao dele. Esta ferramenta conta os fatos: pergunta
a propria Binance o historico de ordens e de resultado financeiro.

Responde tres perguntas que o log nao responde:

  1. As ordens de STOP LOSS e TAKE PROFIT foram mesmo criadas?
     (o executor loga "[ENVIADO] ... id=None", o que e ambiguo)
  2. Como cada posicao foi encerrada — stop, alvo, ou pelo proprio bot?
  3. Para onde foi o dinheiro: resultado das operacoes, taxas ou funding?

Uso:
  python v6_auditoria.py                # ultimos 7 dias
  python v6_auditoria.py --dias 30
  python v6_auditoria.py --real         # conta de PRODUCAO (padrao: testnet)
"""
import os
import sys
import argparse
import time
from datetime import datetime, timedelta, timezone

for _s in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_s, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

from v6_executor import FuturesExecutor
from v5_data_prep import ASSETS


def ts(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%d/%m %H:%M:%S")


def pagina_tudo(chamada, inicio, campo_ts="time", limite=1000, **kw):
    """
    Le um endpoint de historico cobrindo a janela inteira.

    Duas armadilhas da API, ambas silenciosas (medidas em 25/08/2026):

      1. JANELA MAXIMA DE 7 DIAS. allOrders, userTrades e income aceitam no
         maximo 7 dias entre startTime e endTime. Pedindo 30, a corretora
         devolve so os 7 primeiros dias a partir do startTime. Foi o que fez
         --dias 30 reportar MENOS perda que --dias 7.
      2. LIMITE DE REGISTROS. Dentro de cada janela, so vem `limite` registros
         (os mais antigos). Por isso tambem paginamos dentro da fatia.
    """
    SETE_DIAS = 7 * 24 * 60 * 60 * 1000
    agora = int(time.time() * 1000)
    saida, visto = [], set()

    ini_fatia = inicio
    while ini_fatia < agora:
        fim_fatia = min(ini_fatia + SETE_DIAS, agora)
        cursor = ini_fatia
        while True:
            lote = chamada(startTime=cursor, endTime=fim_fatia,
                           limit=limite, **kw)
            if isinstance(lote, dict):
                lote = lote.get("orders") or []
            if not lote:
                break
            for x in lote:
                chave = id_registro(x)
                if chave not in visto:
                    visto.add(chave)
                    saida.append(x)
            if len(lote) < limite:
                break
            tempos = [int(x[campo_ts]) for x in lote
                      if x.get(campo_ts) is not None]
            if not tempos:
                break
            prox = max(tempos) + 1
            if prox <= cursor:
                break
            cursor = prox
            time.sleep(0.2)
        ini_fatia = fim_fatia + 1
        time.sleep(0.2)
    return saida


def id_registro(x):
    """
    Chave EXATA do registro — o dicionario inteiro.

    Nao use campos de id aqui. Medido em 25/08/2026: no historico financeiro,
    a COMMISSION e o REALIZED_PNL da mesma operacao compartilham tranId e
    tradeId, e o funding vem com tradeId vazio. Deduplicar por esses campos
    apagava um lado de cada par e somava todo o funding num registro so.
    As fatias de tempo nao se sobrepoem, entao isto so protege as bordas.
    """
    return tuple(sorted((k, str(v)) for k, v in x.items()))


def main():
    ap = argparse.ArgumentParser(description="Auditoria da conta na Binance Futures")
    ap.add_argument("--dias", type=int, default=7, help="Janela em dias (padrao 7)")
    ap.add_argument("--real", action="store_true", help="Conta de producao (padrao: testnet)")
    ap.add_argument("--assets", default=None, help="Simbolos separados por virgula")
    a = ap.parse_args()

    inicio = int((datetime.now(timezone.utc) - timedelta(days=a.dias)).timestamp() * 1000)
    simbolos = ([s.strip().upper() for s in a.assets.split(",")] if a.assets else ASSETS)

    ex = FuturesExecutor(testnet=not a.real, dry_run=True)   # dry_run: so leitura
    ex.conectar()

    # ── 0. Saldo — a pergunta que a ferramenta nunca respondia ──────────────
    print("\n" + "=" * 78)
    print("  SALDO DA CONTA — agora")
    print("=" * 78)
    try:
        cont = ex.api.futures_account()
        saldo = float(cont["totalWalletBalance"])
        naorealizado = float(cont["totalUnrealizedProfit"])
        margem = float(cont["totalMarginBalance"])
        print(f"\n  Carteira (fechado)      : ${saldo:>12,.2f}")
        print(f"  Posicoes abertas (PnL)  : ${naorealizado:>+12,.2f}")
        print(f"  {'-'*40}")
        print(f"  TOTAL                   : ${margem:>12,.2f}")
        abertas = [pp for pp in cont.get("positions", [])
                   if abs(float(pp.get("positionAmt", 0))) > 0]
        if abertas:
            print(f"\n  {len(abertas)} posicao(oes) aberta(s):")
            for pp in abertas:
                print(f"    {pp['symbol']:<10} qtd {float(pp['positionAmt']):>+12.4f} "
                      f"| entrada {float(pp['entryPrice']):>12,.4f} "
                      f"| PnL ${float(pp['unrealizedProfit']):>+9.2f}")
        else:
            print("\n  Nenhuma posicao aberta.")
    except Exception as e:
        print(f"  [aviso] nao consegui ler o saldo: {e}")

    # ── 1. Ordens criadas no periodo ────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  ORDENS NA CORRETORA — ultimos {a.dias} dias")
    print("=" * 78)

    todas = []
    for sym in simbolos:
        try:
            todas += pagina_tudo(ex.api.futures_get_all_orders, inicio,
                                 campo_ts="time", limite=500, symbol=sym)
        except Exception as e:
            print(f"  [aviso] {sym}: {e}")

    # As ordens CONDICIONAIS (stop e alvo) vivem no sistema de algo orders e
    # NAO aparecem na consulta acima. Foi o que fez a primeira versao desta
    # ferramenta concluir "nenhuma posicao teve stop" — conclusao errada.
    condicionais = []
    for sym in simbolos:
        try:
            r = pagina_tudo(ex.api.futures_get_all_algo_orders, inicio,
                            campo_ts="bookTime", limite=100, symbol=sym)
        except Exception as e:
            print(f"  [aviso] condicionais de {sym}: {e}")
            continue
        if isinstance(r, dict):
            r = r.get("orders") or []
        for o in r:
            o["time"] = o.get("createTime", o.get("time", 0))
            o["type"] = o.get("orderType", "CONDITIONAL")
            o["stopPrice"] = o.get("triggerPrice")
            o["status"] = o.get("algoStatus", "?")
            o["clientOrderId"] = o.get("clientAlgoId", "?")
            o["executedQty"] = o.get("executedQty", o.get("quantity", "0"))
            o["_condicional"] = True
        condicionais += r

    todas += condicionais
    todas.sort(key=lambda o: o["time"])
    print(f"\n  ({len(todas) - len(condicionais)} ordens comuns + "
          f"{len(condicionais)} condicionais)")

    if not todas:
        print("  Nenhuma ordem no periodo.")
    else:
        print(f"\n  {'Quando':<15} {'Simbolo':<10} {'Tipo':<18} {'Lado':<5} "
              f"{'Status':<9} {'Exec':>10}  Origem (clientOrderId)")
        print("  " + "-" * 92)
        for o in todas:
            print(f"  {ts(o['time']):<15} {o['symbol']:<10} {o['type']:<18} "
                  f"{o['side']:<5} {o['status']:<9} {o.get('executedQty','0'):>10}  "
                  f"{o.get('clientOrderId','?')}")

        # ── Quem mandou cada ordem? ─────────────────────────────────────────
        # A Binance marca as ordens que ELA gera com prefixos reservados. Se
        # uma saida tem esses prefixos, nao foi o bot que fechou a posicao.
        print("\n  Legenda de origem:")
        SISTEMA = {
            "autoclose-":            "LIQUIDACAO pela corretora",
            "adl_autoclose":         "ADL (auto-deleveraging)",
            "settlement_autoclose-": "liquidacao por settlement do contrato",
        }
        achou_sistema = False
        for o in todas:
            cid = str(o.get("clientOrderId", ""))
            for prefixo, desc in SISTEMA.items():
                if cid.startswith(prefixo):
                    print(f"    [SISTEMA] {ts(o['time'])} {o['symbol']:<10} -> {desc}")
                    achou_sistema = True
        if not achou_sistema:
            print("    Nenhuma ordem gerada pela corretora (sem liquidacoes/ADL).")

        # Detalhes crus das ordens de saida — para entender quem as criou
        print("\n  Detalhe das ordens de SAIDA (reduceOnly / closePosition / origType):")
        for o in todas:
            if o.get("reduceOnly") or o.get("closePosition") or o["side"] == "BUY":
                print(f"    {ts(o['time'])} {o['symbol']:<10} type={o['type']:<16} "
                      f"origType={o.get('origType','?'):<18} "
                      f"reduceOnly={o.get('reduceOnly')} "
                      f"closePosition={o.get('closePosition')} "
                      f"workingType={o.get('workingType','?')}")

    # ── 2. Toda entrada teve protecao? ──────────────────────────────────────
    print("\n" + "=" * 78)
    print("  VERIFICACAO DE PROTECAO — toda entrada teve STOP e ALVO?")
    print("=" * 78)

    print("""
  Esta secao ja consulta o sistema de ordens CONDICIONAIS (algo orders), onde
  stop e alvo realmente vivem. Leitura:
    [OK]            stop e alvo encontrados    -> protecao comprovada
    [INDETERMINADO] sem condicional no registro, mas a saida tem
                    closePosition=True         -> protecao provavelmente existia
                                                  (ordem antiga, antes do fix)
    [FALHA]         nem uma coisa nem outra    -> posicao ficou desprotegida
""")

    entradas = [o for o in todas if o["type"] == "MARKET" and not o.get("reduceOnly")]
    if not entradas:
        print("  Nenhuma entrada a mercado no periodo.")
    for e in entradas:
        posteriores = [o for o in todas
                       if o["symbol"] == e["symbol"] and o["time"] >= e["time"]]
        janela = [o for o in posteriores if o["time"] - e["time"] <= 120_000]
        sl = [o for o in janela if o["type"] in ("STOP_MARKET", "STOP")]
        tp = [o for o in janela if o["type"] in ("TAKE_PROFIT_MARKET", "TAKE_PROFIT")]

        # A saida daquela posicao: primeira ordem reduceOnly depois da entrada
        saida = next((o for o in posteriores
                      if o.get("reduceOnly") and o["time"] > e["time"]), None)
        por_close = bool(saida and saida.get("closePosition"))

        if sl and tp:
            marca, nota = "OK           ", ""
        elif por_close:
            marca = "INDETERMINADO"
            nota = "   (saida via closePosition — protecao existia, gatilho nao auditavel)"
        else:
            marca = "FALHA        "
            nota = "   <-- POSICAO FICOU DESPROTEGIDA"

        dur = ((saida["time"] - e["time"]) / 60000) if saida else 0
        print(f"  [{marca}] {ts(e['time'])} {e['symbol']:<10} "
              f"stop={len(sl)} alvo={len(tp)} | durou {dur:.0f} min{nota}")
        for o in sl + tp:
            print(f"           -> {o['type']:<20} {o['status']:<10} "
                  f"gatilho {o.get('stopPrice','-')}")

    # ── 3. Para onde foi o dinheiro ─────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  RESULTADO FINANCEIRO — o que somou e o que subtraiu")
    print("=" * 78)

    try:
        renda = pagina_tudo(ex.api.futures_income_history, inicio,
                            campo_ts="time", limite=1000)
    except Exception as e:
        print(f"  [aviso] nao foi possivel ler o historico financeiro: {e}")
        renda = []

    por_tipo, por_simbolo = {}, {}
    for r in renda:
        v = float(r["income"])
        por_tipo[r["incomeType"]] = por_tipo.get(r["incomeType"], 0.0) + v
        if r.get("symbol"):
            por_simbolo[r["symbol"]] = por_simbolo.get(r["symbol"], 0.0) + v

    if por_tipo:
        print(f"\n  {'Categoria':<22} {'USD':>12}")
        print("  " + "-" * 36)
        for k, v in sorted(por_tipo.items(), key=lambda x: x[1]):
            print(f"  {k:<22} {v:>+12.2f}")
        print("  " + "-" * 36)
        print(f"  {'TOTAL':<22} {sum(por_tipo.values()):>+12.2f}")

        print(f"\n  {'Por ativo':<22} {'USD':>12}")
        print("  " + "-" * 36)
        for k, v in sorted(por_simbolo.items(), key=lambda x: x[1]):
            print(f"  {k:<22} {v:>+12.2f}")
    else:
        print("  Nenhuma movimentacao financeira no periodo.")

    # ── 4. Execucoes (fills) — hora exata e resultado de cada uma ───────────
    print("\n" + "=" * 78)
    print("  EXECUCOES — cada fill, com resultado realizado")
    print("=" * 78)
    fills = []
    for sym in simbolos:
        try:
            fills += pagina_tudo(ex.api.futures_account_trades, inicio,
                                 campo_ts="time", limite=500, symbol=sym)
        except Exception as e:
            print(f"  [aviso] {sym}: {e}")
    fills.sort(key=lambda t: t["time"])
    if fills:
        print(f"\n  {'Quando':<15} {'Simbolo':<10} {'Lado':<5} {'Qtd':>10} "
              f"{'Preco':>12} {'PnL':>10} {'Taxa':>8}")
        print("  " + "-" * 76)
        for t in fills:
            print(f"  {ts(t['time']):<15} {t['symbol']:<10} {t['side']:<5} "
                  f"{t['qty']:>10} {t['price']:>12} "
                  f"{float(t.get('realizedPnl', 0)):>+10.2f} "
                  f"{float(t.get('commission', 0)):>8.3f}")
    else:
        print("  Nenhuma execucao no periodo.")

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
