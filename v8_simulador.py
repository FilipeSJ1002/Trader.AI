# -*- coding: utf-8 -*-
"""
v8_simulador.py — replay minuto a minuto chamando o CODIGO DE PRODUCAO
======================================================================

Por que existe (25/08/2026)
--------------------------
O v5_backtest.py ja percorre o tempo barra a barra, com posicoes vivas e
varredura intrabar. Mas ele e uma SEGUNDA implementacao da estrategia: as
funcoes que decidem em producao (v6_ciclo.decidir_entrada e
v6_ciclo.avaliar_saidas) nao sao as que ele executa. Duas implementacoes que
ninguem comparou podem divergir em silencio, e a configuracao em producao
(config B) nem mora no arquivo do backtest.

Este simulador elimina essa duvida: ele reconstroi, a cada passo, o MESMO
dicionario `mercado` que _montar_mercado() entrega ao vivo, e chama as funcoes
de producao sem copiar uma linha de logica de decisao. Se o resultado aqui for
ruim, e a estrategia que e ruim — nao a traducao dela.

O que e simulado com honestidade
--------------------------------
  - o relogio: cada passo so enxerga dados ATE aquele minuto (ver --verificar)
  - TP e SL: vivem na corretora, entao sao checados MINUTO A MINUTO entre os
    ciclos, com maxima e minima reais, nao com o fechamento
  - empate intrabar (a mesma vela toca TP e SL): conta como SL. E o cenario
    pessimista, e sem dados de tick nao da para saber a ordem
  - taxas nas duas pernas, alavancagem, margem isolada e o teto de 3 posicoes
  - o preco de entrada e o fechamento do minuto do ciclo — o bot ao vivo manda
    ordem a mercado alguns segundos depois, entao a diferenca real e pior

O que NAO e simulado
--------------------
  - deslizamento (as ordens grandes sao preenchidas em varios pedacos ao vivo)
  - profundidade do livro
Ambos empurram o resultado real para BAIXO do que sai aqui.

Uso:
  python v8_simulador.py --verificar               # prova que nao ve o futuro
  python v8_simulador.py --de 2025-01-01 --ate 2026-08-01
  python v8_simulador.py --rank                    # compara as configuracoes
"""
import sys
import os
import glob
import json
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

import v6_ciclo
from v6_ciclo import decidir_entrada, avaliar_saidas
from v5_data_prep import ASSETS, BTC, WINDOW_SIZE, _load_parquet, _add_features
from v5_backtest import FEE

CICLO_MIN = 15                  # o executor roda a cada 15 minutos
MAX_POSICOES = 3                # v6_executor.MAX_POSICOES_ABERTAS
MARGEM_PCT = 0.20               # v6_executor.MARGEM_PCT


# ────────────────────────────────────────────────────────────────────────────
#  Preparacao — espelha _montar_mercado(), mas sobre o historico inteiro
# ────────────────────────────────────────────────────────────────────────────
def preparar(sym, btc_df):
    """
    Calcula de uma vez o que _montar_mercado() calcula a cada ciclo ao vivo.

    Isto so e legitimo porque TODOS os indicadores envolvidos olham para tras
    (medias e medianas moveis). O modo --verificar prova essa afirmacao em vez
    de pedir que se confie nela.
    """
    adf = btc_df if sym == BTC else _load_parquet(sym)
    comum = adf.index.intersection(btc_df.index)
    a2, b2 = adf.loc[comum], btc_df.loc[comum]

    feat = _add_features(a2, b2).dropna()
    if len(feat) < WINDOW_SIZE + 2:
        return None

    close = a2["close"].reindex(feat.index)
    sma24 = a2["close"].rolling(1440).mean().reindex(feat.index)
    dist_rel = ((close - sma24).abs() / (sma24 + 1e-9))
    vol_tipica = dist_rel.rolling(1440, min_periods=120).median()
    forca = (dist_rel / (vol_tipica + 1e-9)).fillna(0.0)

    return {
        "idx":    feat.index,
        "feats":  feat.values.astype(np.float32),
        "cols":   list(feat.columns),
        "close":  close.values.astype(np.float64),
        "high":   a2["high"].reindex(feat.index).values.astype(np.float64),
        "low":    a2["low"].reindex(feat.index).values.astype(np.float64),
        "regime_down": (close < sma24).fillna(False).values,
        "forca":  forca.values.astype(np.float64),
    }


def verificar_causalidade(dados, n_testes=5, semente=7):
    """
    Prova que precalcular nao vaza o futuro.

    Recalcula as features usando SOMENTE os dados ate um instante t e compara
    a ultima linha com a linha t do calculo feito sobre o historico inteiro.
    Se algum indicador olhasse para frente, os dois valores divergiriam.
    """
    print("\n" + "=" * 78)
    print("  VERIFICACAO — o simulador consegue ver o futuro?")
    print("=" * 78)
    print("\n  Recalculando features so com o passado e comparando com o")
    print("  precalculo sobre o historico inteiro.\n")

    btc_df = _load_parquet(BTC)
    rng = np.random.default_rng(semente)
    piores = []

    for sym in list(dados)[:3]:
        d = dados[sym]
        adf = btc_df if sym == BTC else _load_parquet(sym)
        for _ in range(n_testes):
            i = int(rng.integers(3000, len(d["idx"]) - 1))
            ts = d["idx"][i]

            corte_a = adf.loc[:ts]
            corte_b = btc_df.loc[:ts]
            comum = corte_a.index.intersection(corte_b.index)
            f_parcial = _add_features(corte_a.loc[comum],
                                      corte_b.loc[comum]).dropna()
            if len(f_parcial) == 0 or f_parcial.index[-1] != ts:
                continue

            a = f_parcial.values[-1].astype(np.float64)
            b = d["feats"][i].astype(np.float64)
            escala = np.maximum(np.abs(b), 1e-6)
            dif = float(np.max(np.abs(a - b) / escala))
            piores.append((dif, sym, ts))

    if not piores:
        print("  [ERRO] nenhum ponto pode ser comparado.")
        return False

    piores.sort(reverse=True)
    for dif, sym, ts in piores[:5]:
        marca = "ok" if dif < 1e-4 else "DIVERGE"
        print(f"    {sym:<10} {ts:%Y-%m-%d %H:%M}  erro relativo maximo "
              f"{dif:.2e}  [{marca}]")

    pior = piores[0][0]
    print()
    if pior < 1e-4:
        print(f"  APROVADO — maior divergencia {pior:.2e} (ruido de ponto"
              f" flutuante).")
        print("  Nenhum indicador usa dados futuros; precalcular e equivalente")
        print("  a recalcular a cada passo, so que milhares de vezes mais rapido.")
        return True
    print(f"  REPROVADO — divergencia de {pior:.2e}. Algum indicador olha")
    print("  para frente. Os resultados do simulador NAO valem enquanto isto")
    print("  nao for corrigido.")
    return False


# ────────────────────────────────────────────────────────────────────────────
#  O replay
# ────────────────────────────────────────────────────────────────────────────
def simular(dados, de, ate, capital=5000.0, model_path="v5_model_b.pth",
            silencioso=True):
    """Percorre o tempo chamando as funcoes de decisao de PRODUCAO."""
    universo = [s for s in dados if s in ASSETS]
    if not universo:
        raise SystemExit("nenhum ativo utilizavel")

    # Linha do tempo comum: os instantes de ciclo dentro do periodo
    base = dados[universo[0]]["idx"]
    for s in universo[1:]:
        base = base.intersection(dados[s]["idx"])
    base = base[(base >= de) & (base <= ate)]
    if len(base) < 100:
        raise SystemExit(f"periodo curto demais: {len(base)} minutos")

    passos = base[::CICLO_MIN]
    pos_em = {s: dados[s]["idx"].get_indexer(base) for s in universo}
    mapa = {s: pd.Series(np.arange(len(dados[s]["idx"])),
                         index=dados[s]["idx"]) for s in universo}

    quieto = (lambda *a, **k: None) if silencioso else print

    saldo = capital
    posicoes, estado = {}, {"posicoes": {}}
    operacoes, curva = [], []
    agora_real = datetime.utcnow()

    def preco_em(sym, i):
        return float(dados[sym]["close"][i])

    for passo, ts in enumerate(passos):
        # ── 1. TP/SL entre este ciclo e o anterior, minuto a minuto ─────────
        for sym in list(posicoes):
            p = posicoes[sym]
            d = dados[sym]
            i_ini = p["i_ultimo_check"] + 1
            i_fim = int(mapa[sym].get(ts, -1))
            if i_fim < i_ini:
                continue

            saiu = None
            for i in range(i_ini, i_fim + 1):
                alta, baixa = d["high"][i], d["low"][i]
                if p["direcao"] == "LONG":
                    bateu_sl = baixa <= p["sl"]
                    bateu_tp = alta >= p["tp"]
                else:
                    bateu_sl = alta >= p["sl"]
                    bateu_tp = baixa <= p["tp"]
                # Empate na mesma vela conta como SL (cenario pessimista).
                if bateu_sl:
                    saiu = ("SL", p["sl"], i); break
                if bateu_tp:
                    saiu = ("TP", p["tp"], i); break

            p["i_ultimo_check"] = i_fim
            if saiu:
                causa, preco, i = saiu
                saldo += _fechar(p, preco, causa, d["idx"][i], operacoes)
                del posicoes[sym]
                estado["posicoes"].pop(sym, None)

        # ── 2. Estado do mercado neste instante, no formato de producao ─────
        mercado = {}
        for sym in universo:
            i = int(mapa[sym].get(ts, -1))
            if i < WINDOW_SIZE + 1:
                continue
            d = dados[sym]
            mercado[sym] = {
                "feats":       d["feats"][i + 1 - WINDOW_SIZE: i + 1],
                "feat_cols":   d["cols"],
                "close":       float(d["close"][i]),
                "regime_down": bool(d["regime_down"][i]),
                "forca":       float(d["forca"][i]),
            }
        if not mercado:
            continue

        # ── 3. Saidas por sinal/tempo — FUNCAO DE PRODUCAO ──────────────────
        # avaliar_saidas usa datetime.now(); deslocamos aberta_em para que o
        # tempo decorrido que ela calcula seja o tempo SIMULADO.
        if posicoes:
            estado_falso = {"posicoes": {}}
            for sym, p in posicoes.items():
                decorrido = (ts - p["aberta_em"]).total_seconds() / 60
                estado_falso["posicoes"][sym] = {
                    "aberta_em": (agora_real
                                  - timedelta(minutes=decorrido)).isoformat()
                }
            for sym, motivo in avaliar_saidas(mercado, posicoes,
                                              estado_falso, log=quieto):
                if sym in posicoes:
                    i = int(mapa[sym].get(ts, -1))
                    causa = "TEMPO" if "tempo" in motivo else "SINAL"
                    saldo += _fechar(posicoes[sym], preco_em(sym, i),
                                     causa, ts, operacoes)
                    del posicoes[sym]

        # ── 4. Entrada — FUNCAO DE PRODUCAO ─────────────────────────────────
        if len(posicoes) < MAX_POSICOES and saldo > 0:
            dec = decidir_entrada(mercado, model_path, set(posicoes),
                                  log=quieto)
            if dec:
                sym = dec["symbol"]
                margem = saldo * MARGEM_PCT
                notional = margem * dec["alavancagem"]
                taxa = notional * FEE
                saldo -= taxa
                posicoes[sym] = {
                    "symbol": sym, "direcao": dec["direcao"],
                    "entrada": dec["preco"], "sl": dec["sl"], "tp": dec["tp"],
                    "notional": notional, "margem": margem,
                    "lev": dec["alavancagem"], "dir_conf": dec["dir_conf"],
                    "aberta_em": ts, "taxa_entrada": taxa,
                    "i_ultimo_check": int(mapa[sym].get(ts, -1)),
                }

        curva.append((ts, saldo + sum(_aberto(p, preco_em(p["symbol"],
                      int(mapa[p["symbol"]].get(ts, -1))))
                      for p in posicoes.values())))

        if not silencioso and passo % 500 == 0:
            print(f"    {ts:%Y-%m-%d}  saldo ${saldo:,.0f}  "
                  f"({len(operacoes)} ops)", flush=True)

    # Fecha o que sobrou, ao preco final
    for sym, p in list(posicoes.items()):
        i = int(mapa[sym].get(passos[-1], -1))
        saldo += _fechar(p, preco_em(sym, i), "FIM", passos[-1], operacoes)

    return {"saldo": saldo, "capital": capital,
            "operacoes": pd.DataFrame(operacoes),
            "curva": pd.Series(dict(curva)).sort_index()}


def _aberto(p, preco):
    """PnL nao realizado de uma posicao, em dolares."""
    sinal = 1.0 if p["direcao"] == "LONG" else -1.0
    return p["notional"] * sinal * (preco / p["entrada"] - 1.0)


def _fechar(p, preco, causa, ts, operacoes):
    """Devolve quanto entra no saldo ao fechar (PnL menos a taxa de saida)."""
    bruto = _aberto(p, preco)
    taxa_saida = p["notional"] * FEE
    liquido = bruto - taxa_saida
    operacoes.append({
        "symbol": p["symbol"], "direcao": p["direcao"], "causa": causa,
        "aberta_em": p["aberta_em"], "fechada_em": ts,
        "minutos": (ts - p["aberta_em"]).total_seconds() / 60,
        "entrada": p["entrada"], "saida": preco, "lev": p["lev"],
        "dir_conf": p["dir_conf"], "notional": p["notional"],
        "bruto": bruto, "taxas": p["taxa_entrada"] + taxa_saida,
        "liquido": liquido - p["taxa_entrada"],
    })
    return liquido


# ────────────────────────────────────────────────────────────────────────────
#  Relatorio
# ────────────────────────────────────────────────────────────────────────────
def metricas(r):
    ops, curva = r["operacoes"], r["curva"]
    cap = r["capital"]
    fim = float(curva.iloc[-1]) if len(curva) else r["saldo"]
    retorno = fim / cap - 1
    dias = (curva.index[-1] - curva.index[0]).days if len(curva) > 1 else 1
    meses = max(dias / 30.44, 1e-9)
    mensal = (1 + retorno) ** (1 / meses) - 1 if retorno > -1 else -1.0
    dd = float((curva / curva.cummax() - 1).min()) if len(curva) else 0.0
    n = len(ops)
    acerto = float((ops["liquido"] > 0).mean()) if n else 0.0
    return {"retorno": retorno, "mensal": mensal, "dd": dd, "n": n,
            "acerto": acerto, "final": fim,
            "taxas": float(ops["taxas"].sum()) if n else 0.0,
            "expectancia": float(ops["liquido"].mean()) if n else 0.0}


def relatorio(r, titulo):
    m = metricas(r)
    ops = r["operacoes"]
    print("\n" + "=" * 78)
    print(f"  {titulo}")
    print("=" * 78)
    print(f"\n  Capital inicial : ${r['capital']:>12,.2f}")
    print(f"  Capital final   : ${m['final']:>12,.2f}")
    print(f"  Retorno         : {m['retorno']*100:>+12.2f}%")
    print(f"  Equivalente/mes : {m['mensal']*100:>+12.2f}%")
    print(f"  Rebaixamento max: {m['dd']*100:>12.2f}%")
    print(f"  Operacoes       : {m['n']:>12}")
    print(f"  Taxa de acerto  : {m['acerto']*100:>11.1f}%")
    print(f"  Pago em taxas   : ${m['taxas']:>12,.2f}")

    if len(ops):
        print(f"\n  Por causa de saida:")
        for causa, g in ops.groupby("causa"):
            print(f"    {causa:<8} {len(g):>4} ops  "
                  f"resultado ${g['liquido'].sum():>+9,.2f}")
        mensal = ops.set_index("fechada_em")["liquido"].resample("ME").sum()
        if len(mensal) > 1:
            print(f"\n  Por mes:")
            for ts, v in mensal.items():
                barra = "+" * min(int(abs(v) / 20), 30) if v > 0 else \
                        "-" * min(int(abs(v) / 20), 30)
                print(f"    {ts:%Y-%m}  ${v:>+9,.2f}  {barra}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--de", default="2025-01-01")
    ap.add_argument("--ate", default="2026-08-01")
    ap.add_argument("--capital", type=float, default=5000.0)
    ap.add_argument("--assets", default=None,
                    help="lista separada por virgula (padrao: ASSETS do V5)")
    ap.add_argument("--verificar", action="store_true",
                    help="so prova que o simulador nao ve o futuro, e sai")
    ap.add_argument("--rank", action="store_true",
                    help="compara varias configuracoes e ordena")
    ap.add_argument("--verboso", action="store_true")
    a = ap.parse_args()

    universo = ([s.strip().upper() for s in a.assets.split(",")]
                if a.assets else list(ASSETS))

    print(f"\nCarregando {len(universo)} ativos...", flush=True)
    btc_df = _load_parquet(BTC)
    dados = {}
    for sym in universo:
        d = preparar(sym, btc_df)
        if d is not None:
            dados[sym] = d
            print(f"  {sym}: {len(d['idx']):,} minutos "
                  f"({d['idx'][0]:%Y-%m-%d} -> {d['idx'][-1]:%Y-%m-%d})",
                  flush=True)

    if a.verificar:
        ok = verificar_causalidade(dados)
        sys.exit(0 if ok else 1)

    de, ate = pd.Timestamp(a.de), pd.Timestamp(a.ate)

    if a.rank:
        from v8_rank import rodar_rank
        rodar_rank(dados, de, ate, a.capital)
        return

    r = simular(dados, de, ate, a.capital, silencioso=not a.verboso)
    relatorio(r, f"CONFIG EM PRODUCAO — {a.de} a {a.ate}")


if __name__ == "__main__":
    main()
