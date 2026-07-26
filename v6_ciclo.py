# -*- coding: utf-8 -*-
"""
v6_ciclo.py — Trader.AI V6: a decisao de um ciclo de operacao
==============================================================

Ponte entre a ESTRATEGIA (validada em backtest) e a EXECUCAO (ordens reais).

Princípio de projeto: este modulo NAO reimplementa a estrategia. Ele importa
as mesmas funcoes usadas pelo v5_backtest.py e pelo v5_live.py — `v1_scores`,
`leverage_for` — garantindo que a logica que opera dinheiro seja bit a bit a
mesma que foi validada. Qualquer divergencia aqui invalidaria os backtests.

Fluxo de um ciclo:
  1. Busca candles recentes (REST publica, sem autenticacao)
  2. Calcula as features com a MESMA funcao do treino
  3. Aplica os tres portoes: gatilho V1 -> regime SMA24h -> confirmacao neural
  4. Devolve a decisao ao executor, que traduz em ordens

O fechamento de posicoes NAO e responsabilidade deste modulo: TP e SL vivem
como ordens condicionais na propria corretora (ver v6_executor). Aqui tratamos
apenas saidas por sinal contrario e por tempo maximo.
"""
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone

from v5_model import load_model
from v5_data_prep import ASSETS, BTC, WINDOW_SIZE, _add_features, _load_parquet
from v5_backtest import v1_scores, leverage_for

# Parametros da estrategia — identicos aos validados no backtest
V1_BUY_THRESH  = 60
V1_SELL_THRESH = 60
DIRCONF_MIN    = 0.52
SL_PCT         = 0.005
TP_PCT         = 0.010
MAX_LEV        = 5.0
MAX_HOLD_MIN   = 360
LEV_CURVE      = "v59"
KLINES_NEED    = 1800     # SMA 24h (1440) + folga para EMA200 e dropna

API = "https://api.binance.com/api/v3/klines"
_modelo_cache = {}


def _fetch_klines(symbol, total=KLINES_NEED):
    """Candles de 1m via REST publica (mesma fonte do v5_live)."""
    import requests
    frames, end_time, restante = [], None, total
    while restante > 0:
        params = {"symbol": symbol, "interval": "1m", "limit": min(restante, 1000)}
        if end_time:
            params["endTime"] = end_time
        r = requests.get(API, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        frames.append(rows)
        end_time = rows[0][0] - 1
        restante -= len(rows)
        if len(rows) < params["limit"]:
            break

    raw = [row for chunk in reversed(frames) for row in chunk]
    df = pd.DataFrame(raw, columns=["t", "open", "high", "low", "close", "volume",
                                    "ct", "qav", "n", "tb", "tq", "ig"])
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df.iloc[:-1]        # descarta o candle em formacao


def _montar_mercado(simbolos):
    """Baixa candles e calcula features + regime para cada ativo."""
    btc = _fetch_klines(BTC)
    mercado = {}
    for s in simbolos:
        adf = btc if s == BTC else _fetch_klines(s)
        comum = adf.index.intersection(btc.index)
        a2, b2 = adf.loc[comum], btc.loc[comum]
        feat = _add_features(a2, b2).dropna()
        if len(feat) < WINDOW_SIZE + 2:
            continue
        close = a2["close"].reindex(feat.index)
        sma24 = a2["close"].rolling(1440).mean().reindex(feat.index)
        mercado[s] = {
            "feats":     feat.values.astype(np.float32),
            "feat_cols": list(feat.columns),
            "close":     float(close.iloc[-1]),
            "regime_down": bool(close.iloc[-1] < sma24.iloc[-1])
                           if not np.isnan(sma24.iloc[-1]) else False,
        }
    return mercado


def _carregar_modelo(model_path, n_features, device="cpu"):
    chave = (model_path, n_features, device)
    if chave not in _modelo_cache:
        _modelo_cache[chave] = load_model(model_path, n_features, device)
    return _modelo_cache[chave]


def decidir_entrada(mercado, model_path, ja_abertas, log=print):
    """
    Aplica os tres portoes da estrategia e devolve UMA decisao (ou None).
    Identico ao v5_backtest: gatilho V1 -> regime -> confirmacao neural.
    """
    disponiveis = [s for s in mercado if s in ASSETS and s not in ja_abertas]
    if not disponiveis:
        return None

    # Portao 1 e 2 — gatilho tecnico + regime diario (barato, sem GPU)
    candidatos = []
    for s in disponiveis:
        m = mercado[s]
        b_sc, s_sc, _ = v1_scores(m["feats"][-1], m["feats"][-2], m["feat_cols"])
        rd = m["regime_down"]
        if b_sc >= V1_BUY_THRESH and not rd:
            candidatos.append((s, "LONG", b_sc))
        elif s_sc >= V1_SELL_THRESH and rd:
            candidatos.append((s, "SHORT", s_sc))

    if not candidatos:
        log("  Nenhum gatilho tecnico neste ciclo.")
        return None

    log(f"  {len(candidatos)} candidato(s) do V1: "
        + ", ".join(f"{s}/{d}({sc})" for s, d, sc in candidatos))

    # Portao 3 — confirmacao neural
    janelas = [mercado[s]["feats"][-WINDOW_SIZE:] for s, _, _ in candidatos]
    modelo = _carregar_modelo(model_path, janelas[0].shape[1])
    X = torch.tensor(np.stack(janelas), dtype=torch.float32)
    with torch.no_grad():
        logits, _ = modelo(X)
        probs = torch.softmax(logits, 1).numpy()

    p_up, p_dn = probs[:, 2], probs[:, 0]
    p_dir = p_up + p_dn + 1e-9

    melhor, forca_melhor = None, 0.0
    for i, (s, direcao, v1_sc) in enumerate(candidatos):
        dc = float((p_up[i] if direcao == "LONG" else p_dn[i]) / p_dir[i])
        if dc < DIRCONF_MIN:
            log(f"    {s}: dir_conf {dc:.3f} < {DIRCONF_MIN} — vetado pela rede")
            continue
        forca = v1_sc * dc
        log(f"    {s}: dir_conf {dc:.3f} — aprovado (forca {forca:.1f})")
        if forca > forca_melhor:
            forca_melhor, melhor = forca, (s, direcao, v1_sc, dc)

    if melhor is None:
        return None

    symbol, direcao, v1_sc, dc = melhor
    lev = leverage_for(dc, True, MAX_LEV, LEV_CURVE)
    if lev == 0.0:
        return None

    preco = mercado[symbol]["close"]
    if direcao == "LONG":
        sl, tp = preco * (1 - SL_PCT), preco * (1 + TP_PCT)
    else:
        sl, tp = preco * (1 + SL_PCT), preco * (1 - TP_PCT)

    return {"symbol": symbol, "direcao": direcao, "preco": preco,
            "alavancagem": lev, "dir_conf": dc, "v1_score": v1_sc,
            "sl": sl, "tp": tp}


def avaliar_saidas(mercado, posicoes, estado, log=print):
    """
    Saidas por SINAL contrario ou TEMPO maximo.
    TP e SL nao entram aqui — vivem como ordens na corretora.
    """
    fechar = []
    agora = datetime.now(timezone.utc).replace(tzinfo=None)
    for symbol, pos in posicoes.items():
        if symbol not in mercado:
            continue
        m = mercado[symbol]
        b_sc, s_sc, _ = v1_scores(m["feats"][-1], m["feats"][-2], m["feat_cols"])

        contrario = (s_sc >= V1_SELL_THRESH if pos["direcao"] == "LONG"
                     else b_sc >= V1_BUY_THRESH)
        if contrario:
            fechar.append((symbol, "sinal tecnico contrario"))
            continue

        aberta_em = estado.get("posicoes", {}).get(symbol, {}).get("aberta_em")
        if aberta_em:
            try:
                t0 = datetime.fromisoformat(aberta_em).replace(tzinfo=None)
                minutos = (agora - t0).total_seconds() / 60
                if minutos >= MAX_HOLD_MIN:
                    fechar.append((symbol, f"tempo maximo ({int(minutos)}min)"))
            except Exception:
                pass
    return fechar


def _ler_estado(path):
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"posicoes": {}, "historico": []}


def _salvar_estado(path, estado):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2, default=str)


import json  # usado por _ler_estado/_salvar_estado


def executar_ciclo(ex, saldo, model_path="v5_model_b.pth",
                   margem_pct=0.20, log=print):
    """Um ciclo completo: avalia saidas, depois procura uma entrada."""
    estado_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "v6_executor_state.json")
    estado = _ler_estado(estado_path)

    posicoes = ex.posicoes_abertas()          # fonte da verdade: a corretora
    simbolos = sorted(set(ASSETS) | {BTC} | set(posicoes.keys()))

    log("  Buscando dados de mercado...")
    mercado = _montar_mercado(simbolos)
    if not mercado:
        log("  Sem dados de mercado neste ciclo.")
        return

    # 1. Saidas
    for symbol, motivo in avaliar_saidas(mercado, posicoes, estado, log):
        ex.fechar_posicao(symbol, motivo=motivo)
        estado.get("posicoes", {}).pop(symbol, None)
        estado.setdefault("historico", []).append({
            "symbol": symbol, "evento": "fechamento", "motivo": motivo,
            "quando": datetime.now(timezone.utc).isoformat(),
        })

    # 2. Entrada (uma por ciclo, como no backtest)
    posicoes = ex.posicoes_abertas()
    pode, motivo = ex.pode_abrir(saldo)
    if not pode:
        log(f"  Entrada bloqueada: {motivo}")
        _salvar_estado(estado_path, estado)
        return

    decisao = decidir_entrada(mercado, model_path, set(posicoes.keys()), log)
    if decisao is None:
        log("  Nenhuma entrada aprovada neste ciclo.")
        _salvar_estado(estado_path, estado)
        return

    log(f"  DECISAO: {decisao['direcao']} {decisao['symbol']} "
        f"| V1={decisao['v1_score']} dir_conf={decisao['dir_conf']:.3f} "
        f"-> {decisao['alavancagem']:.0f}x")

    aberta = ex.abrir_posicao(
        symbol=decisao["symbol"], direcao=decisao["direcao"],
        margem_usd=saldo * margem_pct, alavancagem=decisao["alavancagem"],
        sl_price=decisao["sl"], tp_price=decisao["tp"],
        preco_ref=decisao["preco"],
    )
    if aberta:
        estado.setdefault("posicoes", {})[decisao["symbol"]] = aberta
        estado.setdefault("historico", []).append({
            "symbol": decisao["symbol"], "evento": "abertura",
            "direcao": decisao["direcao"], "dir_conf": decisao["dir_conf"],
            "quando": datetime.now(timezone.utc).isoformat(),
        })
    _salvar_estado(estado_path, estado)
