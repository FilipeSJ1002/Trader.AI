"""
v5_live.py — Trader.AI V5.9: Motor Hibrido em Tempo Real (PAPER TRADING)
=========================================================================

Roda a estrategia hibrida V1+NN ao vivo com precos reais da Binance,
mas com ordens SIMULADAS (paper). Mesmas regras do v5_backtest.py:

  ENTRADA : V1 score >= 60 + NN dir_conf >= 52% + regime SMA 24h
  SAIDA   : TP +1.0% | SL -0.5% (scan minuto a minuto) | sinal V1 | 6h max
  TAMANHO : 20% do capital por operacao | alavancagem 1-5x por confianca

Estado persistente: v5_live_state.json (sobrevive a reinicios)
Diario de operacoes: v5_live_trades.csv
Log: v5_live.log

Uso:
  python v5_live.py                 -> loop continuo (avalia a cada 15min)
  python v5_live.py --once          -> um unico ciclo (teste)
  python v5_live.py --reset         -> zera o estado (capital volta ao inicial)
  python v5_live.py --model v5_model_wf3.pth   -> usa outro modelo
"""
import os
import sys
import csv
import json
import time
import argparse
import requests
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timezone

# Console oculto do Windows usa cp1252 — evita crash de encoding em print()
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from v5_model import load_model
from v5_data_prep import ASSETS, BTC, WINDOW_SIZE, _add_features
from v5_backtest import (v1_scores, leverage_for, _calc_pnl,
                         FEE, MARGIN_PCT, LEV_LABELS)

BASE        = os.path.dirname(os.path.abspath(__file__))
STATE_PATH  = os.path.join(BASE, "v5_live_state.json")
TRADES_CSV  = os.path.join(BASE, "v5_live_trades.csv")
LOG_PATH    = os.path.join(BASE, "v5_live.log")

# ── Parametros (identicos ao backtest validado) ─────────────────────────────
MODEL_PATH     = "v5_model_b.pth"
START_USD      = 10_000.0
V1_BUY_THRESH  = 60
V1_SELL_THRESH = 60
DIRCONF_MIN    = 0.52
SL_PCT         = 0.005
TP_PCT         = 0.010
MAX_LEV        = 5.0
MAX_HOLD_MIN   = 360
INTERVAL_MIN   = 15
KLINES_NEED    = 1800     # SMA 24h (1440) + folga p/ EMA200 e dropna

API = "https://api.binance.com/api/v3/klines"


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ── Dados de mercado (REST publica — sem API key) ───────────────────────────

def fetch_klines(symbol: str, total: int = KLINES_NEED) -> pd.DataFrame:
    """Busca os ultimos `total` candles de 1m (paginado, max 1000/chamada)."""
    frames = []
    end_time = None
    remaining = total
    while remaining > 0:
        params = {"symbol": symbol, "interval": "1m",
                  "limit": min(remaining, 1000)}
        if end_time:
            params["endTime"] = end_time
        r = requests.get(API, params=params, timeout=15)
        r.raise_for_status()
        rows = r.json()
        if not rows:
            break
        frames.append(rows)
        end_time = rows[0][0] - 1          # pagina para tras
        remaining -= len(rows)
        if len(rows) < params["limit"]:
            break

    raw = [row for chunk in reversed(frames) for row in chunk]
    df = pd.DataFrame(raw, columns=[
        "t", "open", "high", "low", "close", "volume",
        "ct", "qav", "n", "tb", "tq", "ig"])
    df["date"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("date", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # Descarta o candle em formacao (ainda nao fechou)
    return df.iloc[:-1]


def build_market(symbols):
    """Baixa klines de todos os ativos + BTC e calcula features.
    Usa o gerador definido em _FEAT_FN (v5=18 features, v6=26)."""
    feat_fn = globals().get("_FEAT_FN", _add_features)
    btc_df = fetch_klines(BTC)
    market = {}
    for s in symbols:
        adf = btc_df if s == BTC else fetch_klines(s)
        common = adf.index.intersection(btc_df.index)
        adf_c, btc_c = adf.loc[common], btc_df.loc[common]
        feat_df = feat_fn(adf_c, btc_c).dropna()
        if len(feat_df) < WINDOW_SIZE + 2:
            log(f"  [{s}] features insuficientes ({len(feat_df)}) — pulando ciclo")
            continue
        sma24h = adf_c["close"].rolling(1440).mean().reindex(feat_df.index)
        close_al = adf_c["close"].reindex(feat_df.index)
        market[s] = {
            "ohlc":        adf_c,
            "feats":       feat_df.values.astype(np.float32),
            "feat_cols":   list(feat_df.columns),
            "feat_index":  feat_df.index,
            "close":       float(close_al.iloc[-1]),
            "regime_down": bool(close_al.iloc[-1] < sma24h.iloc[-1])
                           if not np.isnan(sma24h.iloc[-1]) else False,
        }
    return market


# ── Estado persistente ───────────────────────────────────────────────────────

def load_state(reset=False):
    if reset or not os.path.exists(STATE_PATH):
        return {"cap": START_USD, "positions": {}, "n_trades": 0,
                "started": datetime.now(timezone.utc).isoformat()}
    with open(STATE_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_state(state):
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def record_trade(row: dict):
    new_file = not os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)


# ── Ciclo principal ──────────────────────────────────────────────────────────

def check_exits(state, market):
    """Verifica TP/SL (minuto a minuto), sinal contrario e tempo maximo."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    for sym in list(state["positions"].keys()):
        p = state["positions"][sym]
        if sym not in market:
            continue
        m    = market[sym]
        ohlc = m["ohlc"]

        entry_ts = pd.Timestamp(p["entry_ts"])
        since    = ohlc[ohlc.index > entry_ts]
        if since.empty:
            continue

        exit_price = None
        cause      = None

        # 1. TP/SL intrabar (SL prioritario — conservador)
        for ts, candle in since.iterrows():
            if p["direction"] == "LONG":
                if candle["low"] <= p["sl_price"]:
                    exit_price, cause = p["sl_price"], "SL"; break
                if candle["high"] >= p["tp_price"]:
                    exit_price, cause = p["tp_price"], "TP"; break
            else:
                if candle["high"] >= p["sl_price"]:
                    exit_price, cause = p["sl_price"], "SL"; break
                if candle["low"] <= p["tp_price"]:
                    exit_price, cause = p["tp_price"], "TP"; break

        # 2. Sinal V1 contrario ou tempo maximo
        if exit_price is None:
            feats = m["feats"]
            b_sc, s_sc, _ = v1_scores(feats[-1], feats[-2], m["feat_cols"])
            hold_min = (now - entry_ts.to_pydatetime()).total_seconds() / 60
            opposite = (s_sc >= V1_SELL_THRESH if p["direction"] == "LONG"
                        else b_sc >= V1_BUY_THRESH)
            if opposite:
                exit_price, cause = m["close"], "SINAL"
            elif hold_min >= MAX_HOLD_MIN:
                exit_price, cause = m["close"], "TEMPO"

        if exit_price is None:
            continue

        low_min  = float(since["low"].min())
        high_max = float(since["high"].max())
        pnl, outcome = _calc_pnl(p["direction"], p["entry"], exit_price,
                                 low_min, high_max, p["margin"], p["lev"])
        state["cap"]      += pnl
        state["n_trades"] += 1

        hold_min = int((now - entry_ts.to_pydatetime()).total_seconds() / 60)
        trade = {
            "n": state["n_trades"], "sym": sym, "direction": p["direction"],
            "entry_ts": p["entry_ts"], "exit_ts": now.isoformat(),
            "hold_min": hold_min, "entry": p["entry"], "exit": round(exit_price, 6),
            "conf": p["conf"], "lev": p["lev"], "margin": round(p["margin"], 2),
            "pnl": round(pnl, 2), "cause": cause, "outcome": outcome,
            "cap": round(state["cap"], 2),
        }
        record_trade(trade)
        del state["positions"][sym]

        icon = "+" if outcome == "WIN" else "-"
        log(f"  [{icon}] FECHOU {p['direction']} {sym} | {cause} | "
            f"entrada {p['entry']:.4f} -> saida {exit_price:.4f} "
            f"({hold_min}min) | P&L ${pnl:+.2f} | capital ${state['cap']:,.2f}")


def check_entries(state, market, model, device, universo=None):
    """Procura UMA nova entrada por ciclo (a mais forte), como no backtest."""
    universo = universo or ASSETS
    available = [s for s in market if s not in state["positions"] and s in universo]
    if not available:
        return

    # 1. Filtro V1 + regime (barato)
    candidates = []
    for s in available:
        m = market[s]
        feats = m["feats"]
        b_sc, s_sc, _ = v1_scores(feats[-1], feats[-2], m["feat_cols"])
        rd = m["regime_down"]
        if b_sc >= V1_BUY_THRESH and not rd:
            candidates.append((s, "LONG", b_sc))
        elif s_sc >= V1_SELL_THRESH and rd:
            candidates.append((s, "SHORT", s_sc))

    if not candidates:
        return

    # 2. NN confirma direcao (so para os candidatos)
    windows = [market[s]["feats"][-WINDOW_SIZE:] for s, _, _ in candidates]
    X = torch.tensor(np.stack(windows), dtype=torch.float32).to(device)
    with torch.no_grad():
        logits, _ = model(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    p_up, p_down = probs[:, 2], probs[:, 0]
    p_dir = p_up + p_down + 1e-9

    best = None
    best_strength = 0.0
    for i, (s, direction, v1_sc) in enumerate(candidates):
        dc = float((p_up if direction == "LONG" else p_down)[i] / p_dir[i])
        if dc < DIRCONF_MIN:
            continue
        if v1_sc * dc > best_strength:
            best_strength = v1_sc * dc
            best = (s, direction, v1_sc, dc)

    if best is None:
        return

    sym, direction, v1_sc, dc = best
    lev = leverage_for(dc, True, MAX_LEV)
    if lev == 0.0:
        return

    entry  = market[sym]["close"]
    margin = state["cap"] * MARGIN_PCT
    if direction == "LONG":
        sl, tp = entry * (1 - SL_PCT), entry * (1 + TP_PCT)
    else:
        sl, tp = entry * (1 + SL_PCT), entry * (1 - TP_PCT)

    state["positions"][sym] = {
        "direction": direction, "entry": entry,
        "entry_ts": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
        "margin": margin, "lev": lev, "conf": round(dc, 4),
        "sl_price": sl, "tp_price": tp,
    }
    log(f"  [>] ABRIU {direction} {sym} @ {entry:.4f} | "
        f"V1={v1_sc} dir_conf={dc*100:.1f}% -> {LEV_LABELS.get(lev, str(lev)+'x').strip()} | "
        f"margem ${margin:,.2f} | SL {sl:.4f} TP {tp:.4f}")


def cycle(state, model, device, universo=None):
    universo = universo or ASSETS
    symbols = set(universo) | {BTC} | set(state["positions"].keys())
    market = build_market(sorted(symbols))
    if not market:
        log("  Sem dados de mercado neste ciclo.")
        return

    check_exits(state, market)
    check_entries(state, market, model, device, universo)
    save_state(state)

    pos_str = ", ".join(
        f"{s}:{p['direction']}" for s, p in state["positions"].items()) or "nenhuma"
    ret = (state["cap"] / START_USD - 1) * 100
    log(f"  capital ${state['cap']:,.2f} ({ret:+.2f}%) | "
        f"posicoes: {pos_str} | trades: {state['n_trades']}")


def main():
    ap = argparse.ArgumentParser(description="Trader.AI V5.9 — Live Paper Trading")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--once",  action="store_true", help="Um ciclo e sai")
    ap.add_argument("--reset", action="store_true", help="Zera capital e posicoes")
    ap.add_argument("--interval", type=int, default=INTERVAL_MIN,
                    help="Minutos entre ciclos (padrao 15)")
    ap.add_argument("--gpu", action="store_true",
                    help="Usa GPU (padrao CPU — nao disputa com treinos)")
    ap.add_argument("--featset", choices=["v5", "v6"], default="v5",
                    help="Conjunto de features (deve casar com o modelo)")
    ap.add_argument("--assets", default=None,
                    help="Universo de ativos separado por virgula. "
                         "'ALL' usa todos os parquets de data/. Padrao: os 6 do treino")
    args = ap.parse_args()

    device = "cuda" if (args.gpu and torch.cuda.is_available()) else "cpu"

    # Gerador de features compativel com o modelo
    if args.featset == "v6":
        from v6_data_prep import add_features_v6 as feat_fn
    else:
        feat_fn = _add_features
    globals()["_FEAT_FN"] = feat_fn

    universo = None
    if args.assets:
        if args.assets.upper() == "ALL":
            import glob as _glob
            universo = sorted(os.path.basename(p).replace("_1m.parquet", "")
                              for p in _glob.glob("data/*_1m.parquet"))
        else:
            universo = [s.strip().upper() for s in args.assets.split(",")]

    state = load_state(reset=args.reset)
    log("=" * 64)
    log(f"LIVE PAPER | modelo {args.model} | featset {args.featset} | device {device}")
    log(f"Universo: {len(universo) if universo else len(ASSETS)} ativos"
        + (f" ({', '.join(universo)})" if universo else ""))
    log(f"Capital ${state['cap']:,.2f} | {len(state['positions'])} posicao(oes) aberta(s)")
    log("=" * 64)

    # Descobre n_features com um fetch leve do BTC
    probe = fetch_klines(BTC, total=KLINES_NEED)
    probe_feats = feat_fn(probe, probe).dropna()
    n_features = probe_feats.shape[1]
    model = load_model(args.model, n_features, device)
    log(f"Modelo carregado ({n_features} features)")

    while True:
        try:
            cycle(state, model, device, universo)
        except KeyboardInterrupt:
            log("Interrompido pelo usuario. Estado salvo.")
            break
        except Exception as e:
            log(f"[ERRO] ciclo falhou: {e} — tenta de novo no proximo")

        if args.once:
            break

        # Alinha o proximo ciclo ao relogio (ex: :00, :15, :30, :45)
        now = time.time()
        step = args.interval * 60
        time.sleep(step - (now % step) + 2)


if __name__ == "__main__":
    main()
