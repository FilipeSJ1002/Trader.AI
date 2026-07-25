"""
v5_backtest.py — Trader.AI V5 (Hibrido V1+V5 com Gestao de Posicao)
====================================================================

Estrategia Hibrida:
  ENTRADA : V1 buy_score >= threshold (RSI+MACD+BB+EMA200) E NN confirma direcao
  SAIDA   : V1 sell_score >= threshold (sinal tecnico de exaustao) OU tempo maximo
            --> Saida dinamica = captura o pico do recovery, nao deixa inverter

Por que isso funciona melhor que saida fixa em 2h:
  V1 entra na minima local (RSI<30 + preco no BB inferior)
  V1 sai quando preco se recupera (RSI>65 + BB superior + MACD caindo)
  O tempo ate a recuperacao varia: 15min a 4h — saida fixa em 2h perde o timing

SHORT em bear market:
  Quando preco < EMA200 (regime de baixa), sell_score >= threshold → SHORT
  Saida quando buy_score >= threshold OU tempo maximo

Uso:
  python v5_backtest.py                           -> teste (pos VAL_END)
  python v5_backtest.py --val                     -> validacao
  python v5_backtest.py --no-short                -> apenas LONG
  python v5_backtest.py --no-lev                  -> sem alavancagem
  python v5_backtest.py --from 2026-04-01 --to 2026-05-05 --realop
  python v5_backtest.py --eval-step 5 --max-hold 240   -> mais sinais, saida ate 4h
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from datetime import date, timedelta

from v5_model import load_model, get_device
from v5_data_prep import (ASSETS, BTC, WINDOW_SIZE, HORIZON_H,
                          TRAIN_END, VAL_END,
                          _load_parquet, _add_features)

MODEL_PATH   = "v5_model.pth"
FEE          = 0.0004     # 0.04% por lado (taxa Binance Futures)
START_USD    = 10_000.0
MARGIN_PCT   = 0.20       # 20% do capital por operacao
MAINT_MARGIN = 0.005      # margem de manutencao (liquidacao)

LEV_LABELS = {
    20.0: "20x FUTUROS",
    10.0: "10x FUTUROS",
    5.0:  " 5x FUTUROS",
    2.0:  " 2x FUTUROS",
    1.0:  " 1x SPOT    ",
}


def leverage_for(dir_conf: float, use_lev: bool, max_lev: float = 5.0) -> float:
    """
    Converte confianca direcional do NN em alavancagem.
    dir_conf = p_direcao / (p_alta + p_queda)  [0.5 = empate, 1.0 = certeza]

    max_lev limita a alavancagem: os backtests mostraram que a faixa 62-67%
    (5x) e a unica consistentemente lucrativa — confiancas de 67%+ nao sao
    calibradas (poucas amostras no treino) e perdem dinheiro em 10x/20x.
    """
    if not use_lev:
        return 1.0 if dir_conf >= 0.52 else 0.0
    if   dir_conf >= 0.72: lev = 20.0
    elif dir_conf >= 0.67: lev = 10.0
    elif dir_conf >= 0.62: lev =  5.0
    elif dir_conf >= 0.57: lev =  2.0
    elif dir_conf >= 0.52: lev =  1.0
    else:                  lev =  0.0
    return min(lev, max_lev)


def precompute(sym, btc_df, feat_fn=None):
    """Pre-computa features e OHLC alinhados por timestamp.
    feat_fn permite trocar o gerador de features (ex: 26 features da V6)."""
    if feat_fn is None:
        feat_fn = _add_features
    adf    = _load_parquet(sym)
    common = adf.index.intersection(btc_df.index)
    adf, btc_al = adf.loc[common], btc_df.loc[common]
    feat_df = feat_fn(adf, btc_al).dropna()
    feats   = feat_df.values.astype(np.float32)
    idx     = feat_df.index.values.astype("int64")
    close   = adf["close"].reindex(feat_df.index).values.astype(np.float64)
    low     = adf["low"].reindex(feat_df.index).values.astype(np.float64)
    high    = adf["high"].reindex(feat_df.index).values.astype(np.float64)

    # Regime DIARIO: preco vs media movel de 24h (1440 min).
    # A EMA200 de 1m cobre so ~3.3h — inutil como filtro de tendencia.
    # regime_down=True -> mercado em queda no diario -> permite SHORT, bloqueia LONG
    sma_24h = adf["close"].rolling(1440).mean().reindex(feat_df.index)
    regime_down = (adf["close"].reindex(feat_df.index) < sma_24h).fillna(False).values

    return {
        "feats": feats, "idx": idx, "close": close, "low": low,
        "high": high, "ts": feat_df.index,
        "feat_cols": list(feat_df.columns),
        "regime_down": regime_down,
    }


def v1_scores(last: np.ndarray, prev: np.ndarray, cols: list) -> tuple:
    """
    Reconstroi buy_score e sell_score do V1 (strategy.py) a partir das
    features pre-computadas. Sem re-calculo de indicadores.

    Campos usados (normalizados por _add_features):
      rsi:         (RSI_raw - 50) / 50   → RSI_raw = rsi*50 + 50
      macdh:       MACD_hist / close
      dist_bbu:    (BB_upper - close) / close  → <= 0 se close >= BB_upper
      dist_bbl:    (close - BB_lower) / close  → <= 0 se close <= BB_lower
      ema200_dist: (close - EMA200) / close    → < 0 se close < EMA200
      vol_norm:    volume / vol_sma20
    """
    def fi(name):
        return cols.index(name) if name in cols else -1

    i_rsi  = fi("rsi");      i_macdh = fi("macdh")
    i_bbu  = fi("dist_bbu"); i_bbl   = fi("dist_bbl")
    i_ema  = fi("ema200_dist"); i_vol = fi("vol_norm")

    if i_rsi < 0 or i_macdh < 0:
        return 0, 0

    raw_rsi   = float(last[i_rsi])  * 50.0 + 50.0
    macdh     = float(last[i_macdh])
    prev_m    = float(prev[i_macdh])
    dist_bbu  = float(last[i_bbu])  if i_bbu >= 0 else  1.0
    dist_bbl  = float(last[i_bbl])  if i_bbl >= 0 else  1.0
    ema200_d  = float(last[i_ema])  if i_ema >= 0 else  0.0
    vol_norm  = float(last[i_vol])  if i_vol >= 0 else  1.0

    buy_score = sell_score = 0

    # RSI
    if   raw_rsi < 30: buy_score  += 40
    elif raw_rsi < 40: buy_score  += 20
    if   raw_rsi > 75: sell_score += 40
    elif raw_rsi > 65: sell_score += 20

    # Bollinger Bands
    if dist_bbl <= 0: buy_score  += 35   # close <= BB_lower
    if dist_bbu <= 0: sell_score += 35   # close >= BB_upper

    # MACD hist direcao
    if macdh > prev_m: buy_score  += 25
    else:              sell_score += 25

    # Volume bonus
    if vol_norm >= 1.5:
        buy_score  += 15
        sell_score += 10

    # EMA200 filter: mercado em baixa bloqueia entradas LONG
    if ema200_d < 0:
        buy_score = 0

    return buy_score, sell_score, ema200_d


def _calc_pnl(direction, entry, exit_, low_min, high_max, margin, lev):
    """P&L e verificacao de liquidacao para LONG ou SHORT."""
    if direction == "LONG":
        liq_price  = entry * (1 - 1.0 / lev + MAINT_MARGIN)
        liquidated = (low_min <= liq_price)
    else:
        liq_price  = entry * (1 + 1.0 / lev - MAINT_MARGIN)
        liquidated = (high_max >= liq_price)

    if liquidated:
        return -margin, "LIQUIDADO"

    raw_ret  = (exit_ / entry - 1) if direction == "LONG" else (1 - exit_ / entry)
    notional = margin * lev
    pnl      = notional * raw_ret - notional * FEE * 2
    return max(pnl, -margin), ("WIN" if pnl > 0 else "LOSS")


def run_backtest(split="test", use_lev=True, allow_short=True,
                 from_date=None, to_date=None, realop=False,
                 v1_buy_thresh=60, v1_sell_thresh=60,
                 eval_step=15, max_hold_min=360,
                 model_path=MODEL_PATH,
                 sl_pct=0.005, tp_pct=0.010, max_lev=5.0,
                 skip_syms=None, featset="v5"):
    """
    eval_step    : frequencia de avaliacao em minutos (padrao 15)
    max_hold_min : tempo maximo de posicao aberta em minutos (padrao 6h)
    model_path   : caminho do modelo treinado (.pth)
    sl_pct       : stop loss (padrao 0.5%)
    tp_pct       : take profit (padrao 1.0% — R/R 2:1)
    max_lev      : alavancagem maxima (padrao 5x — faixa calibrada do NN)
    skip_syms    : ativos a NAO operar (continuam como contexto, ex: BTC)
    """
    device = get_device()
    skip_syms    = set(skip_syms or [])
    trade_assets = [s for s in ASSETS if s not in skip_syms]

    # Seleciona o conjunto de features compativel com o modelo
    if featset == "v6":
        from v6_data_prep import add_features_v6 as feat_fn
    else:
        feat_fn = _add_features

    _val_start  = str(date.fromisoformat(TRAIN_END) + timedelta(days=1))
    _test_start = str(date.fromisoformat(VAL_END)   + timedelta(days=1))

    if from_date and to_date:
        p_start, p_end = from_date, to_date
        label_split = f"Personalizado ({from_date} -> {to_date})"
    elif split == "val":
        p_start, p_end = _val_start, VAL_END
        label_split = f"Validacao ({_val_start[:10]} -> {VAL_END})"
    else:
        p_start, p_end = _test_start, "2099-12-31"
        label_split = f"Teste ({_test_start[:10]} em diante)"

    dir_str = "LONG+SHORT" if allow_short else "Apenas LONG"
    lev_str = "(ate 20x)" if use_lev else "(spot 1x)"

    print(f"\n{'='*70}")
    print(f"  TRADER.AI V5.8 — HIBRIDO V1+V5 | {dir_str} {lev_str}")
    print(f"  {label_split}")
    print(f"  Capital: ${START_USD:,.0f} | Margem: {MARGIN_PCT*100:.0f}%/op | Fee: {FEE*100:.2f}%/lado")
    print(f"  ENTRADA: V1 score >= {v1_buy_thresh} + NN dir_conf >= 52%")
    print(f"           Regime SMA 24h: alta->so LONG | baixa->so SHORT")
    print(f"  SAIDA  : TP +{tp_pct*100:.1f}% | SL -{sl_pct*100:.1f}% | "
          f"sinal V1 contrario | max {max_hold_min}min")
    print(f"  Alavancagem max: {max_lev:.0f}x | Verifica a cada {eval_step}min")
    print(f"  Modelo: {model_path}")
    if skip_syms:
        print(f"  Ativos excluidos: {', '.join(sorted(skip_syms))} (so contexto)")
    print(f"{'='*70}\n")

    btc_df = _load_parquet(BTC)
    sample = precompute(trade_assets[0], btc_df, feat_fn)
    model  = load_model(model_path, sample["feats"].shape[1], device)

    print(f"Pre-computando features ({featset}: {sample['feats'].shape[1]} cols)...")
    data = {s: precompute(s, btc_df, feat_fn) for s in trade_assets}
    for s in trade_assets:
        print(f"  {s}: {len(data[s]['feats'])} candles")

    period_idx = btc_df.index[(btc_df.index >= p_start) & (btc_df.index <= p_end)]
    if len(period_idx) < max_hold_min * 2:
        print("Dados insuficientes.")
        return {}

    timestamps = period_idx[::eval_step]
    fwd_ns_max = max_hold_min * 60 * 1_000_000_000

    print(f"\nSimulando {len(timestamps)} steps de {eval_step}min "
          f"| saida max {max_hold_min}min "
          f"({period_idx[0].date()} -> {period_idx[-1].date()})...\n")

    cap       = START_USD
    trades    = []
    op_num    = 0
    # sym -> dict com info da posicao aberta
    positions = {}

    if realop:
        print(f"{'─'*70}  DIARIO DE OPERACOES  {'─'*70}")

    for ts in timestamps:
        ts64 = np.int64(ts.value)

        # ── 1. Verificar saidas para posicoes abertas ───────────────────
        for sym in list(positions.keys()):
            p = positions[sym]
            d = data[sym]
            cur = np.searchsorted(d["idx"], ts64)
            if cur >= len(d["idx"]) or d["idx"][cur] != ts64:
                continue
            if cur < 1:
                continue

            lo, hi = d["low"], d["high"]

            # 1a. TP/SL intrabar — varre cada minuto desde a ultima checagem.
            # SL tem prioridade quando ambos disparam no mesmo candle (conservador).
            exit_row = None
            exit_price = None
            cause = None
            scan_start = max(p["scan_from"], p["entry_pos"] + 1)
            for r in range(scan_start, cur + 1):
                if p["direction"] == "LONG":
                    if lo[r] <= p["sl_price"]:
                        exit_row, exit_price, cause = r, p["sl_price"], "SL"
                        break
                    if hi[r] >= p["tp_price"]:
                        exit_row, exit_price, cause = r, p["tp_price"], "TP"
                        break
                else:  # SHORT
                    if hi[r] >= p["sl_price"]:
                        exit_row, exit_price, cause = r, p["sl_price"], "SL"
                        break
                    if lo[r] <= p["tp_price"]:
                        exit_row, exit_price, cause = r, p["tp_price"], "TP"
                        break
            p["scan_from"] = cur + 1

            # 1b. Sinal V1 contrario ou tempo maximo
            if exit_row is None:
                b_sc, s_sc, _ema = v1_scores(d["feats"][cur], d["feats"][cur - 1],
                                             d["feat_cols"])
                expired = (ts64 >= p["deadline"])
                if p["direction"] == "LONG":
                    exit_signal = (s_sc >= v1_sell_thresh)
                else:
                    exit_signal = (b_sc >= v1_buy_thresh)
                if exit_signal or expired:
                    exit_row   = cur
                    exit_price = d["close"][cur]
                    cause      = "SINAL" if exit_signal else "TEMPO"

            if exit_row is None:
                continue

            # Fechar posicao
            entry    = p["entry"]
            ep       = p["entry_pos"]
            low_min  = lo[ep:exit_row + 1].min()  if exit_row > ep else lo[ep]
            high_max = hi[ep:exit_row + 1].max() if exit_row > ep else hi[ep]
            margin   = p["margin"]
            lev      = p["lev"]

            pnl, outcome = _calc_pnl(p["direction"], entry, exit_price,
                                     low_min, high_max, margin, lev)
            cap    += pnl
            op_num += 1

            exit_ts  = pd.Timestamp(d["idx"][exit_row])
            hold_min = int((d["idx"][exit_row] - p["ts64"]) / 60_000_000_000)

            trade = {
                "op": op_num, "ts": p["ts"], "exit_ts": exit_ts,
                "sym": sym, "direction": p["direction"],
                "conf": p["conf"], "lev": lev,
                "entry": entry, "exit": exit_price,
                "pnl": pnl, "cap": cap, "outcome": outcome,
                "cause": cause, "hold_min": hold_min,
            }
            trades.append(trade)
            del positions[sym]

            if realop:
                icon = "+" if outcome == "WIN" else ("!!" if outcome == "LIQUIDADO" else "-")
                di   = "^LONG " if p["direction"] == "LONG" else "vSHORT"
                rm   = (exit_price / entry - 1) * 100 * (1 if p["direction"] == "LONG" else -1)
                print(f"\n  [{icon}] #{op_num:3d} {p['ts'].strftime('%d/%m %H:%M')}->{exit_ts.strftime('%H:%M')} "
                      f"({hold_min}min) [{di}] {sym}")
                print(f"      Entrada ${entry:>10,.4f} | Saida ${exit_price:>10,.4f} "
                      f"({rm:+.2f}%) | {cause}")
                print(f"      NN conf {p['conf']*100:.1f}% -> {LEV_LABELS.get(lev, str(lev)+'x')} "
                      f"| P&L ${pnl:>+8,.2f} | Capital ${cap:>10,.2f}")

        # ── 2. Procurar novas entradas ─────────────────────────────────
        available = [s for s in trade_assets if s not in positions]
        if not available:
            continue

        # Primeiro filtra por V1 (barato — sem GPU)
        v1_hits = []
        for s in available:
            d   = data[s]
            cur = np.searchsorted(d["idx"], ts64)
            if cur < WINDOW_SIZE or cur >= len(d["idx"]) or d["idx"][cur] != ts64:
                continue
            b_sc, s_sc, _ema = v1_scores(d["feats"][cur], d["feats"][cur - 1],
                                         d["feat_cols"])
            rd = bool(d["regime_down"][cur])
            # Opera A FAVOR da tendencia diaria (SMA 24h):
            #   mercado em alta  -> compra os dips (LONG)
            #   mercado em queda -> shorta os ralis (SHORT)
            if b_sc >= v1_buy_thresh and not rd:
                v1_hits.append((s, cur, "LONG", b_sc))
            elif allow_short and s_sc >= v1_sell_thresh and rd:
                v1_hits.append((s, cur, "SHORT", s_sc))

        if not v1_hits:
            continue   # sem sinal V1 → nao precisa rodar NN

        # Coletar janelas dos candidatos para rodar NN em batch
        hit_syms = [h[0] for h in v1_hits]
        windows, metas = [], []
        for s, cur, direction, v1_sc in v1_hits:
            d = data[s]
            windows.append(d["feats"][cur - WINDOW_SIZE:cur])
            metas.append((s, cur, direction, v1_sc))

        X = torch.tensor(np.stack(windows), dtype=torch.float32).to(device)
        with torch.no_grad():
            logits, _ = model(X)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        p_up   = probs[:, 2]
        p_down = probs[:, 0]
        p_dir  = p_up + p_down + 1e-9
        dir_long  = p_up   / p_dir
        dir_short = p_down / p_dir

        # Seleciona melhor candidato (V1 score × dir_conf)
        best = None; best_strength = 0.0
        for i, (s, cur, direction, v1_sc) in enumerate(metas):
            dc = float(dir_long[i] if direction == "LONG" else dir_short[i])
            if dc < 0.52:
                continue
            strength = v1_sc * dc
            if strength > best_strength:
                best_strength = strength
                best = (s, cur, direction, v1_sc, dc)

        if best is None:
            continue

        sym_b, cur_b, dir_b, v1_b, dc_b = best
        lev = leverage_for(dc_b, use_lev, max_lev)
        if lev == 0.0:
            continue

        d      = data[sym_b]
        entry  = d["close"][cur_b]
        margin = cap * MARGIN_PCT

        if dir_b == "LONG":
            sl_price = entry * (1 - sl_pct)
            tp_price = entry * (1 + tp_pct)
        else:
            sl_price = entry * (1 + sl_pct)
            tp_price = entry * (1 - tp_pct)

        positions[sym_b] = {
            "ts":        ts,
            "ts64":      ts64,
            "entry":     entry,
            "entry_pos": cur_b,
            "scan_from": cur_b + 1,
            "deadline":  ts64 + fwd_ns_max,
            "direction": dir_b,
            "margin":    margin,
            "lev":       lev,
            "conf":      dc_b,
            "sl_price":  sl_price,
            "tp_price":  tp_price,
        }

        if realop:
            print(f"\n  [>] ENTRADA #{op_num+1} "
                  f"{ts.strftime('%d/%m/%Y %H:%M')} "
                  f"{'LONG' if dir_b=='LONG' else 'SHORT'} {sym_b}")
            print(f"      V1 score={v1_b} | NN dir_conf={dc_b*100:.1f}% "
                  f"-> {LEV_LABELS.get(lev, str(lev)+'x')}")
            print(f"      Entrada: ${entry:>10,.4f} | Margem: ${margin:,.2f} "
                  f"| Nocional: ${margin*lev:,.2f}")

        if cap < START_USD * 0.02:
            break

    # Fechar posicoes abertas no ultimo timestamp disponivel
    last_ts = timestamps[-1] if len(timestamps) > 0 else None
    for sym in list(positions.keys()):
        p = positions[sym]
        d = data[sym]
        last_pos = len(d["close"]) - 1
        entry    = p["entry"]
        exit_    = d["close"][last_pos]
        ep       = p["entry_pos"]
        low_min  = d["low"][ep:last_pos + 1].min()
        high_max = d["high"][ep:last_pos + 1].max()
        pnl, outcome = _calc_pnl(p["direction"], entry, exit_,
                                  low_min, high_max, p["margin"], p["lev"])
        cap    += pnl
        op_num += 1
        hold_min = int((d["idx"][last_pos] - p["ts64"]) / 60_000_000_000)
        trades.append({
            "op": op_num, "ts": p["ts"], "exit_ts": last_ts,
            "sym": sym, "direction": p["direction"],
            "conf": p["conf"], "lev": p["lev"],
            "entry": entry, "exit": exit_,
            "pnl": pnl, "cap": cap, "outcome": outcome,
            "cause": "FIM_PERIODO", "hold_min": hold_min,
        })

    # ── Resultados ─────────────────────────────────────────────────────
    if not trades:
        print("Nenhuma operacao — V1 nunca gerou sinal no periodo.")
        print(f"Capital preservado: ${cap:,.2f}")
        return {}

    df_t = pd.DataFrame(trades)
    n    = len(df_t)
    wins = (df_t["outcome"] == "WIN").sum()
    liqs = (df_t["outcome"] == "LIQUIDADO").sum()

    avg_hold = df_t["hold_min"].mean()

    n_long  = (df_t["direction"] == "LONG").sum()
    n_short = (df_t["direction"] == "SHORT").sum()
    pnl_long  = df_t[df_t["direction"] == "LONG"]["pnl"].sum()
    pnl_short = df_t[df_t["direction"] == "SHORT"]["pnl"].sum()

    real_start = pd.Timestamp(p_start)
    real_end   = df_t["exit_ts"].dropna().max()
    n_days     = max((real_end - real_start).days, 1) if pd.notna(real_end) else 1

    total_ret   = (cap / START_USD - 1) * 100
    daily_ret   = total_ret / n_days
    monthly_ret = daily_ret * 30

    btc_start    = float(btc_df["close"].loc[p_start:].iloc[0])
    btc_end      = float(btc_df["close"].loc[:p_end].iloc[-1])
    btc_hold_usd = START_USD * (btc_end / btc_start)
    btc_var_pct  = (btc_end / btc_start - 1) * 100
    btc_acc_pct  = (cap / btc_end) / (START_USD / btc_start) * 100 - 100

    sep = "=" * 70

    print(f"\n{'─'*70}")
    print(f"{sep}")
    print(f"  RESUMO FINAL ({p_start} -> {real_end.date() if pd.notna(real_end) else '?'})")
    print(f"{sep}")
    print(f"  Capital inicial       : ${START_USD:>12,.2f}")
    print(f"  Capital final         : ${cap:>12,.2f}   ({total_ret:+.1f}%)")
    print(f"  Lucro/prejuizo USD    : ${cap-START_USD:>+12,.2f}")
    print(f"  Retorno medio/dia     : {daily_ret:>+9.2f}%/dia")
    print(f"  Equiv. mensal (30d)   : {monthly_ret:>+9.2f}%/mes")
    print()
    print(f"  Total de operacoes    : {n}")
    print(f"    LONG  (aposta alta) : {n_long:>4}  | PnL ${pnl_long:>+10,.2f}")
    print(f"    SHORT (aposta baixa): {n_short:>4}  | PnL ${pnl_short:>+10,.2f}")
    print(f"  Vitorias              : {wins} ({wins/n*100:.1f}%)")
    print(f"  Derrotas              : {n-wins-liqs} ({(n-wins-liqs)/n*100:.1f}%)")
    print(f"  Liquidacoes           : {liqs} ({liqs/n*100:.1f}%)")
    print(f"  Tempo medio de hold   : {avg_hold:.0f}min")
    print()
    print(f"  Saidas por causa:")
    for cause_name in ["TP", "SL", "SINAL", "TEMPO", "FIM_PERIODO"]:
        cd = df_t[df_t["cause"] == cause_name]
        if len(cd) == 0:
            continue
        cw = int((cd["outcome"] == "WIN").sum())
        print(f"    {cause_name:<12}: {len(cd):>3} ops ({len(cd)/n*100:>3.0f}%) | "
              f"win {cw}/{len(cd)} | PnL ${cd['pnl'].sum():>+10,.2f}")
    print()

    # Distribuicao de alavancagem
    lev_counts = df_t.groupby("lev").agg(
        ops=("pnl", "count"),
        wins_pct=("outcome", lambda x: (x == "WIN").mean() * 100),
        pnl_total=("pnl", "sum"),
    ).sort_index(ascending=False)
    print(f"  Distribuicao de alavancagem:")
    for lev_v, row in lev_counts.iterrows():
        label = LEV_LABELS.get(float(lev_v), f"{lev_v}x")
        print(f"    {label} : {int(row.ops):>3} ops | "
              f"win {row.wins_pct:>4.0f}% | "
              f"PnL ${row.pnl_total:>+10,.2f}")
    print()

    print(f"  Por ativo:")
    for s in sorted(df_t["sym"].unique()):
        sd     = df_t[df_t["sym"] == s]
        w      = (sd["outcome"] == "WIN").sum()
        longs  = (sd["direction"] == "LONG").sum()
        shorts = (sd["direction"] == "SHORT").sum()
        print(f"    {s:<10}: {len(sd):>3} ops "
              f"(L:{longs} S:{shorts}) | "
              f"win {w}/{len(sd)} | "
              f"PnL ${sd['pnl'].sum():>+8,.2f}")
    print()

    print(f"  {'─'*55}")
    print(f"  Comparacao de desempenho ({n_days} dias):")
    print(f"    Esta estrategia   : ${cap:>12,.2f}  ({total_ret:+.1f}%)")
    print(f"    Hold de BTC       : ${btc_hold_usd:>12,.2f}  ({btc_var_pct:+.1f}%)")
    print(f"    USDT parado       : ${START_USD:>12,.2f}  (  0.0%)")
    print()
    print(f"  Em termos de BTC (acumulacao):")
    cap_btc_start = START_USD / btc_start
    cap_btc_end   = cap / btc_end
    print(f"    Comecou com:  {cap_btc_start:.5f} BTC")
    print(f"    Terminou com: {cap_btc_end:.5f} BTC  ({btc_acc_pct:+.1f}%)")
    print(f"{sep}")

    if total_ret > 0:
        print(f"\n  LUCRO: +{total_ret:.1f}% em {n_days} dias "
              f"= ~{monthly_ret:.1f}%/mes")
    else:
        print(f"\n  PREJUIZO: {total_ret:.1f}% em {n_days} dias")

    print(f"{'='*70}")

    return {
        "cap_final": cap, "n": n, "wins": int(wins),
        "liquidations": int(liqs), "total_ret": total_ret,
        "n_long": int(n_long), "n_short": int(n_short),
        "pnl_long": float(pnl_long), "pnl_short": float(pnl_short),
        "avg_hold_min": float(avg_hold),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Trader.AI V5.8 — Hibrido V1+V5 com Gestao de Posicao")
    ap.add_argument("--val",       action="store_true",
                    help="Usar split de validacao em vez de teste")
    ap.add_argument("--no-lev",    action="store_true",
                    help="Sem alavancagem (spot 1x)")
    ap.add_argument("--no-short",  action="store_true",
                    help="Desativa SHORT (apenas LONG)")
    ap.add_argument("--realop",    action="store_true",
                    help="Diario detalhado de cada operacao")
    ap.add_argument("--from",      dest="from_date", default=None,
                    metavar="YYYY-MM-DD")
    ap.add_argument("--to",        dest="to_date",   default=None,
                    metavar="YYYY-MM-DD")
    ap.add_argument("--buy",       dest="buy_thresh",  type=int, default=60,
                    help="V1 buy_score minimo (padrao 60)")
    ap.add_argument("--sell",      dest="sell_thresh", type=int, default=60,
                    help="V1 sell_score minimo (padrao 60)")
    ap.add_argument("--eval-step", dest="eval_step",   type=int, default=15,
                    help="Frequencia de avaliacao em minutos (padrao 15)")
    ap.add_argument("--max-hold",  dest="max_hold",    type=int, default=360,
                    help="Tempo maximo de posicao em minutos (padrao 360 = 6h)")
    ap.add_argument("--model",     dest="model_path",  default=MODEL_PATH,
                    help="Arquivo do modelo treinado (padrao v5_model.pth)")
    ap.add_argument("--sl",        dest="sl_pct",  type=float, default=0.005,
                    help="Stop loss em fracao (padrao 0.005 = 0.5%%)")
    ap.add_argument("--tp",        dest="tp_pct",  type=float, default=0.010,
                    help="Take profit em fracao (padrao 0.010 = 1.0%%)")
    ap.add_argument("--max-lev",   dest="max_lev", type=float, default=5.0,
                    help="Alavancagem maxima (padrao 5x — faixa calibrada)")
    ap.add_argument("--skip",      dest="skip_syms", default=None,
                    help="Ativos a nao operar, separados por virgula "
                         "(ex: BTCUSDT). Continuam como contexto.")
    ap.add_argument("--featset",   choices=["v5", "v6"], default="v5",
                    help="Conjunto de features: v5 (18) ou v6 (26). "
                         "Deve casar com o modelo usado.")
    a = ap.parse_args()

    split = "val" if a.val else "test"
    run_backtest(
        split          = split,
        use_lev        = not a.no_lev,
        allow_short    = not a.no_short,
        from_date      = a.from_date,
        to_date        = a.to_date,
        realop         = a.realop,
        v1_buy_thresh  = a.buy_thresh,
        v1_sell_thresh = a.sell_thresh,
        eval_step      = a.eval_step,
        max_hold_min   = a.max_hold,
        model_path     = a.model_path,
        sl_pct         = a.sl_pct,
        tp_pct         = a.tp_pct,
        max_lev        = a.max_lev,
        skip_syms      = a.skip_syms.split(",") if a.skip_syms else None,
        featset        = a.featset,
    )
