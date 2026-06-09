"""
v5_backtest.py — Trader.AI V5.1 (Defensivo + Agressivo com 20x)
Backtest honesto: base defensiva em USDT, long alavancado so em ALTA forte,
COM MODELAGEM DE LIQUIDACAO (com 20x, ~5% contra ja liquida).

Logica:
  - Estado padrao: USDT (defensivo — fora do mercado nas quedas)
  - A cada passo, modelo preve direcao de cada ativo (QUEDA/NEUTRO/ALTA)
  - Se prob_ALTA do melhor ativo >= 0.75 -> long 20x
        0.70-0.75 -> 10x | 0.65-0.70 -> 5x | 0.60-0.65 -> 2x | senao USDT
  - Liquidacao: se durante a janela o preco cair >= (1/L - margem), perde a margem

Mede em USD e em BTC-equivalente. Compara com hold de BTC.

Uso (com python do venv):
  venv/Scripts/python.exe v5_backtest.py          -> teste 2025+
  venv/Scripts/python.exe v5_backtest.py --val    -> validacao 2024
  venv/Scripts/python.exe v5_backtest.py --no-lev -> sem alavancagem
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch

from v5_model import load_model, get_device
from v5_data_prep import (ASSETS, BTC, WINDOW_SIZE, HORIZON_H,
                          _load_parquet, _add_features)

MODEL_PATH   = "v5_model.pth"
FEE          = 0.0004    # 0.04% por lado (taxa de futuros Binance)
START_USD    = 10_000.0
MARGIN_PCT   = 0.20      # fracao do capital usada como margem por operacao
MAINT_MARGIN = 0.005     # margem de manutencao (liquidacao)

# prob_ALTA -> alavancagem
# Com alavancagem: escala de 1x (>=50%) ate 20x (>=75%)
# Sem alavancagem (--no-lev): spot puro a partir de 50%
def leverage_for(prob_up: float, use_lev: bool) -> float:
    if not use_lev:
        return 1.0 if prob_up >= 0.50 else 0.0
    if prob_up >= 0.75: return 20.0
    if prob_up >= 0.70: return 10.0
    if prob_up >= 0.65: return 5.0
    if prob_up >= 0.60: return 2.0
    if prob_up >= 0.50: return 1.0   # spot sem alavancagem para sinais moderados
    return 0.0


def precompute(sym, btc_df):
    adf = _load_parquet(sym)
    common = adf.index.intersection(btc_df.index)
    adf, btc_al = adf.loc[common], btc_df.loc[common]
    feat_df = _add_features(adf, btc_al).dropna()
    feats = feat_df.values.astype(np.float32)
    idx   = feat_df.index.values.astype("int64")
    close = adf["close"].reindex(feat_df.index).values.astype(np.float64)
    low   = adf["low"].reindex(feat_df.index).values.astype(np.float64)
    return {"feats": feats, "idx": idx, "close": close, "low": low}


def run_backtest(split="test", use_lev=True):
    device = get_device()
    if split == "val":
        p_start, p_end = "2024-01-01", "2024-12-31"
    else:
        p_start, p_end = "2025-01-01", "2099-12-31"

    print(f"\n{'='*68}")
    print(f"BACKTEST V5.3 — Defensivo + Agressivo {'(ate 20x)' if use_lev else '(spot 1x)'}")
    print(f"Periodo: {p_start[:7]} a {p_end[:7]} | Base: USDT (defensivo)")
    print(f"Liquidacao modelada | Fee: {FEE*100:.2f}%/lado | Margem/op: {MARGIN_PCT*100:.0f}%")
    print(f"{'='*68}")

    btc_df  = _load_parquet(BTC)
    sample  = precompute(ASSETS[0], btc_df)
    model   = load_model(MODEL_PATH, sample["feats"].shape[1], device)
    print(f"Modelo: {sample['feats'].shape[1]} features\n")

    print("Pre-computando features...")
    data = {s: precompute(s, btc_df) for s in ASSETS}
    for s in ASSETS:
        print(f"  {s}: {len(data[s]['feats'])} candles")

    period_idx = btc_df.index[(btc_df.index >= p_start) & (btc_df.index <= p_end)]
    if len(period_idx) < HORIZON_H * 2:
        print("Dados insuficientes."); return {}

    timestamps = period_idx[::HORIZON_H]   # passos nao-sobrepostos de HORIZON_H min
    # CORRECAO: timestamps pandas sao em nanossegundos; 1 minuto = 60 * 1e9 ns
    fwd_ms     = HORIZON_H * 60 * 1_000_000_000

    print(f"\nSimulando {len(timestamps)} passos de {HORIZON_H//60}h "
          f"({period_idx[0].date()} a {period_idx[-1].date()})...\n")

    cap = START_USD
    trades, liquidations, days_in_market = [], 0, 0

    for ts in timestamps:
        ts64 = np.int64(ts.value)
        windows, metas = [], []
        for s in ASSETS:
            d = data[s]
            pos = np.searchsorted(d["idx"], ts64)
            if pos < WINDOW_SIZE or pos >= len(d["idx"]) or d["idx"][pos] != ts64:
                continue
            windows.append(d["feats"][pos - WINDOW_SIZE:pos])
            metas.append((s, pos))
        if not windows:
            continue

        X = torch.tensor(np.stack(windows), dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(X)[0], dim=1).cpu().numpy()  # [n,3]

        p_up = probs[:, 2]
        best = int(np.argmax(p_up))
        best_prob = float(p_up[best])
        sym, pos = metas[best]

        lev = leverage_for(best_prob, use_lev)
        if lev == 0.0:
            continue  # defensivo: fica em USDT

        d = data[sym]
        exit_pos = np.searchsorted(d["idx"], ts64 + fwd_ms)
        if exit_pos >= len(d["close"]):
            continue

        entry = d["close"][pos]
        exit_ = d["close"][exit_pos]
        # pior preco durante a janela (para checar liquidacao)
        low_min = d["low"][pos:exit_pos + 1].min() if exit_pos > pos else d["low"][pos]

        margin = cap * MARGIN_PCT
        liq_level = entry * (1 - 1.0 / lev + MAINT_MARGIN)

        if low_min <= liq_level:
            # LIQUIDADO: perde a margem inteira
            pnl = -margin
            liquidations += 1
            outcome = "LIQ"
        else:
            ret = exit_ / entry - 1
            notional = margin * lev
            pnl = notional * ret - notional * FEE * 2
            pnl = max(pnl, -margin)  # nunca perde mais que a margem
            outcome = "WIN" if pnl > 0 else "LOSS"

        cap += pnl
        days_in_market += HORIZON_H / (60 * 24)
        trades.append({"ts": ts, "sym": sym, "prob": best_prob, "lev": lev,
                       "pnl": pnl, "cap": cap, "outcome": outcome})
        if cap < START_USD * 0.02:
            print(f"  [FALENCIA] Capital praticamente zerado em {ts.date()}.")
            break

    # ── Resultados ────────────────────────────────────────────────────────────
    if not trades:
        print("Nenhuma operacao (modelo nunca atingiu confianca minima de ALTA).")
        print("Isso significa: 100% defensivo, capital intacto em USDT.\n")
        return {}

    df = pd.DataFrame(trades)
    wins = (df["outcome"] == "WIN").sum()
    n = len(df)
    btc_start = float(btc_df["close"].loc[p_start:].iloc[0])
    btc_end   = float(btc_df["close"].loc[:p_end].iloc[-1])
    btc_hold_usd = START_USD * (btc_end / btc_start)

    cap_btc_start = START_USD / btc_start    # quanto BTC daria pra comprar no inicio
    cap_btc_end   = cap / btc_end            # quanto BTC o capital final compra

    print(f"{'-'*68}")
    print(f"  Capital inicial:      ${START_USD:>12,.2f}")
    print(f"  Capital final:        ${cap:>12,.2f}   ({(cap/START_USD-1)*100:+.1f}%)")
    print(f"  ----------------------------------------------------------")
    print(f"  Operacoes:            {n}")
    print(f"  Vitorias:             {wins} ({wins/n*100:.1f}%)")
    print(f"  Liquidacoes (20x):    {liquidations} ({liquidations/n*100:.1f}%)")
    print(f"  Tempo exposto:        {days_in_market:.0f} dias (resto em USDT, seguro)")
    print(f"  ----------------------------------------------------------")
    print(f"  [Comparacao em USD]")
    print(f"    Esta estrategia:    ${cap:>12,.2f}")
    print(f"    Hold de BTC:        ${btc_hold_usd:>12,.2f}")
    print(f"    100% USDT (parado): ${START_USD:>12,.2f}")
    print(f"  ----------------------------------------------------------")
    print(f"  [Comparacao em BTC]  (seu objetivo real)")
    print(f"    Comecou com:        {cap_btc_start:.4f} BTC de poder de compra")
    print(f"    Terminou com:       {cap_btc_end:.4f} BTC de poder de compra")
    print(f"{'-'*68}")

    if cap_btc_end > cap_btc_start:
        print(f"\n  RESULTADO: ACUMULOU BTC "
              f"({(cap_btc_end/cap_btc_start-1)*100:+.1f}% em poder de compra de BTC)")
    else:
        print(f"\n  RESULTADO: PERDEU poder de compra de BTC "
              f"({(cap_btc_end/cap_btc_start-1)*100:+.1f}%)")

    print(f"\n  Por ativo:")
    for s in df["sym"].unique():
        sd = df[df["sym"] == s]
        print(f"    {s}: {len(sd)} ops | PnL ${sd['pnl'].sum():+,.0f} | "
              f"liq {(sd['outcome']=='LIQ').sum()}")
    print(f"\n{'='*68}")
    return {"cap_final": cap, "n": n, "liquidations": liquidations}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--no-lev", action="store_true")
    a = ap.parse_args()
    run_backtest("val" if a.val else "test", use_lev=not a.no_lev)
