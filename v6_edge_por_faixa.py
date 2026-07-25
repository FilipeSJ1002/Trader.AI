# -*- coding: utf-8 -*-
"""
v6_edge_por_faixa.py — Trader.AI V6: o edge cresce com a confianca do modelo?
=============================================================================

A pergunta que decide o rumo da V6.

Contexto (medido em 25/07/2026): o edge direcional GERAL do modelo V5.9-B e
0,528 — muito fraco e praticamente igual em todos os 11 ativos. Isso explica
por que ampliar o universo e mexer nos stops nao adiantou: o gargalo e o
poder preditivo, nao a execucao.

Mas o backtest mostra que a faixa dir_conf>=0.62 (alavancagem 5x) e a mais
lucrativa. Este script verifica se isso vem de EDGE REAL crescente ou se e
so ruido/alavancagem.

Se o edge crescer com a confianca (ex.: 0,52 -> 0,58 -> 0,64), a estrategia
correta e SER MAIS SELETIVO: operar so alta conviccao, com mais alavancagem.
Se ficar chato em ~0,53 em todas as faixas, o modelo nao sabe quando sabe —
e a unica saida e um modelo melhor.

Uso:
  python v6_edge_por_faixa.py                                  # teste 2026
  python v6_edge_por_faixa.py --from 2026-06-01                # holdout
  python v6_edge_por_faixa.py --model v6_model.pth --featset v6
"""
import os
import sys
import glob
import argparse
import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from v5_model import load_model
from v5_data_prep import BTC, WINDOW_SIZE, HORIZON_H, _load_parquet, _add_features

FAIXAS = [(0.50, 0.52, "abaixo do limiar"),
          (0.52, 0.57, "1x"),
          (0.57, 0.62, "2x"),
          (0.62, 0.67, "5x  <- a faixa lucrativa"),
          (0.67, 0.72, "(10x desativada)"),
          (0.72, 1.01, "(20x desativada)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v5_model_b.pth")
    ap.add_argument("--featset", choices=["v5", "v6"], default="v5")
    ap.add_argument("--from", dest="dt_from", default="2026-01-01")
    ap.add_argument("--to",   dest="dt_to",   default="2099-12-31")
    ap.add_argument("--thresh", type=float, default=0.004)
    ap.add_argument("--step", type=int, default=15)
    ap.add_argument("--assets", default="6",
                    help="'6' (os do treino) ou 'ALL'")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    device = "cuda" if (a.gpu and torch.cuda.is_available()) else "cpu"
    if a.featset == "v6":
        from v6_data_prep import add_features_v6 as feat_fn
    else:
        feat_fn = _add_features

    if a.assets.upper() == "ALL":
        ativos = sorted(os.path.basename(p).replace("_1m.parquet", "")
                        for p in glob.glob("data/*_1m.parquet"))
    else:
        from v5_data_prep import ASSETS
        ativos = list(ASSETS)

    btc = _load_parquet(BTC)
    confs, ganhos = [], []
    model = None

    print(f"\nColetando predicoes ({len(ativos)} ativos, {a.dt_from} -> {a.dt_to})...")
    for sym in ativos:
        adf = _load_parquet(sym)
        com = adf.index.intersection(btc.index)
        a2, b2 = adf.loc[com], btc.loc[com]
        fdf = feat_fn(a2, b2).dropna()
        m = (fdf.index >= a.dt_from) & (fdf.index <= a.dt_to)
        idxs = np.where(m)[0]
        idxs = idxs[idxs >= WINDOW_SIZE][::a.step]
        if len(idxs) < 20:
            continue
        X = fdf.values.astype(np.float32)
        if model is None:
            model = load_model(a.model, X.shape[1], device)
        close = a2["close"].reindex(fdf.index).values

        for i0 in range(0, len(idxs), 256):
            sel = idxs[i0:i0 + 256]
            sel = sel[sel + HORIZON_H < len(close)]
            if len(sel) == 0:
                continue
            batch = np.stack([X[i - WINDOW_SIZE:i] for i in sel])
            with torch.no_grad():
                logits, _ = model(torch.tensor(batch).to(device))
                p = torch.softmax(logits, 1).cpu().numpy()
            p_dn, p_up = p[:, 0], p[:, 2]
            dirc = np.maximum(p_up, p_dn) / (p_up + p_dn + 1e-9)
            is_long = p_up >= p_dn
            fwd = close[sel + HORIZON_H] / close[sel] - 1
            confs.append(dirc)
            ganhos.append(np.where(is_long, fwd, -fwd))
        del adf, a2, b2, fdf, X
        print(f"  {sym} ok", flush=True)

    conf  = np.concatenate(confs)
    ganho = np.concatenate(ganhos)

    print(f"\n{'='*78}")
    print(f"  EDGE POR FAIXA DE CONFIANCA — {a.model} ({a.featset}) | "
          f"{len(ativos)} ativos")
    print(f"  Periodo {a.dt_from} -> {a.dt_to} | movimento minimo {a.thresh*100:.1f}%")
    print(f"{'='*78}")
    print(f"  {'Faixa':>11} {'alavanc.':>22} {'amostras':>9} {'%':>6} "
          f"{'FAVOR':>7} {'CONTRA':>7} {'EDGE':>7}")
    print("  " + "-" * 76)
    total = len(conf)
    for lo, hi, rot in FAIXAS:
        m = (conf >= lo) & (conf < hi)
        n = int(m.sum())
        if n == 0:
            print(f"  {lo:.2f}-{hi:.2f} {rot:>22} {n:>9} {'—':>6}")
            continue
        g = ganho[m]
        fav = int((g >= a.thresh).sum())
        con = int((g <= -a.thresh).sum())
        edge = fav / (fav + con) if (fav + con) else float("nan")
        print(f"  {lo:.2f}-{hi:.2f} {rot:>22} {n:>9} {n/total*100:>5.1f}% "
              f"{fav:>7} {con:>7} {edge:>7.3f}")

    # Visao acumulada: e melhor ser mais seletivo?
    print("\n  Acumulado (operar SO acima do limiar):")
    print(f"  {'limiar':>8} {'amostras':>9} {'% do total':>11} {'EDGE':>7} "
          f"{'retorno medio':>14}")
    print("  " + "-" * 56)
    for lim in [0.52, 0.57, 0.62, 0.67, 0.72]:
        m = conf >= lim
        n = int(m.sum())
        if n < 30:
            print(f"  {lim:>8.2f} {n:>9} {'(poucas)':>11}")
            continue
        g = ganho[m]
        fav = int((g >= a.thresh).sum()); con = int((g <= -a.thresh).sum())
        edge = fav / (fav + con) if (fav + con) else float("nan")
        print(f"  {lim:>8.2f} {n:>9} {n/total*100:>10.1f}% {edge:>7.3f} "
              f"{g.mean()*100:>13.3f}%")

    print(f"{'='*78}")
    print("  LEITURA: se o EDGE cresce com o limiar -> ser mais seletivo aumenta o lucro.")
    print("           se fica ~0,53 em todas as faixas -> o modelo nao sabe quando sabe.")


if __name__ == "__main__":
    main()
