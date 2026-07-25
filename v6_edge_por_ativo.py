# -*- coding: utf-8 -*-
"""
v6_edge_por_ativo.py — Trader.AI V6: o modelo tem EDGE em cada ativo?
======================================================================

Pergunta que os backtests nao respondem diretamente: quando o modelo aponta
uma direcao num ativo, ele ACERTA mais do que erra? (independente de stops,
alavancagem ou taxas)

Mede, direto dos precos, para cada ativo:
  - amostras em que o modelo favorece uma direcao com dir_conf >= limiar
  - FAVOR : o preco andou >= thresh na direcao apontada em 120 min
  - CONTRA: andou >= thresh na direcao oposta
  - EDGE  : FAVOR / (FAVOR + CONTRA)   -> 0,50 = moeda ao ar; >0,55 = util

Se ADA/DOT/LINK tiverem EDGE ~0,50, o problema NAO e o stop (nem o ATR):
e ausencia de poder preditivo — e a expansao do universo deve ser
arquivada ate um retreino que inclua esses ativos.

Uso:
  python v6_edge_por_ativo.py                       # teste 2026 (jan-jul)
  python v6_edge_por_ativo.py --from 2026-06-01     # so o holdout virgem
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from v5_model import load_model
from v5_data_prep import BTC, WINDOW_SIZE, HORIZON_H, _load_parquet, _add_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="v5_model_b.pth")
    ap.add_argument("--featset", choices=["v5", "v6"], default="v5")
    ap.add_argument("--from", dest="dt_from", default="2026-01-01")
    ap.add_argument("--to",   dest="dt_to",   default="2099-12-31")
    ap.add_argument("--conf", type=float, default=0.52,
                    help="limiar de dir_conf (padrao 0.52 = o do backtest)")
    ap.add_argument("--thresh", type=float, default=0.004,
                    help="movimento minimo para contar acerto (padrao 0.4%%)")
    ap.add_argument("--step", type=int, default=15, help="passo em minutos")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    device = "cuda" if (a.gpu and torch.cuda.is_available()) else "cpu"
    if a.featset == "v6":
        from v6_data_prep import add_features_v6 as feat_fn
    else:
        feat_fn = _add_features

    import glob
    ativos = sorted(os.path.basename(p).replace("_1m.parquet", "")
                    for p in glob.glob("data/*_1m.parquet"))

    btc = _load_parquet(BTC)
    print(f"\n{'='*74}")
    print(f"  EDGE DIRECIONAL POR ATIVO — modelo {a.model} ({a.featset})")
    print(f"  Periodo {a.dt_from} -> {a.dt_to} | dir_conf >= {a.conf} | "
          f"mov >= {a.thresh*100:.1f}%")
    print(f"{'='*74}")
    print(f"  {'Ativo':<10} {'sinais':>7} {'FAVOR':>7} {'CONTRA':>7} "
          f"{'EDGE':>7}  veredito")
    print("  " + "-" * 62)

    model = None
    linhas = []
    for sym in ativos:
        adf = _load_parquet(sym)
        com = adf.index.intersection(btc.index)
        a2, b2 = adf.loc[com], btc.loc[com]
        feats_df = feat_fn(a2, b2).dropna()
        m = (feats_df.index >= a.dt_from) & (feats_df.index <= a.dt_to)
        idx_validos = np.where(m)[0]
        idx_validos = idx_validos[idx_validos >= WINDOW_SIZE][::a.step]
        if len(idx_validos) < 20:
            continue

        X = feats_df.values.astype(np.float32)
        if model is None:
            model = load_model(a.model, X.shape[1], device)

        close = a2["close"].reindex(feats_df.index).values
        favor = contra = sinais = 0

        for i0 in range(0, len(idx_validos), 256):
            sel = idx_validos[i0:i0 + 256]
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
            ok = dirc >= a.conf
            if not ok.any():
                continue
            fwd = close[sel + HORIZON_H] / close[sel] - 1
            sinais += int(ok.sum())
            ganho = np.where(is_long, fwd, -fwd)          # retorno na direcao apontada
            favor  += int(((ganho >=  a.thresh) & ok).sum())
            contra += int(((ganho <= -a.thresh) & ok).sum())

        edge = favor / (favor + contra) if (favor + contra) else float("nan")
        vered = ("SEM EDGE" if not np.isfinite(edge) or edge < 0.52 else
                 "fraco" if edge < 0.55 else "UTIL")
        print(f"  {sym:<10} {sinais:>7} {favor:>7} {contra:>7} {edge:>7.3f}  {vered}")
        linhas.append((sym, edge, favor, contra))
        del adf, a2, b2, feats_df, X

    print("  " + "-" * 62)
    tf = sum(l[2] for l in linhas); tc = sum(l[3] for l in linhas)
    if tf + tc:
        print(f"  {'GERAL':<10} {'':>7} {tf:>7} {tc:>7} "
              f"{tf/(tf+tc):>7.3f}")
    print(f"{'='*74}")
    print("  EDGE = FAVOR/(FAVOR+CONTRA). 0,50 = aleatorio. >0,55 = util.")


if __name__ == "__main__":
    main()
