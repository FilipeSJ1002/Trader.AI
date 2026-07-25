# -*- coding: utf-8 -*-
"""
v6_calibracao.py — Trader.AI V6: análise de calibração da confiança direcional
===============================================================================

Mede, num split (validação por padrão), a distribuição da dir_conf e a
precisão REAL por faixa — contra os rótulos verdadeiros do dataset.

É o instrumento que responde às duas perguntas da V6:
  1. O modelo novo gera MAIS amostras nas faixas altas (>=0.62)?
  2. As faixas altas são CALIBRADAS (precisão condiz com a confiança)?
     -> se sim, dá para destravar alavancagem acima de 5x COM EVIDÊNCIA.

Definições por amostra:
  direcao favorecida = LONG se p_alta > p_queda, senão SHORT
  dir_conf           = p_favor / (p_alta + p_queda)
  FAVOR   = rótulo verdadeiro na direção favorecida  (ganho provável)
  NEUTRO  = rótulo NEUTRO                            (TP/SL decide, ~neutro)
  CONTRA  = rótulo na direção oposta                 (perda provável)

Uso:
  python v6_calibracao.py --model v6_model.pth  --data data_v6              # V6
  python v6_calibracao.py --model v5_model_b.pth --data data_v5a --y-dir data_v5b  # baseline V5.9
  Flags: --split val|test | --max-per-asset N | --gpu
"""
import os
import sys
import argparse
import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from v5_model import load_model
from v5_data_prep import ASSETS

BANDAS = [(0.50, 0.52), (0.52, 0.57), (0.57, 0.62),
          (0.62, 0.67), (0.67, 0.72), (0.72, 1.01)]
ROTULO_BANDA = {(0.50, 0.52): "sem operacao",
                (0.52, 0.57): "1x",
                (0.57, 0.62): "2x",
                (0.62, 0.67): "5x  <- alvo",
                (0.67, 0.72): "(10x desativada)",
                (0.72, 1.01): "(20x desativada)"}


def carrega(split, sym, data_dir, y_dir=None):
    p = os.path.join(data_dir, f"{split}_{sym}.npz")
    if not os.path.exists(p):
        return None, None
    with np.load(p) as d:
        X = d["X"]
        y = d["y"]
    if y_dir:
        py = os.path.join(y_dir, f"{split}_{sym}.npz")
        with np.load(py) as d:
            y = d["y"]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data",  required=True, help="Diretorio dos npz (X)")
    ap.add_argument("--y-dir", default=None, help="Diretorio alternativo dos rotulos")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--max-per-asset", type=int, default=8000,
                    help="Subamostra por ativo (0 = tudo; padrao 8000)")
    ap.add_argument("--gpu", action="store_true",
                    help="Usa GPU (padrao CPU para nao disputar com treinos)")
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()

    device = "cuda" if (a.gpu and torch.cuda.is_available()) else "cpu"

    # n_features do proprio dataset
    n_feat = None
    for sym in ASSETS:
        X, _ = carrega(a.split, sym, a.data, a.y_dir)
        if X is not None:
            n_feat = X.shape[2]
            break
    if n_feat is None:
        print(f"Nenhum npz de split '{a.split}' em {a.data}/")
        return

    model = load_model(a.model, n_feat, device)
    print(f"\n{'='*66}")
    print(f"  CALIBRACAO DIR_CONF — modelo {a.model} ({n_feat} features)")
    print(f"  Split: {a.split} | device: {device} | max/ativo: {a.max_per_asset or 'tudo'}")
    print(f"{'='*66}")

    confs, favors, neutros = [], [], []
    rng = np.random.default_rng(42)

    for sym in ASSETS:
        X, y = carrega(a.split, sym, a.data, a.y_dir)
        if X is None:
            continue
        n = len(X)
        idx = np.arange(n)
        if a.max_per_asset and n > a.max_per_asset:
            idx = rng.choice(n, a.max_per_asset, replace=False)
        print(f"  [{sym}] {len(idx)}/{n} amostras...", flush=True)

        for i0 in range(0, len(idx), a.batch):
            sel = idx[i0:i0 + a.batch]
            xb = torch.tensor(X[sel], dtype=torch.float32).to(device)
            with torch.no_grad():
                logits, _ = model(xb)
                probs = torch.softmax(logits, 1).cpu().numpy()
            p_dn, p_up = probs[:, 0], probs[:, 2]
            p_dir = p_up + p_dn + 1e-9
            long_mask = p_up >= p_dn
            conf = np.where(long_mask, p_up, p_dn) / p_dir
            yb = y[sel]
            favor  = np.where(long_mask, yb == 2, yb == 0)
            neutro = (yb == 1)
            confs.append(conf); favors.append(favor); neutros.append(neutro)
        del X, y

    conf   = np.concatenate(confs)
    favor  = np.concatenate(favors)
    neutro = np.concatenate(neutros)
    contra = ~(favor | neutro)
    total  = len(conf)

    print(f"\n  Total avaliado: {total} amostras")
    print(f"  dir_conf: media {conf.mean():.3f} | p95 {np.percentile(conf,95):.3f} "
          f"| max {conf.max():.3f}")
    print(f"\n  {'Faixa':>12} {'Alav.':>18} {'Amostras':>9} {'% total':>8} "
          f"{'FAVOR':>7} {'NEUTRO':>7} {'CONTRA':>7} {'F/C':>6}")
    print("  " + "-" * 84)
    for lo, hi in BANDAS:
        m = (conf >= lo) & (conf < hi)
        n = int(m.sum())
        pct = n / total * 100
        if n == 0:
            print(f"  {lo:.2f}-{hi:.2f} {ROTULO_BANDA[(lo,hi)]:>18} {n:>9} {pct:>7.2f}%"
                  f" {'—':>7} {'—':>7} {'—':>7} {'—':>6}")
            continue
        f = favor[m].mean() * 100
        nt = neutro[m].mean() * 100
        c = contra[m].mean() * 100
        fc = (favor[m].sum() / max(contra[m].sum(), 1))
        print(f"  {lo:.2f}-{hi:.2f} {ROTULO_BANDA[(lo,hi)]:>18} {n:>9} {pct:>7.2f}%"
              f" {f:>6.1f}% {nt:>6.1f}% {c:>6.1f}% {fc:>6.2f}")

    n62 = int(((conf >= 0.62)).sum())
    print(f"\n  >> METRICA-ALVO DA V6: amostras com dir_conf >= 0.62: "
          f"{n62} ({n62/total*100:.2f}% do split)")
    print(f"  >> F/C = razao FAVOR/CONTRA (>1.0 = modelo acerta mais que erra na direcao)")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
