# -*- coding: utf-8 -*-
"""
v6_data_prep.py — Trader.AI V6: features enriquecidas (18 -> 26)
=================================================================

Objetivo da V6 (Etapa 7): aumentar o volume de sinais de ALTA CONVICCAO
(dir_conf >= 0.62) — o gargalo de lucro comprovado da V5.9.

O que muda em relacao as 18 features da V5 (v5_data_prep._add_features):

  +2  dist_bbu / dist_bbl  Bandas de Bollinger — CORRIGE o bug do pandas-ta
                           0.4.71b0 (colunas viraram BBU_20_2.0_2.0; o "if"
                           antigo nunca disparava). Tambem reativa os +35 pts
                           de Bollinger no v1_scores() do gatilho de entrada.
  +1  bb_width             Largura das bandas / preco (squeeze de volatilidade)
  +1  sma24h_dist          Distancia a SMA de 24h — O REGIME como feature.
                           Ate a V5 so o filtro externo enxergava o regime;
                           agora o proprio modelo sabe em que lado do mercado esta.
  +4  hour_sin/cos,        Sazonalidade intradiaria e semanal (cripto tem
      dow_sin/cos          padroes por horario/dia; encoding circular).

Labels: receita B ("especialista em quedas": ALTA 0.8% / QUEDA 0.4%),
splits identicos ao experimento A/B (train<=2025-06-30, val<=2025-12-31,
test 2026+) — comparacao limpa V6 vs V5.9: SO as features mudam.

Uso:
  python v6_data_prep.py            -> gera data_v6/{split}_{sym}.npz
"""
import os
import gc
import numpy as np
import pandas as pd
import pandas_ta as _pandas_ta
from datetime import datetime
from typing import Any

# Ver nota em v5_data_prep.py: alias tipado para o editor nao confundir os
# submodulos do pandas-ta com as funcoes de mesmo nome.
ta: Any = _pandas_ta

from v5_data_prep import (ASSETS, BTC, WINDOW_SIZE, HORIZON_H, SUBSAMPLE,
                          TRAIN_END, VAL_END, CLASS_NAMES,
                          THRESH_B_UP, THRESH_B_DOWN,
                          _load_parquet, _add_features, _compute_target_asym)

OUTPUT_DIR = "data_v6"


def add_features_v6(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    """18 features da V5 + 8 novas = 26. Mesma normalizacao adimensional."""
    out = _add_features(df, btc)          # 18 base (BB ausentes por causa do bug)
    c = df["close"]

    # ── Bandas de Bollinger (fix: busca por prefixo, robusto a versao) ──
    bb = ta.bbands(c, length=20, std=2)
    if bb is not None:
        u_cols = [col for col in bb.columns if col.startswith("BBU")]
        l_cols = [col for col in bb.columns if col.startswith("BBL")]
        if u_cols and l_cols:
            bbu, bbl = bb[u_cols[0]], bb[l_cols[0]]
            out["dist_bbu"] = (bbu - c) / (c + 1e-9)   # <=0: preco na banda superior
            out["dist_bbl"] = (c - bbl) / (c + 1e-9)   # <=0: preco na banda inferior
            out["bb_width"] = (bbu - bbl) / (c + 1e-9) # squeeze/expansao de vol

    # ── Regime diario como feature (o filtro externo, agora visivel ao modelo) ──
    sma24 = c.rolling(1440).mean()
    out["sma24h_dist"] = (c - sma24) / (c + 1e-9)

    # ── Sazonalidade (encoding circular: 23h e 0h ficam proximos) ──
    idx = out.index
    hora = idx.hour + idx.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hora / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hora / 24.0)
    dow = idx.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    return out.astype("float32")


def prepare_v6():
    """Gera data_v6/ com as 26 features e labels B, nos splits canonicos."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*62}")
    print(f"Preparando V6 — 26 features | labels B "
          f"(ALTA {THRESH_B_UP*100:.1f}% / QUEDA {THRESH_B_DOWN*100:.1f}%)")
    print(f"Splits: train<={TRAIN_END} | val<={VAL_END} | test>{VAL_END}")
    print(f"{'='*62}")

    btc_df = _load_parquet(BTC)
    splits = ["train", "val", "test"]

    for sym in ASSETS:
        out_files = [os.path.join(OUTPUT_DIR, f"{s}_{sym}.npz") for s in splits]
        if all(os.path.exists(p) for p in out_files):
            print(f"  [{sym}] ja existe (3 splits) — pulando.")
            continue

        print(f"  [{sym}] calculando 26 features...", flush=True)
        adf = _load_parquet(sym)
        common = adf.index.intersection(btc_df.index)
        adf, btc_al = adf.loc[common], btc_df.loc[common]

        feat_df = add_features_v6(adf, btc_al)
        feat_df["target"] = _compute_target_asym(adf["close"],
                                                 THRESH_B_UP, THRESH_B_DOWN)
        feat_df = feat_df[feat_df["target"] >= 0]
        feat_df.dropna(inplace=True)

        n_feats = feat_df.shape[1] - 1
        assert n_feats == 26, f"Esperava 26 features, gerou {n_feats}!"

        for split in splits:
            if split == "train":
                mask = feat_df.index <= TRAIN_END
            elif split == "val":
                mask = (feat_df.index > TRAIN_END) & (feat_df.index <= VAL_END)
            else:
                mask = feat_df.index > VAL_END
            sub = feat_df[mask]

            if len(sub) < WINDOW_SIZE + 10:
                print(f"  [{sym}] {split}: dados insuficientes.")
                continue

            cols  = [c for c in sub.columns if c != "target"]
            feats = sub[cols].values
            targs = sub["target"].values

            X_list, y_list = [], []
            for i in range(WINDOW_SIZE, len(feats), SUBSAMPLE):
                X_list.append(feats[i - WINDOW_SIZE:i])
                y_list.append(targs[i])

            X = np.array(X_list, dtype=np.float32)
            y = np.array(y_list, dtype=np.int8)
            dist = {CLASS_NAMES[k]: int((y == k).sum()) for k in [0, 1, 2]}
            print(f"  [{sym}] {split}: {len(X)} amostras | {dist}", flush=True)

            np.savez_compressed(os.path.join(OUTPUT_DIR, f"{split}_{sym}.npz"),
                                X=X, y=y)
            del X, y, X_list, y_list, feats, targs
            gc.collect()

        del feat_df, adf
        gc.collect()

    print(f"\n[V6] Concluido em {OUTPUT_DIR}/ (26 features por amostra)")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    print("=== Trader.AI V6 — Preparacao de Features Enriquecidas ===\n")
    t0 = datetime.now()
    prepare_v6()
    print(f"Concluido em {(datetime.now()-t0).seconds}s.")
