"""
v5_data_prep.py — Trader.AI V5.1 (Defensivo + Agressivo)
Preparacao de features e ALVO DIRECIONAL de 3 classes.

Mudanca chave vs V5.0:
  - Alvo agora e a DIRECAO do ativo nas proximas HORIZON_H horas:
        0 = QUEDA  (retorno < -MOVE_THRESH)  -> sinal defensivo (vender)
        1 = NEUTRO (entre os limites)
        2 = ALTA   (retorno > +MOVE_THRESH)  -> sinal agressivo (comprar)
  - Horizonte de 6h (mais sinal que 2h), mas o modelo roda a cada minuto.
  - Features multi-timeframe via retornos em 5/15/60/240 min
    (e assim que o dado de 1-min "enxerga" os graficos maiores).

Saida: data_v5/{split}_{SYMBOL}.npz  com X (sequencias) e y (classe 0/1/2)
"""
import os
import gc
import numpy as np
import pandas as pd
import pandas_ta as ta
from datetime import datetime

# ── Parametros ────────────────────────────────────────────────────────────────
WINDOW_SIZE = 120    # candles de historia por amostra (2h em 1-min) — era 60
HORIZON_H   = 120    # minutos a frente para o alvo (2 horas) — era 360 (6h)
MOVE_THRESH = 0.008  # +-0.8% em 2h define ALTA / QUEDA  — era 0.015 (1.5%)
SUBSAMPLE   = 15     # 1 candle a cada 15 — arquivos ~640MB (cabe na RAM disponivel)
DATA_DIR    = "data"
OUTPUT_DIR  = "data_v5"

# Todos os ativos sao previstos (inclusive BTC, para o sinal defensivo)
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT"]
BTC    = "BTCUSDT"

# Compatibilidade com codigo antigo
ALTS = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "AVAXUSDT"]
BASE = "BTCUSDT"
FORWARD_MINUTES = HORIZON_H

TRAIN_END = "2025-06-30"   # era 2023-12-31 — adiciona 18 meses (bull 2024 + H1 2025)
VAL_END   = "2025-12-31"   # era 2024-12-31 — H2 2025 para validacao

CLASS_NAMES = {0: "QUEDA", 1: "NEUTRO", 2: "ALTA"}


def _load_parquet(symbol: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{symbol}_1m.parquet")
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df.set_index("date", inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    df.sort_index(inplace=True)
    return df


def _add_features(df: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    """Features do ativo + contexto de mercado (BTC). ~18 colunas."""
    out = pd.DataFrame(index=df.index)
    c = df["close"]

    # Geometria do candle
    out["body"]       = (df["close"] - df["open"]) / c
    out["upper_wick"] = (df["high"]  - df[["open", "close"]].max(axis=1)) / c
    out["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / c
    out["vol_norm"]   = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

    # Retornos multi-timeframe (o "olhar" para graficos maiores)
    for p in [5, 15, 60, 240]:
        out[f"ret_{p}m"] = c.pct_change(p)

    # Indicadores tecnicos
    rsi = ta.rsi(c, length=14)
    out["rsi"] = (rsi - 50) / 50

    macd_df = ta.macd(c, fast=12, slow=26, signal=9)
    if macd_df is not None and "MACDh_12_26_9" in macd_df.columns:
        out["macdh"] = macd_df["MACDh_12_26_9"] / (c + 1e-9)

    atr = ta.atr(df["high"], df["low"], df["close"], length=14)
    out["atr_pct"] = atr / (c + 1e-9)

    bb = ta.bbands(c, length=20, std=2)
    if bb is not None and "BBU_20_2.0" in bb.columns:
        out["dist_bbu"] = (bb["BBU_20_2.0"] - c) / (c + 1e-9)
        out["dist_bbl"] = (c - bb["BBL_20_2.0"]) / (c + 1e-9)

    for ema_p in [20, 50]:
        ema = ta.ema(c, length=ema_p)
        out[f"ema{ema_p}_dist"] = (c - ema) / (c + 1e-9)

    # EMA 200 — tendência maior (acima/abaixo define regime de alta ou baixa)
    ema200 = ta.ema(c, length=200)
    out["ema200_dist"] = (c - ema200) / (c + 1e-9)

    # Z-score do volume — detecta picos de volume (precursores de movimentos bruscos)
    vol_ma  = df["volume"].rolling(20).mean()
    vol_std = df["volume"].rolling(20).std()
    out["vol_zscore"] = (df["volume"] - vol_ma) / (vol_std + 1e-9)

    # Estocástico %K normalizado — oscilador de momentum complementar ao RSI
    stoch = ta.stoch(df["high"], df["low"], df["close"], k=14, d=3, smooth_k=3)
    if stoch is not None:
        k_cols = [col for col in stoch.columns if col.startswith("STOCHk")]
        if k_cols:
            out["stoch_k"] = (stoch[k_cols[0]] - 50) / 50

    # Contexto de mercado: o que o BTC esta fazendo (regime)
    btc_c = btc["close"].reindex(df.index, method="ffill")
    out["btc_ret_60"]  = btc_c.pct_change(60)
    out["btc_ret_240"] = btc_c.pct_change(240)

    return out.astype("float32")


def _compute_target(close: pd.Series) -> pd.Series:
    """
    Direcao nas proximas HORIZON_H velas:
      0 = QUEDA (< -MOVE_THRESH), 1 = NEUTRO, 2 = ALTA (> +MOVE_THRESH)
    """
    fwd_ret = close.shift(-HORIZON_H) / close - 1
    y = pd.Series(1, index=close.index, dtype="int8")  # NEUTRO por padrao
    y[fwd_ret >  MOVE_THRESH] = 2  # ALTA
    y[fwd_ret < -MOVE_THRESH] = 0  # QUEDA
    y[fwd_ret.isna()] = -1         # sem futuro -> descartar
    return y


def _compute_target_asym(close: pd.Series, up_thresh: float,
                         down_thresh: float) -> pd.Series:
    """Target com thresholds assimetricos (ALTA e QUEDA independentes)."""
    fwd_ret = close.shift(-HORIZON_H) / close - 1
    y = pd.Series(1, index=close.index, dtype="int8")
    y[fwd_ret >  up_thresh]   = 2  # ALTA
    y[fwd_ret < -down_thresh] = 0  # QUEDA
    y[fwd_ret.isna()] = -1
    return y


# ── Experimentos V5.9 (dual) ─────────────────────────────────────────────────
# A "mais sinais"          : ALTA/QUEDA simetricos em 0.4% (~25/50/25)
# B "especialista em queda": ALTA rigida 0.8%, QUEDA sensivel 0.4%
OUTPUT_A   = "data_v5a"
THRESH_A_UP, THRESH_A_DOWN = 0.004, 0.004
OUTPUT_B   = "data_v5b"
THRESH_B_UP, THRESH_B_DOWN = 0.008, 0.004


def prepare_dual():
    """
    Gera os DOIS datasets numa unica passada:
      data_v5a/{split}_{sym}.npz : X + y do experimento A
      data_v5b/{split}_{sym}.npz : apenas y do experimento B (X identico ao A)
    Features sao computadas uma vez por ativo (nao por split) — 3x mais rapido.
    """
    os.makedirs(OUTPUT_A, exist_ok=True)
    os.makedirs(OUTPUT_B, exist_ok=True)
    print(f"\n{'='*62}")
    print(f"Preparando V5.9 DUAL — A: +-{THRESH_A_UP*100:.1f}% | "
          f"B: ALTA {THRESH_B_UP*100:.1f}% / QUEDA {THRESH_B_DOWN*100:.1f}%")
    print(f"Janela: {WINDOW_SIZE}m | Horizonte: {HORIZON_H}m | Subsample: 1/{SUBSAMPLE}")
    print(f"{'='*62}")

    btc_df = _load_parquet(BTC)
    print(f"[BTC] {len(btc_df)} candles | {btc_df.index[0].date()} a {btc_df.index[-1].date()}")

    splits = ["train", "val", "test"]

    for sym in ASSETS:
        out_files = ([os.path.join(OUTPUT_A, f"{s}_{sym}.npz") for s in splits] +
                     [os.path.join(OUTPUT_B, f"{s}_{sym}.npz") for s in splits])
        if all(os.path.exists(p) for p in out_files):
            print(f"  [{sym}] ja existe (A+B, 3 splits) — pulando.")
            continue

        print(f"  [{sym}] calculando features...", flush=True)
        adf = _load_parquet(sym)
        common = adf.index.intersection(btc_df.index)
        adf, btc_al = adf.loc[common], btc_df.loc[common]

        feat_df = _add_features(adf, btc_al)
        feat_df["ta"] = _compute_target_asym(adf["close"], THRESH_A_UP, THRESH_A_DOWN)
        feat_df["tb"] = _compute_target_asym(adf["close"], THRESH_B_UP, THRESH_B_DOWN)
        feat_df = feat_df[(feat_df["ta"] >= 0) & (feat_df["tb"] >= 0)]
        feat_df.dropna(inplace=True)

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

            cols = [c for c in sub.columns if c not in ("ta", "tb")]
            feats = sub[cols].values
            t_a   = sub["ta"].values
            t_b   = sub["tb"].values

            X_list, ya_list, yb_list = [], [], []
            for i in range(WINDOW_SIZE, len(feats), SUBSAMPLE):
                X_list.append(feats[i - WINDOW_SIZE:i])
                ya_list.append(t_a[i])
                yb_list.append(t_b[i])

            X   = np.array(X_list, dtype=np.float32)
            y_a = np.array(ya_list, dtype=np.int8)
            y_b = np.array(yb_list, dtype=np.int8)

            dist_a = {CLASS_NAMES[k]: int((y_a == k).sum()) for k in [0, 1, 2]}
            dist_b = {CLASS_NAMES[k]: int((y_b == k).sum()) for k in [0, 1, 2]}
            print(f"  [{sym}] {split}: {len(X)} amostras | "
                  f"A={dist_a} | B={dist_b}", flush=True)

            np.savez_compressed(os.path.join(OUTPUT_A, f"{split}_{sym}.npz"),
                                X=X, y=y_a)
            np.savez_compressed(os.path.join(OUTPUT_B, f"{split}_{sym}.npz"),
                                y=y_b)
            del X, y_a, y_b, X_list, ya_list, yb_list, feats
            gc.collect()

        del feat_df, adf
        gc.collect()

    print(f"\n[DUAL] Concluido: {OUTPUT_A}/ (X+y_A) e {OUTPUT_B}/ (y_B)")
    print(f"{'='*62}\n")


def prepare_all(split: str = "train"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n{'='*62}")
    print(f"Preparando V5.1 — split: {split.upper()}")
    print(f"Janela: {WINDOW_SIZE}m | Horizonte: {HORIZON_H}m ({HORIZON_H//60}h) | "
          f"Limite: +-{MOVE_THRESH*100:.1f}% | Subsample: 1/{SUBSAMPLE}")
    print(f"{'='*62}")

    btc_df = _load_parquet(BTC)
    print(f"[BTC] {len(btc_df)} candles | {btc_df.index[0].date()} a {btc_df.index[-1].date()}")

    total = 0
    for sym in ASSETS:
        out_path = os.path.join(OUTPUT_DIR, f"{split}_{sym}.npz")
        if os.path.exists(out_path):
            n = len(np.load(out_path)["X"])
            print(f"  [{sym}] ja existe: {n} amostras — pulando.")
            total += n
            continue

        print(f"  [{sym}] calculando features...")
        adf = _load_parquet(sym)
        common = adf.index.intersection(btc_df.index)
        adf, btc_al = adf.loc[common], btc_df.loc[common]

        feat_df = _add_features(adf, btc_al)
        feat_df["target"] = _compute_target(adf["close"])
        feat_df = feat_df[feat_df["target"] >= 0]   # remove sem-futuro
        feat_df.dropna(inplace=True)

        if split == "train":
            mask = feat_df.index <= TRAIN_END
        elif split == "val":
            mask = (feat_df.index > TRAIN_END) & (feat_df.index <= VAL_END)
        else:
            mask = feat_df.index > VAL_END
        feat_df = feat_df[mask]

        if len(feat_df) < WINDOW_SIZE + 10:
            print(f"  [{sym}] dados insuficientes.")
            continue

        cols = [c for c in feat_df.columns if c != "target"]
        feats   = feat_df[cols].values
        targets = feat_df["target"].values

        X_list, y_list = [], []
        for i in range(WINDOW_SIZE, len(feats), SUBSAMPLE):
            X_list.append(feats[i - WINDOW_SIZE:i])
            y_list.append(targets[i])

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int8)

        dist = {CLASS_NAMES[k]: int((y == k).sum()) for k in [0, 1, 2]}
        print(f"  [{sym}] {split}: {len(X)} amostras | {dist} | "
              f"{X.nbytes/1024/1024:.0f} MB")

        np.savez_compressed(out_path, X=X, y=y)
        total += len(X)
        del X, y, X_list, y_list, feat_df, feats, targets
        gc.collect()

    print(f"\n[TOTAL {split.upper()}] ~{total} amostras | em {OUTPUT_DIR}/{split}_*.npz")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual", action="store_true",
                    help="Gera datasets dos experimentos A (0.4%% simetrico) "
                         "e B (ALTA 0.8%% / QUEDA 0.4%%) numa unica passada")
    args = ap.parse_args()

    t0 = datetime.now()
    if args.dual:
        print("=== Trader.AI V5.9 — Preparacao DUAL (experimentos A + B) ===\n")
        prepare_dual()
    else:
        print("=== Trader.AI V5.1 — Preparacao de Features (direcional) ===\n")
        for split in ["train", "val", "test"]:
            prepare_all(split)
    print(f"Concluido em {(datetime.now()-t0).seconds}s.")
