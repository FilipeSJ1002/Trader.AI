"""
train_model.py — V4 (Fase 3)
Treina o modelo de ENTRADA (scalper_model.pkl) com LightGBM.

Melhorias da Fase 3:
  - LightGBM: 3-5x mais rapido e preciso que Random Forest
  - Multi-ativo: treina com todos os 6 pares combinados
  - Target ajustado ao risco: trade e "sucesso" se TP e atingido antes do SL
  - Walk-Forward Validation: garante que o modelo funciona no futuro real
  - Feature importance detalhada no final
"""
import os
import gc
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb
from dotenv import load_dotenv
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, roc_auc_score)

# ─────────────────────────────────────────────────────────────────────────────
# Lista canonica de features — importada por strategy.py, backtest.py, etc.
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    'RSI', 'MACDh', 'ATR_pct',
    'Dist_BBU', 'Dist_BBL',
    'ret_1', 'ret_2', 'ret_3',
    'OBV_pct',
    'VWAP_dist',
    'ROC_5', 'ROC_15',
    'Body_Size', 'Upper_Wick', 'Lower_Wick', 'Wick_Ratio',
    'vol_ratio',   # volume atual / SMA20 volume — força do movimento
    'RSI_slope',   # RSI[t] - RSI[t-3] — aceleração do momentum
    'EMA20_dist',  # (close - EMA20) / close — posição relativa à tendência curta
    'Cup_and_Handle'
]

# Pares usados no treino
ALL_PAIRS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"]

# Hiperparametros LightGBM (entry model)
LGBM_PARAMS = dict(
    n_estimators    = 600,
    learning_rate   = 0.04,
    max_depth       = 6,
    num_leaves      = 40,
    min_child_samples = 80,
    subsample       = 0.80,
    colsample_bytree = 0.80,
    reg_alpha       = 0.1,
    reg_lambda      = 0.2,
    class_weight    = 'balanced',
    random_state    = 42,
    n_jobs          = -1,
    verbose         = -1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df: pd.DataFrame, atr_sl_mult: float = 1.2,
                   atr_tp_mult: float = 3.0) -> pd.DataFrame:
    """
    Calcula todos os indicadores e cria o target ajustado ao risco.

    Target (V4):
      1 = trade vencedor: o high dos proximos 30 candles atinge o TP
                         (close + atr_tp_mult * ATR) ANTES de o low
                         atingir o SL (close - atr_sl_mult * ATR)
      0 = trade perdedor ou ambiguo
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df.set_index('date', inplace=True)

    # Indicadores
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.vwap(append=True)
    df.ta.roc(length=5, append=True)
    df.ta.roc(length=15, append=True)

    df['ret_1'] = df['close'].pct_change(1)
    df['ret_2'] = df['close'].pct_change(2)
    df['ret_3'] = df['close'].pct_change(3)

    df['Body_Size']  = abs(df['close'] - df['open']) / df['close']
    df['Upper_Wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
    df['Lower_Wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
    df['Wick_Ratio'] = df['Lower_Wick'] / (df['Upper_Wick'] + 1e-9)

    # Volume ratio (volume atual / SMA20 do volume)
    vol_sma20 = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / (vol_sma20 + 1e-9)

    # RSI slope (aceleração do momentum — RSI[t] − RSI[t-3])
    # Calcula depois de mapear RSI
    # EMA20 para distância de tendência curta
    df.ta.ema(length=20, append=True)

    # Cup and Handle
    H1 = df['high'].shift(5).rolling(window=35).max()
    L1 = df['low'].shift(5).rolling(window=35).min()
    H2 = df['high'].rolling(window=5).max()
    cup_depth       = (H1 - L1) / H1
    cup_edges_match = abs(H1 - H2) / H1
    handle_drop     = (H2 - df['close']) / H2
    df['Cup_and_Handle'] = (
        (cup_depth > 0.015) & (cup_edges_match < 0.01) &
        (handle_drop > 0.002) & (handle_drop < 0.01)
    ).astype(int)

    # Mapeamento pandas_ta → nomes canonicos
    cols = df.columns.tolist()
    try:
        rsi_col   = next(c for c in cols if c.startswith('RSI_'))
        macd_col  = next(c for c in cols if c.startswith('MACDh_'))
        bbu_col   = next(c for c in cols if c.startswith('BBU_'))
        bbl_col   = next(c for c in cols if c.startswith('BBL_'))
        atr_col   = next(c for c in cols if c.startswith('ATRr_'))
        obv_col   = next(c for c in cols if c.startswith('OBV'))
        vwap_col  = next(c for c in cols if c.startswith('VWAP'))
        roc5_col  = next(c for c in cols if c.startswith('ROC_5'))
        roc15_col = next(c for c in cols if c.startswith('ROC_15'))
    except StopIteration as e:
        print(f"[BUILD] Coluna nao encontrada: {e}")
        return pd.DataFrame()

    df['RSI']      = df[rsi_col]
    df['MACDh']    = df[macd_col]
    df['ATR_pct']  = df[atr_col] / df['close']
    df['Dist_BBU'] = (df['close'] - df[bbu_col]) / df['close']
    df['Dist_BBL'] = (df['close'] - df[bbl_col]) / df['close']
    df['ROC_5']    = df[roc5_col]
    df['ROC_15']   = df[roc15_col]
    df['OBV_pct']  = df[obv_col].pct_change(1).fillna(0).clip(-1, 1)
    df['VWAP_dist']= (df['close'] - df[vwap_col]) / (df[vwap_col] + 1e-9)

    # RSI slope e EMA20 distance (substitutos das features CDL que requerem TA-Lib)
    df['RSI_slope']  = df['RSI'] - df['RSI'].shift(3)
    ema20_col = next((c for c in df.columns if c.startswith('EMA_20')), None)
    if ema20_col:
        df['EMA20_dist'] = (df['close'] - df[ema20_col]) / (df['close'] + 1e-9)
    else:
        df['EMA20_dist'] = 0.0

    # ── Target ajustado ao risco (V4 Fase 3) ─────────────────────────────
    # Para cada candle i, simula uma entrada no close[i] com:
    #   SL = close - atr_sl_mult * ATR
    #   TP = close + atr_tp_mult * ATR
    # Nos proximos 30 candles, verifica qual e atingido primeiro.
    # target=1 se TP for atingido antes do SL (ou TP atingido antes do fim)
    # target=0 se SL for atingido primeiro, ou nenhum dos dois (trade neutro)
    horizon = 30
    sl_price = df['close'] - atr_sl_mult * df[atr_col]
    tp_price = df['close'] + atr_tp_mult * df[atr_col]

    target = pd.Series(0, index=df.index)
    high_arr = df['high'].values
    low_arr  = df['low'].values
    tp_arr   = tp_price.values
    sl_arr   = sl_price.values
    n        = len(df)

    for i in range(n - horizon):
        tp_hit = False
        sl_hit = False
        for j in range(1, horizon + 1):
            if high_arr[i + j] >= tp_arr[i]:
                tp_hit = True
                break
            if low_arr[i + j] <= sl_arr[i]:
                sl_hit = True
                break
        if tp_hit and not sl_hit:
            target.iloc[i] = 1

    df['target'] = target

    return df[FEATURES + ['target']].dropna()


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_eval(df_model: pd.DataFrame, n_folds: int = 4) -> dict:
    """
    Avalia o modelo em N janelas temporais distintas.
    Cada janela treina no passado e testa no futuro imediato.
    Retorna metricas medias e o modelo treinado na janela mais recente.
    """
    n       = len(df_model)
    fold_sz = n // (n_folds + 1)

    precisions, recalls, aucs = [], [], []
    best_model = None

    print(f"\n[WFV] Walk-Forward Validation com {n_folds} folds ({fold_sz:,} candles/fold)...")

    for fold in range(n_folds):
        train_end   = fold_sz * (fold + 1)
        test_start  = train_end
        test_end    = min(train_end + fold_sz, n)

        X_tr = df_model[FEATURES].iloc[:train_end]
        y_tr = df_model['target'].iloc[:train_end]
        X_te = df_model[FEATURES].iloc[test_start:test_end]
        y_te = df_model['target'].iloc[test_start:test_end]

        if y_te.sum() < 10:
            print(f"  Fold {fold+1}: poucos positivos, ignorando.")
            continue

        m = lgb.LGBMClassifier(**LGBM_PARAMS)
        m.fit(X_tr, y_tr,
              eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(period=-1)])

        y_prob = m.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)
        auc  = roc_auc_score(y_te, y_prob)

        precisions.append(prec)
        recalls.append(rec)
        aucs.append(auc)
        best_model = m

        print(f"  Fold {fold+1}: Precision={prec:.3f}  Recall={rec:.3f}  AUC={auc:.3f}  "
              f"(treino={train_end:,} | teste={test_end-test_start:,})")

    results = {
        'avg_precision': float(np.mean(precisions)) if precisions else 0.0,
        'avg_recall':    float(np.mean(recalls))    if recalls    else 0.0,
        'avg_auc':       float(np.mean(aucs))       if aucs       else 0.0,
        'best_model':    best_model
    }

    print(f"\n[WFV] Media: Precision={results['avg_precision']:.3f}  "
          f"Recall={results['avg_recall']:.3f}  AUC={results['avg_auc']:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Treino principal
# ─────────────────────────────────────────────────────────────────────────────
def train(pairs: list[str] = None, candles_per_pair: int = 150_000):
    """
    Treina o modelo de entrada com LightGBM usando dados multi-ativo.

    pairs            : lista de pares; None = todos os 6
    candles_per_pair : ultimos N candles de cada par (padrao 150k ~ 3.5 meses)
    """
    load_dotenv()
    if pairs is None:
        pairs = ALL_PAIRS

    print("=" * 60)
    print("TRADER.AI V4 — Treino do Modelo de Entrada (LightGBM)")
    print(f"Pares: {pairs}")
    print(f"Candles por par: {candles_per_pair:,}")
    print("=" * 60)

    frames = []
    for sym in pairs:
        path = os.path.join("data", f"{sym}_1m.parquet")
        if not os.path.exists(path):
            print(f"[TRAIN] AVISO: {path} nao encontrado. Pulando.")
            continue
        print(f"[TRAIN] Carregando {sym}...")
        raw = pd.read_parquet(path)
        raw = raw.tail(candles_per_pair).copy()
        print(f"[TRAIN] Feature engineering em {sym} ({len(raw):,} candles)...")
        feat = build_features(raw)
        if feat.empty:
            print(f"[TRAIN] AVISO: {sym} gerou DataFrame vazio. Pulando.")
            continue
        frames.append(feat)
        del raw, feat
        gc.collect()

    if not frames:
        print("[TRAIN] ERRO: Nenhum dado disponivel para treino.")
        return

    df_model = pd.concat(frames, ignore_index=True).dropna()
    del frames
    gc.collect()

    print(f"\n[TRAIN] Dataset combinado: {len(df_model):,} amostras")
    dist = df_model['target'].value_counts(normalize=True) * 100
    print(f"[TRAIN] Distribuicao target: WIN={dist.get(1, 0):.1f}%  LOSS={dist.get(0, 0):.1f}%")

    # ── Walk-Forward Validation ───────────────────────────────────────
    wfv = walk_forward_eval(df_model, n_folds=4)

    # ── Treino final em todo o dataset ────────────────────────────────
    print("\n[TRAIN] Treinando modelo final em 100% dos dados...")
    X_all = df_model[FEATURES]
    y_all = df_model['target']

    # Split 90/10 para validacao final (sem early stopping aqui)
    split = int(len(df_model) * 0.90)
    X_tr, X_val = X_all.iloc[:split], X_all.iloc[split:]
    y_tr, y_val = y_all.iloc[:split], y_all.iloc[split:]

    final_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    final_model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=50)]
    )

    y_prob = final_model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    print("\n[TRAIN] --- Avaliacao Final (ultimos 10%) ---")
    print(f"Acuracia : {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"AUC-ROC  : {roc_auc_score(y_val, y_prob):.4f}")
    print(classification_report(y_val, y_pred))

    # ── Importancia das features ──────────────────────────────────────
    importances = sorted(
        zip(FEATURES, final_model.feature_importances_),
        key=lambda x: -x[1]
    )
    print("\n[TRAIN] Importancia das Features (gain):")
    for feat, imp in importances:
        bar = "#" * int(imp / max(v for _, v in importances) * 30)
        print(f"  {feat:<20} {bar:<30} {imp:.0f}")

    # ── Salva o modelo ────────────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), "scalper_model.pkl")
    meta = {
        'model':         final_model,
        'features':      FEATURES,
        'wfv_precision': wfv['avg_precision'],
        'wfv_auc':       wfv['avg_auc'],
        'pairs_trained': pairs,
    }
    joblib.dump(meta, model_path)
    print(f"\n[TRAIN] Modelo salvo em: {model_path}")
    print(f"[TRAIN] WFV Precision media: {wfv['avg_precision']:.4f} | AUC media: {wfv['avg_auc']:.4f}")
    print("[TRAIN] Concluido!")

    return meta  # retorna dict completo (usado pelo retrain_scheduler)


# Alias publico para importacao pelo retrain_scheduler.py
def train_model(pairs: list = None, candles_per_pair: int = 150_000) -> dict:
    """Alias de train() com assinatura padronizada para o retrain_scheduler."""
    return train(pairs=pairs, candles_per_pair=candles_per_pair)


if __name__ == "__main__":
    train()
