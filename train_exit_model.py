"""
train_exit_model.py — V4 (Fase 3.3)
Treina o modelo de SAIDA (exit_model.pkl) com LightGBM.

Objetivo:
  Dado que estamos numa posicao aberta no candle atual,
  prever se o preco vai CAIR de forma significativa nos
  proximos 15 minutos — sinalizando que devemos sair antes
  que o Stop Loss seja atingido.

  Target = 1 (SAIR AGORA) se:
      min(low[j+1 : j+15]) < close[j] * (1 - EXIT_DROP_THRESHOLD)
  Target = 0 (MANTER)   caso contrario.

Features usadas:
  As mesmas do modelo de entrada + indicadores de tendencia curta.
  Sem features de posicao (preco medio, P&L) para manter o modelo
  generalizavel para qualquer par.
"""
import os
import gc
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb
from typing import cast
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, roc_auc_score)
from train_model import FEATURES, ALL_PAIRS, build_features

# ─────────────────────────────────────────────────────────────────────────────
# Configuracao
# ─────────────────────────────────────────────────────────────────────────────
EXIT_DROP_THRESHOLD = 0.003   # queda de 0.3% nos proximos 15 candles = sinal de saida
HORIZON             = 15      # candles a frente para avaliar a queda
CANDLES_PER_PAIR    = 150_000

# Features do modelo de saida (subconjunto + novas features de reversao)
EXIT_FEATURES = [
    'RSI', 'MACDh', 'ATR_pct',
    'Dist_BBU', 'Dist_BBL',
    'ret_1', 'ret_2', 'ret_3',
    'OBV_pct', 'VWAP_dist',
    'ROC_5', 'ROC_15',
    'Body_Size', 'Upper_Wick', 'Lower_Wick', 'Wick_Ratio',
    'vol_ratio', 'RSI_slope', 'EMA20_dist',
    'Cup_and_Handle',
    # Features adicionais de reversao
    'RSI_high',      # RSI no maximo dos ultimos 5 candles (sobrecompra recente)
    'BB_squeeze',    # Largura das Bandas de Bollinger normalizada (volatilidade em queda = reversao)
    'Bear_momentum', # MACDh declinante (sinal de enfraquecimento)
]

LGBM_EXIT_PARAMS = dict(
    n_estimators     = 400,
    learning_rate    = 0.05,
    max_depth        = 5,
    num_leaves       = 31,
    min_child_samples= 60,
    subsample        = 0.80,
    colsample_bytree = 0.80,
    reg_alpha        = 0.1,
    reg_lambda       = 0.3,
    class_weight     = 'balanced',
    random_state     = 42,
    n_jobs           = -1,
    verbose          = -1,
)


def build_exit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula todos os indicadores e cria o target de saida numa passagem unica.
    Evita problemas de alinhamento de indice construindo tudo no mesmo DataFrame.
    """
    df = df.copy()

    # Garante indice DatetimeIndex desde o inicio
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df = df.set_index('date')
        else:
            return pd.DataFrame()

    # ── Indicadores ───────────────────────────────────────────────────────
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.vwap(append=True)
    df.ta.roc(length=5, append=True)
    df.ta.roc(length=15, append=True)

    cols = df.columns.tolist()
    try:
        rsi_col  = next(c for c in cols if c.startswith('RSI_'))
        macd_col = next(c for c in cols if c.startswith('MACDh_'))
        bbu_col  = next(c for c in cols if c.startswith('BBU_'))
        bbl_col  = next(c for c in cols if c.startswith('BBL_'))
        atr_col  = next(c for c in cols if c.startswith('ATRr_'))
        obv_col  = next(c for c in cols if c.startswith('OBV'))
        vwap_col = next(c for c in cols if c.startswith('VWAP'))
        roc5_col = next(c for c in cols if c.startswith('ROC_5'))
        roc15_col= next(c for c in cols if c.startswith('ROC_15'))
        ema20_col= next((c for c in cols if c.startswith('EMA_20')), None)
    except StopIteration:
        return pd.DataFrame()

    # ── Features canonicas (mesmas do modelo de entrada) ─────────────────
    df['RSI']       = df[rsi_col]
    df['MACDh']     = df[macd_col]
    df['ATR_pct']   = df[atr_col] / df['close']
    df['Dist_BBU']  = (df['close'] - df[bbu_col]) / df['close']
    df['Dist_BBL']  = (df['close'] - df[bbl_col]) / df['close']
    df['ROC_5']     = df[roc5_col]
    df['ROC_15']    = df[roc15_col]
    df['OBV_pct']   = df[obv_col].pct_change(1).fillna(0).clip(-1, 1)
    df['VWAP_dist'] = (df['close'] - df[vwap_col]) / (df[vwap_col] + 1e-9)

    df['ret_1'] = df['close'].pct_change(1)
    df['ret_2'] = df['close'].pct_change(2)
    df['ret_3'] = df['close'].pct_change(3)

    df['Body_Size']  = abs(df['close'] - df['open']) / df['close']
    df['Upper_Wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
    df['Lower_Wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
    df['Wick_Ratio'] = df['Lower_Wick'] / (df['Upper_Wick'] + 1e-9)

    vol_sma20 = df['volume'].rolling(window=20).mean()
    df['vol_ratio']  = df['volume'] / (vol_sma20 + 1e-9)
    df['RSI_slope']  = df['RSI'] - df['RSI'].shift(3)
    df['EMA20_dist'] = (df['close'] - df[ema20_col]) / (df['close'] + 1e-9) if ema20_col else 0.0

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

    # ── Features adicionais de reversao ──────────────────────────────────
    df['RSI_high']      = df['RSI'].rolling(window=5).max()
    df['BB_squeeze']    = (df[bbu_col] - df[bbl_col]) / (df['close'] + 1e-9)
    df['Bear_momentum'] = (df['MACDh'] - df['MACDh'].shift(2)).clip(-1, 1)

    # ── Target de saida ───────────────────────────────────────────────────
    # target=1 se o low dos proximos HORIZON candles cair EXIT_DROP_THRESHOLD
    # abaixo do close atual
    n          = len(df)
    close_vals = df['close'].values
    low_vals   = df['low'].values

    target_exit = np.zeros(n, dtype=int)
    for i in range(n - HORIZON):
        threshold = close_vals[i] * (1 - EXIT_DROP_THRESHOLD)
        for j in range(1, HORIZON + 1):
            if low_vals[i + j] < threshold:
                target_exit[i] = 1
                break

    df['target_exit'] = target_exit

    # Garante todas as colunas necessarias
    for col in EXIT_FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    return cast(pd.DataFrame, df[EXIT_FEATURES + ['target_exit']].dropna())


def walk_forward_exit(df_model: pd.DataFrame, n_folds: int = 4) -> dict:
    """Walk-Forward Validation para o modelo de saida."""
    from lightgbm import LGBMClassifier

    n       = len(df_model)
    fold_sz = n // (n_folds + 1)
    precisions, recalls, aucs = [], [], []
    best_model = None

    print(f"\n[WFV-EXIT] Walk-Forward Validation com {n_folds} folds ({fold_sz:,} candles/fold)...")

    for fold in range(n_folds):
        train_end  = fold_sz * (fold + 1)
        test_start = train_end
        test_end   = min(train_end + fold_sz, n)

        X_train = df_model[EXIT_FEATURES].iloc[:train_end]
        y_train = df_model['target_exit'].iloc[:train_end]
        X_test  = df_model[EXIT_FEATURES].iloc[test_start:test_end]
        y_test  = df_model['target_exit'].iloc[test_start:test_end]

        clf = LGBMClassifier(**LGBM_EXIT_PARAMS)
        clf.fit(X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.log_evaluation(period=-1)])

        proba = np.asarray(clf.predict_proba(X_test))[:, 1]
        preds = (proba >= 0.50).astype(int)

        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        a = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.5

        precisions.append(p)
        recalls.append(r)
        aucs.append(a)
        best_model = clf

        print(f"  Fold {fold+1}: Precision={p:.3f}  Recall={r:.3f}  AUC={a:.3f}"
              f"  (treino={train_end:,} | teste={test_end - test_start:,})")

    return {
        'avg_precision': round(np.mean(precisions), 4),
        'avg_recall':    round(np.mean(recalls), 4),
        'avg_auc':       round(np.mean(aucs), 4),
        'last_model':    best_model,
    }


def train_exit_model(pairs: list = None, candles_per_pair: int = CANDLES_PER_PAIR):
    """Treina e salva o modelo de saida."""
    from lightgbm import LGBMClassifier

    if pairs is None:
        pairs = ALL_PAIRS

    model_dir  = os.path.dirname(os.path.abspath(__file__))
    data_dir   = os.path.join(model_dir, "data")
    model_path = os.path.join(model_dir, "exit_model.pkl")

    print("=" * 60)
    print("TRADER.AI V4 - Treino do Modelo de Saida (LightGBM)")
    print(f"Pares: {pairs}")
    print(f"Candles por par: {candles_per_pair:,}")
    print(f"Horizon: {HORIZON} candles | Drop threshold: {EXIT_DROP_THRESHOLD*100:.1f}%")
    print("=" * 60)

    all_dfs = []

    for symbol in pairs:
        parquet_path = os.path.join(data_dir, f"{symbol}_1m.parquet")
        if not os.path.exists(parquet_path):
            print(f"[EXIT] AVISO: {parquet_path} nao encontrado. Pulando {symbol}.")
            continue

        print(f"\n[EXIT] Carregando {symbol}...")
        df = pd.read_parquet(parquet_path)
        df = df.sort_values('date').reset_index(drop=True)
        df = df.tail(candles_per_pair).reset_index(drop=True)
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

        print(f"[EXIT] Feature engineering em {symbol} ({len(df):,} candles)...")
        df_feat = build_exit_features(df)

        if df_feat.empty:
            print(f"[EXIT] Sem dados validos para {symbol}. Pulando.")
            continue

        all_dfs.append(df_feat)
        gc.collect()

    if not all_dfs:
        print("[EXIT] ERRO: Nenhum dado valido para treinar o modelo de saida!")
        return

    df_model = pd.concat(all_dfs, ignore_index=True)
    del all_dfs
    gc.collect()

    exit_rate = df_model['target_exit'].mean()
    print(f"\n[EXIT] Dataset combinado: {len(df_model):,} amostras")
    print(f"[EXIT] Distribuicao target: SAIR={exit_rate*100:.1f}%  MANTER={(1-exit_rate)*100:.1f}%")

    # Walk-Forward Validation
    wfv = walk_forward_exit(df_model, n_folds=4)

    # Modelo final com 100% dos dados
    print("\n[EXIT] Treinando modelo final em 100% dos dados...")
    n_split = int(len(df_model) * 0.90)
    X_all   = df_model[EXIT_FEATURES]
    y_all   = df_model['target_exit']
    X_val   = X_all.iloc[n_split:]
    y_val   = y_all.iloc[n_split:]

    final_model = LGBMClassifier(**LGBM_EXIT_PARAMS)
    final_model.fit(
        X_all, y_all,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=50)],
    )

    # Avaliacao final
    print("\n[EXIT] --- Avaliacao Final (ultimos 10%) ---")
    proba_val = np.asarray(final_model.predict_proba(X_val))[:, 1]
    preds_val = (proba_val >= 0.50).astype(int)

    acc  = accuracy_score(y_val, preds_val)
    prec = precision_score(y_val, preds_val, zero_division=0)
    rec  = recall_score(y_val, preds_val, zero_division=0)
    auc  = roc_auc_score(y_val, proba_val) if len(np.unique(y_val)) > 1 else 0.5
    print(f"Acuracia : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print(classification_report(y_val, preds_val, zero_division=0))

    # Feature importance
    print("\n[EXIT] Importancia das Features (gain):")
    importance = pd.Series(
        final_model.booster_.feature_importance(importance_type='gain'),
        index=EXIT_FEATURES
    ).sort_values(ascending=False)
    max_imp = importance.max() or 1
    for feat, imp in importance.items():
        bar = '#' * int(imp / max_imp * 30)
        print(f"  {feat:<20} {bar:<30} {int(imp)}")

    # Salvar
    meta = {
        'model':           final_model,
        'features':        EXIT_FEATURES,
        'drop_threshold':  EXIT_DROP_THRESHOLD,
        'horizon':         HORIZON,
        'wfv_precision':   wfv['avg_precision'],
        'wfv_recall':      wfv['avg_recall'],
        'wfv_auc':         wfv['avg_auc'],
        'pairs_trained':   pairs,
    }
    joblib.dump(meta, model_path)
    print(f"\n[EXIT] Modelo salvo em: {model_path}")
    print(f"[EXIT] WFV Precision media: {wfv['avg_precision']} | AUC media: {wfv['avg_auc']}")
    print("[EXIT] Concluido!")


if __name__ == "__main__":
    train_exit_model()
