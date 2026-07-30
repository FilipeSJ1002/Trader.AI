"""
train_model.py — V5 (Fase 5)
Treina o modelo de ENTRADA (scalper_model.pkl) com LightGBM.

Mudancas da Fase 5 (overhaul do ML):
  - DADOS: usa TODO o historico disponivel (4-5 anos), nao so 150k candles.
           Treina em multiplos regimes (bear 2022, recuperacao 2023, bull 2024-25).
  - FEATURES MULTI-TIMEFRAME: alem das 20 features de 1M, adiciona contexto de
           15M e 1H (a "visao do todo" que faltava). Sem lookahead (shift+ffill).
  - TARGET R:R: alvo de qualidade de entrada — 1 se TP (+2xATR) e atingido ANTES
           do SL (-1xATR) nos proximos 60 candles (1h). Vetorizado (rapido).
  - MEMORIA: processa par-a-par, downcast float32, gc agressivo.
"""
import os
import gc
import joblib
import numpy as np
import pandas as pd
import pandas_ta as ta
import lightgbm as lgb
from typing import Optional, cast
from dotenv import load_dotenv
from sklearn.metrics import (classification_report, accuracy_score,
                             precision_score, recall_score, roc_auc_score)

# ─────────────────────────────────────────────────────────────────────────────
# Lista canonica de features — importada por strategy.py, backtest.py, etc.
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    # ── Contexto 1M (microestrutura / timing) ──
    'RSI', 'MACDh', 'ATR_pct',
    'Dist_BBU', 'Dist_BBL',
    'ret_1', 'ret_2', 'ret_3',
    'OBV_pct', 'VWAP_dist',
    'ROC_5', 'ROC_15',
    'Body_Size', 'Upper_Wick', 'Lower_Wick', 'Wick_Ratio',
    'vol_ratio', 'RSI_slope', 'EMA20_dist',
    'Cup_and_Handle',
    # ── Contexto 15M (estrutura intraday) ──
    'RSI_15m', 'MACDh_15m', 'trend_15m', 'ATRpct_15m',
    # ── Contexto 1H (a TENDENCIA real) ──
    'RSI_1h', 'trend_1h', 'ret_1h',
]

# Pares usados no treino
ALL_PAIRS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"]

# Parametros do TARGET (qualidade de entrada R:R)
TARGET_TP_MULT  = 2.0   # TP = close + 2.0 x ATR(1M)
TARGET_SL_MULT  = 1.0   # SL = close - 1.0 x ATR(1M)
TARGET_HORIZON  = 60    # avalia os proximos 60 candles (1 hora)

# Hiperparametros LightGBM (entry model)
LGBM_PARAMS = dict(
    n_estimators      = 700,
    learning_rate     = 0.03,
    max_depth         = 7,
    num_leaves        = 56,
    min_child_samples = 120,
    subsample         = 0.80,
    colsample_bytree  = 0.80,
    reg_alpha         = 0.15,
    reg_lambda        = 0.30,
    class_weight      = 'balanced',
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers Multi-Timeframe (sem lookahead)
# ─────────────────────────────────────────────────────────────────────────────
def _resample_htf(df_1m: pd.DataFrame, rule: str, ema_len: int) -> pd.DataFrame:
    """
    Resample 1M -> timeframe maior (15min/1h) e calcula indicadores.
    Retorna DataFrame indexado no inicio de cada candle HTF.
    """
    agg = {'open': 'first', 'high': 'max', 'low': 'min',
           'close': 'last', 'volume': 'sum'}
    htf = df_1m[['open', 'high', 'low', 'close', 'volume']].resample(rule).agg(agg).dropna()
    if len(htf) < ema_len + 30:
        return pd.DataFrame()

    htf.ta.rsi(length=14, append=True)
    htf.ta.macd(fast=12, slow=26, signal=9, append=True)
    htf.ta.atr(length=14, append=True)
    htf.ta.ema(length=ema_len, append=True)

    cols = htf.columns
    rsi_c  = next(c for c in cols if c.startswith('RSI_'))
    mh_c   = next(c for c in cols if c.startswith('MACDh_'))
    atr_c  = next(c for c in cols if c.startswith('ATRr_'))
    ema_c  = next(c for c in cols if c.startswith(f'EMA_{ema_len}'))

    out = pd.DataFrame(index=htf.index)
    out['RSI']    = htf[rsi_c]
    out['MACDh']  = htf[mh_c]
    out['ATRpct'] = htf[atr_c] / htf['close']
    out['trend']  = (htf['close'] - htf[ema_c]) / (htf['close'] + 1e-9)
    out['ret']    = htf['close'].pct_change(1)
    return out.dropna()


def _merge_htf(df_1m: pd.DataFrame, htf: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """
    Alinha features HTF ao index 1M SEM lookahead:
      - shift(1) garante que cada barra 1M so ve o ULTIMO candle HTF FECHADO
      - reindex + ffill propaga o valor ate o proximo candle HTF
    """
    if htf.empty:
        for col in ['RSI', 'MACDh', 'ATRpct', 'trend', 'ret']:
            df_1m[f'{prefix}_{col}'] = 0.0
        return df_1m

    htf_shift = htf.shift(1)  # evita lookahead
    aligned   = htf_shift.reindex(df_1m.index, method='ffill')
    for col in htf.columns:
        df_1m[f'{prefix}_{col}'] = aligned[col].values
    return df_1m


# ─────────────────────────────────────────────────────────────────────────────
# Target vetorizado (qualidade de entrada R:R)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_target_rr(df: pd.DataFrame, atr_col: str,
                       tp_mult: float = TARGET_TP_MULT,
                       sl_mult: float = TARGET_SL_MULT,
                       horizon: int = TARGET_HORIZON) -> np.ndarray:
    """
    Para cada candle i, simula entrada em close[i]:
        TP = close[i] + tp_mult * ATR[i]
        SL = close[i] - sl_mult * ATR[i]
    target=1 se, nos proximos `horizon` candles, o high atinge TP ANTES do
    low atingir SL. Conservador: se TP e SL caem no mesmo candle, conta como SL.

    Vetorizado: loop sobre o horizonte (60 iteracoes), nao sobre n (milhoes).
    """
    n     = len(df)
    close = df['close'].to_numpy(dtype=np.float64)
    high  = df['high'].to_numpy(dtype=np.float64)
    low   = df['low'].to_numpy(dtype=np.float64)
    atr   = df[atr_col].to_numpy(dtype=np.float64)

    tp = close + tp_mult * atr
    sl = close - sl_mult * atr

    tp_first = np.zeros(n, dtype=bool)
    resolved = np.zeros(n, dtype=bool)

    for k in range(1, horizon + 1):
        h_k = np.full(n, -np.inf)
        l_k = np.full(n,  np.inf)
        h_k[:n - k] = high[k:]
        l_k[:n - k] = low[k:]

        hit_tp = (h_k >= tp) & ~resolved
        hit_sl = (l_k <= sl) & ~resolved
        tp_only = hit_tp & ~hit_sl          # TP so vale se SL nao bateu no mesmo candle
        tp_first |= tp_only
        resolved |= (hit_tp | hit_sl)

    target = tp_first.astype(np.float32)
    target[n - horizon:] = np.nan           # ultimas barras nao tem futuro suficiente
    return target


# ─────────────────────────────────────────────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────
def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula TODAS as features (1M + MTF 15M/1H) SEM o target.
    Usada tanto no treino quanto na inferencia (strategy.py / backtest.py),
    garantindo paridade exata. Retorna o df com as colunas de FEATURES preenchidas.
    """
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    # ── Indicadores 1M ────────────────────────────────────────────────────
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.vwap(append=True)
    df.ta.roc(length=5, append=True)
    df.ta.roc(length=15, append=True)
    df.ta.ema(length=20, append=True)

    df['ret_1'] = df['close'].pct_change(1)
    df['ret_2'] = df['close'].pct_change(2)
    df['ret_3'] = df['close'].pct_change(3)
    df['Body_Size']  = abs(df['close'] - df['open']) / df['close']
    df['Upper_Wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
    df['Lower_Wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
    df['Wick_Ratio'] = df['Lower_Wick'] / (df['Upper_Wick'] + 1e-9)

    vol_sma20 = df['volume'].rolling(window=20).mean()
    df['vol_ratio'] = df['volume'] / (vol_sma20 + 1e-9)

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

    # Mapeamento pandas_ta -> nomes canonicos
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
        ema20_col = next(c for c in cols if c.startswith('EMA_20'))
    except StopIteration as e:
        print(f"[BUILD] Coluna nao encontrada: {e}")
        return pd.DataFrame()

    df['RSI']       = df[rsi_col]
    df['MACDh']     = df[macd_col]
    df['ATR_pct']   = df[atr_col] / df['close']
    df['Dist_BBU']  = (df['close'] - df[bbu_col]) / df['close']
    df['Dist_BBL']  = (df['close'] - df[bbl_col]) / df['close']
    df['ROC_5']     = df[roc5_col]
    df['ROC_15']    = df[roc15_col]
    df['OBV_pct']   = df[obv_col].pct_change(1).fillna(0).clip(-1, 1)
    df['VWAP_dist'] = (df['close'] - df[vwap_col]) / (df[vwap_col] + 1e-9)
    df['RSI_slope'] = df['RSI'] - df['RSI'].shift(3)
    df['EMA20_dist']= (df['close'] - df[ema20_col]) / (df['close'] + 1e-9)

    # ── Features Multi-Timeframe (15M + 1H) ───────────────────────────────
    htf_15 = _resample_htf(df, '15min', ema_len=50)   # EMA50 15M = 12.5h trend
    htf_1h = _resample_htf(df, '1h',    ema_len=200)   # EMA200 1H = ~8d trend (real)

    df = _merge_htf(df, htf_15, prefix='m15')
    df = _merge_htf(df, htf_1h, prefix='h1')

    # Renomeia para os nomes canonicos da FEATURES
    df['RSI_15m']    = df['m15_RSI']
    df['MACDh_15m']  = df['m15_MACDh']
    df['trend_15m']  = df['m15_trend']
    df['ATRpct_15m'] = df['m15_ATRpct']
    df['RSI_1h']     = df['h1_RSI']
    df['trend_1h']   = df['h1_trend']
    df['ret_1h']     = df['h1_ret']

    # ATR absoluto reconstruido (usado pelo target; evita depender do nome cru)
    df['ATR_abs'] = df['ATR_pct'] * df['close']

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engenharia de features + TARGET R:R. Usada no TREINO.
    Retorna apenas FEATURES + target (float32), com dropna.
    """
    df = add_all_features(df)
    if df.empty or 'ATR_abs' not in df.columns:
        return pd.DataFrame()

    # ── Target R:R (vetorizado) ───────────────────────────────────────────
    df['target'] = _compute_target_rr(df, 'ATR_abs')

    out = cast(pd.DataFrame,
               df[FEATURES + ['target']].replace([np.inf, -np.inf], np.nan).dropna())
    for c in FEATURES:
        out[c] = out[c].astype(np.float32)
    out['target'] = out['target'].astype(np.int8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward Validation
# ─────────────────────────────────────────────────────────────────────────────
def walk_forward_eval(df_model: pd.DataFrame, n_folds: int = 5) -> dict:
    """Avalia o modelo em N janelas temporais (treina passado, testa futuro)."""
    n       = len(df_model)
    fold_sz = n // (n_folds + 1)

    precisions, recalls, aucs = [], [], []
    best_model = None

    print(f"\n[WFV] Walk-Forward Validation: {n_folds} folds ({fold_sz:,} candles/fold)...")

    for fold in range(n_folds):
        train_end  = fold_sz * (fold + 1)
        test_start = train_end
        test_end   = min(train_end + fold_sz, n)

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
              callbacks=[lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(period=-1)])

        y_prob = np.asarray(m.predict_proba(X_te))[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        prec = precision_score(y_te, y_pred, zero_division=0)
        rec  = recall_score(y_te, y_pred, zero_division=0)
        auc  = roc_auc_score(y_te, y_prob)

        precisions.append(prec); recalls.append(rec); aucs.append(auc)
        best_model = m
        print(f"  Fold {fold+1}: Precision={prec:.3f}  Recall={rec:.3f}  AUC={auc:.3f}  "
              f"(treino={train_end:,} | teste={test_end-test_start:,})")

    results = {
        'avg_precision': float(np.mean(precisions)) if precisions else 0.0,
        'avg_recall':    float(np.mean(recalls))    if recalls    else 0.0,
        'avg_auc':       float(np.mean(aucs))       if aucs       else 0.0,
        'best_model':    best_model,
    }
    print(f"\n[WFV] Media: Precision={results['avg_precision']:.3f}  "
          f"Recall={results['avg_recall']:.3f}  AUC={results['avg_auc']:.3f}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Treino principal
# ─────────────────────────────────────────────────────────────────────────────
def train(pairs: list = None, candles_per_pair: int = None,
          extra_samples: pd.DataFrame = None):
    """
    Treina o modelo de entrada com LightGBM usando dados multi-ativo.

    pairs            : lista de pares; None = todos os 6
    candles_per_pair : ultimos N candles de cada par; None = TUDO (4-5 anos)
    extra_samples    : DataFrame opcional [FEATURES + target + sample_weight]
                       com trades reais do bot (loop de aprendizado, Fase 5.6)
    """
    load_dotenv()
    if pairs is None:
        pairs = ALL_PAIRS

    print("=" * 64)
    print("TRADER.AI V5 — Treino do Modelo de Entrada (LightGBM + MTF)")
    print(f"Pares: {pairs}")
    print(f"Candles por par: {'TODOS' if candles_per_pair is None else f'{candles_per_pair:,}'}")
    print(f"Alvo: TP {TARGET_TP_MULT}xATR antes de SL {TARGET_SL_MULT}xATR em {TARGET_HORIZON} candles")
    print("=" * 64)

    frames = []
    for sym in pairs:
        path = os.path.join("data", f"{sym}_1m.parquet")
        if not os.path.exists(path):
            print(f"[TRAIN] AVISO: {path} nao encontrado. Pulando.")
            continue
        print(f"\n[TRAIN] Carregando {sym}...")
        raw = pd.read_parquet(path)
        if candles_per_pair is not None:
            raw = raw.tail(candles_per_pair).copy()
        print(f"[TRAIN] Feature engineering em {sym} ({len(raw):,} candles)...")
        feat = build_features(raw)
        if feat.empty:
            print(f"[TRAIN] AVISO: {sym} gerou DataFrame vazio. Pulando.")
            continue
        win_pct = feat['target'].mean() * 100
        print(f"[TRAIN] {sym}: {len(feat):,} amostras validas | WIN={win_pct:.1f}%")
        frames.append(feat)
        del raw, feat
        gc.collect()

    if not frames:
        print("[TRAIN] ERRO: Nenhum dado disponivel para treino.")
        return

    df_model = pd.concat(frames, ignore_index=True)
    del frames
    gc.collect()

    # ── Pesos de amostra (trades reais valem mais — loop de aprendizado) ──
    sample_weight = np.ones(len(df_model), dtype=np.float32)
    if extra_samples is not None and not extra_samples.empty:
        ex = extra_samples.copy()
        w_ex = ex.pop('sample_weight').to_numpy(dtype=np.float32) if 'sample_weight' in ex.columns \
               else np.full(len(ex), 5.0, dtype=np.float32)
        for c in FEATURES:
            if c not in ex.columns:
                ex[c] = 0.0
        ex = ex[FEATURES + ['target']]
        df_model = pd.concat([df_model, ex], ignore_index=True)
        sample_weight = np.concatenate([sample_weight, w_ex])
        print(f"[TRAIN] + {len(ex):,} amostras reais de trades (peso medio {w_ex.mean():.1f})")

    print(f"\n[TRAIN] Dataset combinado: {len(df_model):,} amostras x {len(FEATURES)} features")
    dist = df_model['target'].value_counts(normalize=True) * 100
    print(f"[TRAIN] Distribuicao target: WIN={dist.get(1, 0):.1f}%  LOSS={dist.get(0, 0):.1f}%")

    # ── Walk-Forward Validation ───────────────────────────────────────────
    wfv = walk_forward_eval(df_model, n_folds=5)

    # ── Treino final ──────────────────────────────────────────────────────
    print("\n[TRAIN] Treinando modelo final...")
    X_all = df_model[FEATURES]
    y_all = df_model['target']

    split = int(len(df_model) * 0.90)
    X_tr, X_val = X_all.iloc[:split], X_all.iloc[split:]
    y_tr, y_val = y_all.iloc[:split], y_all.iloc[split:]
    w_tr        = sample_weight[:split]

    final_model = lgb.LGBMClassifier(**LGBM_PARAMS)
    final_model.fit(
        X_tr, y_tr,
        sample_weight=w_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(60, verbose=False),
                   lgb.log_evaluation(period=100)]
    )

    y_prob = np.asarray(final_model.predict_proba(X_val))[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    print("\n[TRAIN] --- Avaliacao Final (ultimos 10%) ---")
    print(f"Acuracia : {accuracy_score(y_val, y_pred):.4f}")
    print(f"Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_val, y_pred, zero_division=0):.4f}")
    print(f"AUC-ROC  : {roc_auc_score(y_val, y_prob):.4f}")
    print(classification_report(y_val, y_pred))

    # ── Importancia das features ──────────────────────────────────────────
    importances = sorted(zip(FEATURES, final_model.feature_importances_),
                         key=lambda x: -x[1])
    print("\n[TRAIN] Importancia das Features (gain):")
    max_imp = max(v for _, v in importances) or 1
    for feat, imp in importances:
        bar = "#" * int(imp / max_imp * 30)
        print(f"  {feat:<14} {bar:<30} {imp:.0f}")

    # ── Salva o modelo ────────────────────────────────────────────────────
    model_path = os.path.join(os.path.dirname(__file__), "scalper_model.pkl")
    meta = {
        'model':         final_model,
        'features':      FEATURES,
        'wfv_precision': wfv['avg_precision'],
        'wfv_auc':       wfv['avg_auc'],
        'pairs_trained': pairs,
        'target':        {'tp_mult': TARGET_TP_MULT, 'sl_mult': TARGET_SL_MULT,
                          'horizon': TARGET_HORIZON},
    }
    joblib.dump(meta, model_path)
    print(f"\n[TRAIN] Modelo salvo em: {model_path}")
    print(f"[TRAIN] WFV Precision={wfv['avg_precision']:.4f} | AUC={wfv['avg_auc']:.4f}")
    print("[TRAIN] Concluido!")
    return meta


# Alias publico para importacao pelo retrain_scheduler.py
def train_model(pairs: Optional[list] = None, candles_per_pair: Optional[int] = None,
                extra_samples: Optional[pd.DataFrame] = None) -> dict:
    """Alias de train() com assinatura padronizada para o retrain_scheduler."""
    return train(pairs=pairs, candles_per_pair=candles_per_pair,
                 extra_samples=extra_samples) or {}


if __name__ == "__main__":
    # Treino completo: TODO o historico de todos os pares
    train(candles_per_pair=None)
