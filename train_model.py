import os
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from dotenv import load_dotenv
import numpy as np

# ─────────────────────────────────────────────
# Lista canônica de features — usada em treino,
# strategy.py e backtest.py. Alterar aqui reflete
# em todo o sistema.
# ─────────────────────────────────────────────
FEATURES = [
    'RSI', 'MACDh', 'ATR_pct',
    'Dist_BBU', 'Dist_BBL',
    'ret_1', 'ret_2', 'ret_3',
    'OBV_pct',       # OBV normalizado: variação % do período anterior
    'VWAP_dist',     # VWAP normalizado: (close - VWAP) / VWAP
    'ROC_5', 'ROC_15',
    'Body_Size', 'Upper_Wick', 'Lower_Wick', 'Wick_Ratio',
    'CDL_ENGULFING', 'CDL_HAMMER', 'CDL_MORNINGSTAR',
    'Cup_and_Handle'
]


def create_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o Feature Engineering normalizado e cria a label do modelo.

    Mudanças da V4:
    - OBV agora é OBV_pct (variação % — escala independente do preço)
    - VWAP agora é VWAP_dist ((close-VWAP)/VWAP — desvio relativo)
    - ATR é sempre normalizado (ATR_pct = ATR/close)
    - Upper_Wick e Lower_Wick normalizados pelo close
    """
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df.set_index('date', inplace=True)

    # ── Indicadores técnicos ──────────────────────────────────────────
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.vwap(append=True)
    df.ta.roc(length=5, append=True)
    df.ta.roc(length=15, append=True)

    # ── Retornos passados ─────────────────────────────────────────────
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_2'] = df['close'].pct_change(2)
    df['ret_3'] = df['close'].pct_change(3)

    # ── Geometria dos candles (normalizados pelo close) ───────────────
    df['Body_Size']  = abs(df['close'] - df['open']) / df['close']
    df['Upper_Wick'] = (df['high'] - np.maximum(df['open'], df['close'])) / df['close']
    df['Lower_Wick'] = (np.minimum(df['open'], df['close']) - df['low']) / df['close']
    df['Wick_Ratio'] = df['Lower_Wick'] / (df['Upper_Wick'] + 0.00001)

    # ── Padrões de candle ─────────────────────────────────────────────
    cdl = df.ta.cdl_pattern(name=["engulfing", "hammer", "morningstar"])
    if cdl is not None and isinstance(cdl, pd.DataFrame):
        for col in ['CDL_ENGULFING', 'CDL_HAMMER', 'CDL_MORNINGSTAR']:
            if col in cdl.columns and col not in df.columns:
                df[col] = cdl[col]

    for col in ['CDL_ENGULFING', 'CDL_HAMMER', 'CDL_MORNINGSTAR']:
        if col not in df.columns:
            df[col] = 0
        else:
            df[col] = df[col].fillna(0).astype(int)

    # ── Padrão Cup and Handle ─────────────────────────────────────────
    H1 = df['high'].shift(5).rolling(window=35).max()
    L1 = df['low'].shift(5).rolling(window=35).min()
    H2 = df['high'].rolling(window=5).max()
    cup_depth       = (H1 - L1) / H1
    cup_edges_match = abs(H1 - H2) / H1
    handle_drop     = (H2 - df['close']) / H2
    cup_cond = (cup_depth > 0.015) & (cup_edges_match < 0.01) & (handle_drop > 0.002) & (handle_drop < 0.01)
    df['Cup_and_Handle'] = cup_cond.astype(int)

    # ── Mapeamento de colunas pandas_ta → features canônicas ─────────
    cols = df.columns
    try:
        rsi_col       = [c for c in cols if c.startswith('RSI_')][0]
        macd_hist_col = [c for c in cols if c.startswith('MACDh_')][0]
        bbu_col       = [c for c in cols if c.startswith('BBU_')][0]
        bbl_col       = [c for c in cols if c.startswith('BBL_')][0]
        atr_col       = [c for c in cols if c.startswith('ATRr_')][0]
        obv_col       = [c for c in cols if c.startswith('OBV')][0]
        vwap_col      = [c for c in cols if c.startswith('VWAP')][0]
        roc_5_col     = [c for c in cols if c.startswith('ROC_5')][0]
        roc_15_col    = [c for c in cols if c.startswith('ROC_15')][0]
    except IndexError as e:
        print(f"Erro ao mapear colunas do pandas_ta: {e}")
        return pd.DataFrame()

    df['RSI']      = df[rsi_col]
    df['MACDh']    = df[macd_hist_col]
    df['ATR_pct']  = df[atr_col] / df['close']          # normalizado
    df['Dist_BBU'] = (df['close'] - df[bbu_col]) / df['close']
    df['Dist_BBL'] = (df['close'] - df[bbl_col]) / df['close']
    df['ROC_5']    = df[roc_5_col]
    df['ROC_15']   = df[roc_15_col]

    # ── Features normalizadas (V4) ────────────────────────────────────
    # OBV_pct: variação percentual do OBV — escala-livre em relação ao preço
    df['OBV_pct']   = df[obv_col].pct_change(1).fillna(0).clip(-1, 1)

    # VWAP_dist: distância relativa entre o preço e o VWAP
    # Positivo = preço acima do VWAP (momentum), Negativo = abaixo (fraqueza)
    df['VWAP_dist'] = (df['close'] - df[vwap_col]) / df[vwap_col]

    # ── Label (target) ────────────────────────────────────────────────
    # O trade é "sucesso" se o high dos próximos 30 min subir pelo menos
    # 1.0×ATR a partir do close atual — critério mais alinhado com o TP real
    df['future_high_30'] = df['high'].shift(-30).rolling(window=30, min_periods=1).max()
    df['target'] = (df['future_high_30'] >= (df['close'] + df[atr_col])).astype(int)

    cols_to_keep = FEATURES + ['target']
    df_clean = df.loc[:, cols_to_keep].dropna()

    return df_clean


def train():
    load_dotenv()
    file_path = os.getenv("DATA_FILE_PATH", "data/BTCUSDT_1m.parquet")

    print(f"[TRAIN] Carregando dados de {file_path}...")
    try:
        df = pd.read_parquet(file_path)
        df = df.tail(200000).copy()
    except FileNotFoundError:
        print(f"[TRAIN] ERRO: Arquivo nao encontrado: {file_path}. Verifique o diretorio 'data/'.")
        return

    print("[TRAIN] Realizando Feature Engineering V4 (features normalizadas)...")
    df_model = create_features_and_target(df)

    if df_model.empty:
        print("[TRAIN] ERRO: DataFrame vazio apos feature engineering.")
        return

    dist = df_model['target'].value_counts(normalize=True) * 100
    print(f"[TRAIN] Distribuicao do Target:\n{dist}")

    X = df_model[FEATURES]
    y = df_model['target']

    split_idx = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"[TRAIN] Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    print("[TRAIN] Treinando RandomForestClassifier V4...")
    model = RandomForestClassifier(
        n_estimators=200,      # mais árvores = mais robusto
        max_depth=8,           # mais profundidade = captura padrões mais complexos
        min_samples_leaf=50,   # evita overfitting em folhas muito pequenas
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)

    print("[TRAIN] Avaliando o modelo no conjunto de Teste...")
    y_pred = model.predict(X_test)
    print("[TRAIN] Acuracia:", round(accuracy_score(y_test, y_pred), 4))
    print(classification_report(y_test, y_pred))

    # Importância das features
    importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
    print("\n[TRAIN] Importancia das Features:")
    for feat, imp in importances:
        print(f"  {feat:<20} {imp:.4f}")

    model_path = os.path.join(os.path.dirname(__file__), "scalper_model.pkl")
    joblib.dump(model, model_path)
    print(f"\n[TRAIN] Modelo V4 salvo com sucesso em: {model_path}")


if __name__ == "__main__":
    train()
