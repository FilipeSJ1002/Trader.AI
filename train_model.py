import os
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from dotenv import load_dotenv

def create_features_and_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza o Feature Engineering e cria a label do modelo.
    """
    df = df.copy()
    
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
            
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_2'] = df['close'].pct_change(2)
    df['ret_3'] = df['close'].pct_change(3)
    
    cols = df.columns
    try:
        rsi_col = [c for c in cols if c.startswith('RSI_')][0]
        macd_hist_col = [c for c in cols if c.startswith('MACDh_')][0]
        bbu_col = [c for c in cols if c.startswith('BBU_')][0]
        bbl_col = [c for c in cols if c.startswith('BBL_')][0]
        atr_col = [c for c in cols if c.startswith('ATRr_')][0]
        
        df['RSI'] = df[rsi_col]
        df['MACDh'] = df[macd_hist_col]
        df['ATR'] = df[atr_col]
        df['Dist_BBU'] = (df['close'] - df[bbu_col]) / df['close']
        df['Dist_BBL'] = (df['close'] - df[bbl_col]) / df['close']
    except Exception as e:
        print(f"Erro ao mapear colunas do pandas_ta: {e}")
        return pd.DataFrame()

    features = ['RSI', 'MACDh', 'ATR', 'Dist_BBU', 'Dist_BBL', 'ret_1', 'ret_2', 'ret_3']
    
    df['future_high_5'] = df['high'].shift(-5).rolling(window=5, min_periods=1).max()
    
    df['target'] = (df['future_high_5'] >= (df['close'] * 1.003)).astype(int)
    
    cols_to_keep = features + ['target']
    df_clean = df.loc[:, cols_to_keep].dropna()
    
    return df_clean

def train():
    load_dotenv()
    file_path = os.getenv("DATA_FILE_PATH", "data/BTCUSDT_1m.parquet")
    
    print(f"🔄 Carregando dados de {file_path}...")
    try:
        df = pd.read_parquet(file_path)
        df = df.tail(200000).copy()
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {file_path}. Verifique o diretório 'data/'.")
        return

    print("Realizando Feature Engineering...")
    df_model = create_features_and_target(df)
    
    if df_model.empty:
        print("❌ DataFrame vazio após feature engineering.")
        return

    print(f"Distribuição do Target:\n{df_model['target'].value_counts(normalize=True)*100}")

    features = ['RSI', 'MACDh', 'ATR', 'Dist_BBU', 'Dist_BBL', 'ret_1', 'ret_2', 'ret_3']
    X = df_model[features]
    y = df_model['target']

    split_idx = int(len(df_model) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Treino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

    print("Treinando RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, class_weight='balanced')
    model.fit(X_train, y_train)

    print("Avaliando o modelo no conjunto de Teste...")
    y_pred = model.predict(X_test)
    print("Acurácia:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    model_path = os.path.join(os.path.dirname(__file__), "scalper_model.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Modelo salvo com sucesso em: {model_path}")

if __name__ == "__main__":
    train()
