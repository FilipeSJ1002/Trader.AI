import pandas as pd
import os

def load_historical_data(file_path: str = None, n_candles: int = 10000) -> pd.DataFrame:
    """
    Carrega dados históricos de um arquivo Parquet.
    
    Args:
        file_path: Caminho para o arquivo parquet. Se None, busca em DATA_FILE_PATH.
        n_candles: Número de candles recentes para manter em memória (otimização).
        
    Returns:
        pd.DataFrame: DataFrame com os últimos n_candles, indexado por data.
    """
    if file_path is None:
        file_path = os.getenv("DATA_FILE_PATH")

    print(f"Carregando dados de {file_path}...")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de dados não encontrado: {file_path}")
        
    df = pd.read_parquet(file_path)
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
    
    if len(df) > n_candles:
        df = df.tail(n_candles)
        
    print(f"Dados carregados com sucesso! {len(df)} candles em memória.")
    return df
