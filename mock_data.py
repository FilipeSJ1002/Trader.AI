import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_data(n_candles=300, start_price=65000.0, volatility=0.02):
    """
    Gera um DataFrame mockado com dados OHLCV realistas.
    
    Args:
        n_candles (int): Número de candles a gerar.
        start_price (float): Preço inicial.
        volatility (float): Volatilidade dos movimentos.
        
    Returns:
        pd.DataFrame: DataFrame com colunas ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    
    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=n_candles*60)
    timestamps = [start_time + timedelta(minutes=i*60) for i in range(n_candles)]
    
    returns = np.random.normal(0, volatility/np.sqrt(n_candles), n_candles)
    price_path = start_price * np.exp(np.cumsum(returns))
    
    data = []
    current_price = start_price
    
    for i in range(n_candles):
        close_price = price_path[i]
        
        open_price = price_path[i-1] if i > 0 else start_price
        
        noise_high = np.abs(np.random.normal(0, volatility/5))
        noise_low = np.abs(np.random.normal(0, volatility/5))
        
        high_price = max(open_price, close_price) * (1 + noise_high)
        low_price = min(open_price, close_price) * (1 - noise_low)
        
        volume = int(np.random.lognormal(10, 1))
        
        data.append({
            'timestamp': timestamps[i],
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

if __name__ == "__main__":
    df = generate_mock_data()
    print(df.tail())
    print(f"Total candles: {len(df)}")
