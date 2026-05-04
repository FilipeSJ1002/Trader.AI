import pandas as pd
from data_loader import load_historical_data
from strategy import TechnicalAnalysis
import os
from datetime import datetime

def test_data_loader():
    print("\n--- Testing Data Loader ---")
    try:
        df = load_historical_data()
        print(f"Success! Loaded {len(df)} rows.")
        print(f"Index type: {type(df.index)}")
        print(f"Columns: {df.columns.tolist()}")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            print("ERROR: Index is not DatetimeIndex")
            return None
            
        if len(df) > 10001:
            print(f"WARNING: DataFrame has {len(df)} rows, expected ~10000")
            
        return df
    except Exception as e:
        print(f"FAILED: {e}")
        return None

def test_strategy(df):
    print("\n--- Testing Strategy Compatibility ---")
    if df is None:
        print("Skipping strategy test due to loader failure.")
        return

    strategy = TechnicalAnalysis()
    try:
        result = strategy.analisar_compra_venda(df.copy())
        print("Success! Strategy executed without error.")
        print(f"Decision: {result['decision']}")
        print(f"Analysis: {result['analysis']}")
    except Exception as e:
        print(f"FAILED: Strategy execution error: {e}")
        import traceback
        traceback.print_exc()

def main():
    if not os.path.exists("data/BTCUSDT_real_1m_2022_2025.parquet"):
        print("WARNING: Data file not found at data/BTCUSDT_real_1m_2022_2025.parquet")
        return

    df = test_data_loader()
    test_strategy(df)

if __name__ == "__main__":
    main()
