import pandas as pd
from mock_data import generate_mock_data
from strategy import TechnicalAnalysis
from main import PriceInput, analisar_mercado, startup_event
import asyncio
import sys

# Test Mock Data
print("--- [VERIFY] Mock Data ---")
df = generate_mock_data(n_candles=300)
print(f"Candles generated: {len(df)}")
if len(df) < 200:
    print("FAIL: Not enough candles")
    sys.exit(1)
if 'close' not in df.columns:
    print("FAIL: Missing 'close' column")
    sys.exit(1)
print("PASS: Mock Data generation")

# Test Strategy
print("\n--- [VERIFY] Strategy ---")
ta_engine = TechnicalAnalysis()
try:
    result = ta_engine.analisar_compra_venda(df.copy())
    print("Decision:", result['decision'])
    print("Analysis Keys:", result['analysis'].keys())
    
    analysis = result['analysis']
    if 'rsi' not in analysis or 'macd_signal' not in analysis:
         print("FAIL: Missing analysis keys")
         sys.exit(1)
         
    print("PASS: Strategy execution")
except Exception as e:
    print(f"FAIL: Strategy error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test API Flow (Simulated)
print("\n--- [VERIFY] API Flow ---")
async def test_api():
    await startup_event()
    input_data = PriceInput(price=66000.00, volume=1500)
    try:
        response = await analisar_mercado(input_data)
        print("API Response:", response)
        if "decision" not in response:
            print("FAIL: Invalid API response")
            sys.exit(1)
        print("PASS: API Flow")
    except Exception as e:
        print(f"FAIL: API Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(test_api())
