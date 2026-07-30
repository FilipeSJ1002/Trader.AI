import pandas as pd
from mock_data import generate_mock_data
from strategy import TechnicalAnalysis
from main import app
import sys

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

print("\n--- [VERIFY] Strategy ---")
ta_engine = TechnicalAnalysis()
try:
    result = ta_engine.analisar_compra_venda(df.copy())
    print("Decision:", result['decision'])
    print("Analysis Keys:", result['analysis'].keys())
    
    # Chaves conforme a V4 atual (a V1 devolvia 'macd_signal', que nao existe mais)
    analysis = result['analysis']
    faltando = [k for k in ('rsi', 'buy_score', 'sell_score', 'ml_status')
                if k not in analysis]
    if faltando:
         print(f"FAIL: Missing analysis keys: {faltando}")
         sys.exit(1)
         
    print("PASS: Strategy execution")
except Exception as e:
    print(f"FAIL: Strategy error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# O endpoint POST /analisar (com PriceInput) foi removido quando a API virou
# somente-leitura na V4. Aqui verificamos as rotas que existem hoje, sem
# depender de banco nem de rede.
print("\n--- [VERIFY] API Routes ---")
def test_api():
    esperadas = {"/status", "/trend", "/positions", "/performance", "/regime"}
    rotas = {getattr(r, "path", None) for r in app.routes}
    faltando = esperadas - rotas
    if faltando:
        print(f"FAIL: rotas ausentes na API: {sorted(faltando)}")
        sys.exit(1)
    print(f"PASS: API Routes ({len(esperadas)} rotas registradas)")

if __name__ == "__main__":
    test_api()
