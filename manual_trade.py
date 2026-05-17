import os
import sys
import math
import argparse
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

def _arredondar_fracao(quantidade: float, decimais: int = 5) -> float:
    """Arredonda a quantidade de BTC para evitar erros de LOT_SIZE e precisão da Binance."""
    fator = 10 ** decimais
    return math.floor(quantidade * fator) / fator

def main():
    parser = argparse.ArgumentParser(description="Script para execução manual de ordens no Trader.AI")
    parser.add_argument('action', choices=['buy', 'sell'], help="Ação a ser executada: 'buy' (Compra Total) ou 'sell' (Venda Total)")
    
    args = parser.parse_args()
    
    # Carrega chaves de API
    load_dotenv()
    API_KEY = os.getenv("BINANCE_API_KEY")
    SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")
    
    if not API_KEY or not SECRET_KEY:
        print("❌ Erro: Chaves de API ausentes no arquivo .env!")
        sys.exit(1)
        
    print("⏳ Conectando à Binance Testnet...")
    try:
        client = Client(API_KEY, SECRET_KEY, testnet=True)
        # Testar conexão puxando info da conta
        client.get_account()
    except Exception as e:
        print(f"❌ Erro ao conectar na Binance Testnet: {e}")
        sys.exit(1)
        
    symbol = 'BTCUSDT'
    
    try:
        if args.action == 'buy':
            print("\n🟢 Iniciando COMPRA TOTAL...")
            usdt_balance_info = client.get_asset_balance(asset='USDT')
            usdt_balance = float(usdt_balance_info['free']) if usdt_balance_info else 0.0
            
            if usdt_balance < 11.0:
                print(f"❌ Saldo insuficiente de USDT ({usdt_balance:.2f}). Mínimo necessário é ~11.00 USDT.")
                sys.exit(1)
                
            # Compra total: usar 95% do saldo para evitar erro de saldo insuficiente na hora de pagar taxas
            usdt_utilizado = usdt_balance * 0.95
            
            ticker = client.get_ticker(symbol=symbol)
            current_price = float(ticker['lastPrice'])
            
            qty_btc = usdt_utilizado / current_price
            qty_btc = _arredondar_fracao(qty_btc, 5)
            
            print(f"Preço Atual do BTC: ${current_price:.2f}")
            print(f"Saldo USDT Total:   {usdt_balance:.2f} USDT")
            print(f"USDT Utilizado:     {usdt_utilizado:.2f} USDT")
            print(f"Quantidade de BTC:  {qty_btc} BTC")
            
            confirm = input("\n❓ Deseja confirmar a ordem de COMPRA A MERCADO? (s/n): ")
            if confirm.lower() != 's':
                print("🛑 Ordem cancelada pelo usuário.")
                sys.exit(0)
                
            order = client.order_market_buy(symbol=symbol, quantity=qty_btc)
            print(f"✅ Compra Executada com Sucesso! Order ID: {order.get('orderId')}")
            
        elif args.action == 'sell':
            print("\n🔴 Iniciando VENDA TOTAL...")
            btc_balance_info = client.get_asset_balance(asset='BTC')
            btc_balance = float(btc_balance_info['free']) if btc_balance_info else 0.0
            
            btc_qty = _arredondar_fracao(btc_balance, 5)
            
            if btc_qty <= 0:
                print(f"❌ Saldo de BTC zerado ou muito baixo: {btc_balance:.5f}")
                sys.exit(1)
                
            ticker = client.get_ticker(symbol=symbol)
            current_price = float(ticker['lastPrice'])
            
            print(f"Preço Atual do BTC: ${current_price:.2f}")
            print(f"Saldo disponível:   {btc_qty} BTC")
            print(f"Valor Aproximado:   ${(btc_qty * current_price):.2f} USDT")
            
            confirm = input("\n❓ Deseja confirmar a ordem de VENDA A MERCADO de 100% dos BTCs? (s/n): ")
            if confirm.lower() != 's':
                print("🛑 Ordem cancelada pelo usuário.")
                sys.exit(0)
                
            order = client.order_market_sell(symbol=symbol, quantity=btc_qty)
            print(f"✅ Venda Executada com Sucesso! Order ID: {order.get('orderId')}")
            
    except BinanceAPIException as e:
        print(f"❌ Erro da API da Binance: {e.message}")
    except Exception as e:
        print(f"❌ Erro inesperado: {str(e)}")

if __name__ == "__main__":
    main()
