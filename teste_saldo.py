import os
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

def testar_saldo_testnet():
    load_dotenv()

    api_key = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY")

    if not api_key or not secret_key:
        print("Erro: Chaves de API não encontradas no arquivo .env.")
        print("Certifique-se de ter BINANCE_API_KEY e BINANCE_SECRET_KEY configuradas.")
        return

    print("Iniciando conexão com a Binance Testnet...")
    
    try:
        client = Client(api_key, secret_key, testnet=True)
        
        account = client.get_account()
        balances = account.get('balances', [])
        
        print("\n=== SALDOS DISPONÍVEIS (Testnet) ===")
        ativos_encontrados = False
        
        for balance in balances:
            asset = balance['asset']
            if asset in ['BTC', 'USDT']:
                free_balance = float(balance['free'])
                locked_balance = float(balance['locked'])
                total_balance = free_balance + locked_balance
                
                if total_balance > 0:
                    ativos_encontrados = True
                    print(f"{asset}:")
                    print(f"   ► Livre:  {free_balance:.8f}")
                    print(f"   ► Bloq:   {locked_balance:.8f}")
                    print(f"   ► Total:  {total_balance:.8f}\n")
                    
        if not ativos_encontrados:
            print("   Nenhum saldo maior que zero encontrado para BTC ou USDT.")
            
        print("====================================\n")
            
    except BinanceAPIException as e:
        print("\n❌ [ERRO DE AUTENTICAÇÃO / API]")
        print(f"Código do erro: {e.status_code}")
        print(f"Mensagem: {e.message}")
        print("Verifique se as chaves da Testnet no arquivo .env estão corretas.")
    except Exception as e:
        print(f"\n❌ [ERRO INESPERADO] {str(e)}")

if __name__ == "__main__":
    testar_saldo_testnet()
