import io
import os
import sys
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Fix encoding Windows CP1252
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def testar_saldo_testnet():
    load_dotenv()

    api_key    = os.getenv("BINANCE_API_KEY")
    secret_key = os.getenv("BINANCE_SECRET_KEY")

    if not api_key or not secret_key:
        print("Erro: BINANCE_API_KEY ou BINANCE_SECRET_KEY nao encontradas no .env")
        return

    try:
        client = Client(api_key, secret_key, testnet=True,
                        requests_params={'timeout': 15})

        # Ativos que o Trader.AI pode operar — sempre exibidos, mesmo com saldo 0
        ATIVOS_BOT = ['USDT', 'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'AVAX']

        # 1. Busca conta e monta dicionario de saldos
        print("Buscando saldos...")
        raw = client.get_account().get('balances', [])
        saldos = {b['asset']: float(b['free']) + float(b['locked']) for b in raw}

        # 2. Busca TODOS os precos em uma unica chamada (evita travar)
        print("Buscando precos...")
        tickers = {t['symbol']: float(t['price'])
                   for t in client.get_all_tickers()}

        # 3. Monta tabela com todos os ativos do bot, independente do saldo
        rows = []
        for asset in ATIVOS_BOT:
            qty   = saldos.get(asset, 0.0)
            price = 1.0 if asset == 'USDT' else tickers.get(f"{asset}USDT", 0.0)
            value = qty * price
            rows.append((asset, qty, price, value))

        # 4. Exibe
        print(f"\n{'=' * 62}")
        print(f"  {'Ativo':<8}  {'Quantidade':>16}  {'Preco':>16}  {'Valor USDT':>12}")
        print(f"  {'-' * 8}  {'-' * 16}  {'-' * 16}  {'-' * 12}")

        total = 0.0
        for asset, qty, price, value in rows:
            p_str = f"$ {price:>10,.4f}" if price > 0 else "   sem preco"
            print(f"  {asset:<8}  {qty:>16.4f}  {p_str}  ${value:>11,.2f}")
            total += value

        print(f"  {'=' * 58}")
        print(f"  {'TOTAL DO PORTFOLIO':<40}  ${total:>11,.2f}")
        print(f"  {'=' * 58}\n")

    except BinanceAPIException as e:
        print(f"\n[ERRO DE API] {e.status_code}: {e.message}")
    except Exception as e:
        print(f"\n[ERRO] {e}")


if __name__ == "__main__":
    testar_saldo_testnet()
