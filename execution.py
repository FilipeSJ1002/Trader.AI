import os
import math
import logging
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuração inicial
load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Chaves de API ausentes no .env!")

# Inicialização do Cliente Binance na Testnet
try:
    client = Client(API_KEY, SECRET_KEY, testnet=True)
except Exception as e:
    logger.error(f"Erro ao conectar na Binance Testnet: {e}")
    client = None

# Gerenciamento de Estado de Conta
POSITION_STATE = {"qty": 0.0, "avg_price": 0.0}
SYMBOL = 'BTCUSDT'

def sync_position_state():
    """
    Função de sincronização chamada ao inicializar o módulo.
    Verifica o saldo real de BTC na conta e atualiza POSITION_STATE.
    """
    global POSITION_STATE
    if not client:
        return

    try:
        btc_balance = float(client.get_asset_balance(asset='BTC')['free'])
        if btc_balance > 0.001:
            POSITION_STATE["qty"] = btc_balance
            
            # Buscar avg_price de trades recentes
            trades = client.get_my_trades(symbol=SYMBOL, limit=5)
            buyer_trades = [t for t in trades if t['isBuyer']]
            
            if buyer_trades:
                last_buy_price = float(buyer_trades[-1]['price'])
                POSITION_STATE["avg_price"] = last_buy_price
                logger.info(f"[SYNC] Posição aberta detectada. Saldo BTC: {btc_balance:.5f} | Preço Médio Recuperado: ${last_buy_price:.2f}")
            else:
                current_market_price = float(client.get_ticker(symbol=SYMBOL)['lastPrice'])
                POSITION_STATE["avg_price"] = current_market_price
                logger.warning(f"[SYNC] Posição detectada, mas sem histórico de compra recente. Preço médio ajustado para o mercado atual: ${current_market_price:.2f}")
        else:
            POSITION_STATE["qty"] = 0.0
            POSITION_STATE["avg_price"] = 0.0
            logger.info(f"[SYNC] Nenhuma posição relevante detectada. Saldo BTC: {btc_balance:.5f}")
    except BinanceAPIException as e:
        logger.error(f"[SYNC ERRO] Falha ao sincronizar posição: {e.message}")
    except Exception as e:
        logger.error(f"[SYNC ERRO INESPERADO] {str(e)}")

# Executa sincronização inicial no momento em que o módulo é importado/executado
sync_position_state()

def _arredondar_fracao(quantidade: float, decimais: int = 5) -> float:
    """Arredonda a quantidade de BTC para evitar erros de LOT_SIZE e precisão da Binance."""
    fator = 10 ** decimais
    return math.floor(quantidade * fator) / fator

async def execute_trade(decision: str, current_price: float, peso: float = 1.0):
    """
    Motor de execução de ordens a mercado baseado na decisão da estratégia.
    """
    global POSITION_STATE
    
    if not client:
        logger.error("[EXECUTION ERRO] Cliente Binance não inicializado.")
        return

    try:
        # Lógica de COMPRA
        if decision.startswith("COMPRA_"):
            logger.info(f"[EXECUTION] Iniciando processo de {decision}... (Peso: {peso})")
            
            # Consulta saldo de USDT
            usdt_balance = float(client.get_asset_balance(asset='USDT')['free'])
            
            # Utiliza saldo com base no peso para escalar as compras fracionadas
            usdt_utilizado = (usdt_balance * 0.90) * peso
            
            # Calcula a quantidade de BTC com base no preço atual
            qty_btc = usdt_utilizado / current_price
            qty_btc = _arredondar_fracao(qty_btc, 5) # Arredonda para 5 casas decimais (LOT_SIZE seguro para BTC)
            
            # Proteção contra ordens de tamanho insignificante e lote mínimo da Binance (~11 USDT)
            if usdt_utilizado < 11.0 or qty_btc <= 0:
                logger.warning(f"[EXECUTION] Compra ignorada: Saldo de USDT alocado ({usdt_utilizado:.2f}) é menor que o lote mínimo exigido pela Binance (~11.00 USDT).")
                return

            # Envia Ordem de Compra a Mercado
            order = client.order_market_buy(
                symbol=SYMBOL,
                quantity=qty_btc
            )
            
            # Atualiza o estado da Carteira com a fórmula do Preço Médio Ponderado
            qtd_antiga = POSITION_STATE["qty"]
            preco_antigo = POSITION_STATE["avg_price"]
            
            if qtd_antiga > 0:
                novo_preco_medio = ((preco_antigo * qtd_antiga) + (current_price * qty_btc)) / (qtd_antiga + qty_btc)
            else:
                novo_preco_medio = current_price
                
            POSITION_STATE["avg_price"] = novo_preco_medio
            POSITION_STATE["qty"] += qty_btc
            
            print(f"[EXECUTION] 🟢 COMPRA EXECUTADA ({decision}) | Quantidade: {qty_btc} BTC | Novo Preço Médio Ponderado: ${POSITION_STATE['avg_price']:.2f}")
            logger.info(f"Comprou com Sucesso! Order ID: {order.get('orderId')} | Status: {order.get('status')}")

        # Lógica de VENDA
        elif decision == "VENDA_FORTE":
            if POSITION_STATE["qty"] <= 0:
                logger.info("[EXECUTION] Venda ignorada: Sinal de VENDA detectado, mas não há saldo de BTC em carteira.")
                return
                
            logger.info("[EXECUTION] Iniciando processo de VENDA...")
            
            # Break-Even Protector (Protetor de Taxas)
            preco_minimo_venda = POSITION_STATE["avg_price"] * 1.0025
            if current_price < preco_minimo_venda:
                logger.warning(f"[EXECUTION] Venda Bloqueada: Preço atual (${current_price:.2f}) não cobre o Preço Médio (${POSITION_STATE['avg_price']:.2f}) + Taxas.")
                return
            
            # Consulta saldo total de BTC
            btc_balance = float(client.get_asset_balance(asset='BTC')['free'])
            btc_qty = _arredondar_fracao(btc_balance, 5) # Vende 100% possível (LOT_SIZE)
            
            if btc_qty <= 0:
                 logger.warning(f"[EXECUTION] Saldo de BTC muito baixo ou insuficiente para venda. Saldo BTC: {btc_balance:.5f}")
                 POSITION_STATE["qty"] = 0.0 # Ajusta o estado se estiver dessincronizado
                 POSITION_STATE["avg_price"] = 0.0
                 return
            
            # Envia Ordem de Venda a Mercado
            order = client.order_market_sell(
                symbol=SYMBOL,
                quantity=btc_qty
            )
            
            # Atualiza o estado
            POSITION_STATE["qty"] = 0.0
            POSITION_STATE["avg_price"] = 0.0
            
            print(f"[EXECUTION] 🔴 VENDA EXECUTADA | Quantidade: {btc_qty} BTC | Preço de Venda: ${current_price:.2f}")
            logger.info(f"Vendeu com Sucesso! Order ID: {order.get('orderId')} | Status: {order.get('status')}")

    except BinanceAPIException as e:
        logger.error(f"[EXECUTION API ERRO] Código: {e.status_code} | Mensagem: {e.message}")
        print(f"❌ [API EXCEPTION] Falha ao executar a ordem na Binance: {e.message}")
    except Exception as e:
        logger.error(f"[EXECUTION ERRO INESPERADO] {str(e)}")
        print(f"❌ [ERRO INESPERADO] Falha inesperada no roteamento de ordem: {str(e)}")
