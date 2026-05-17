import os
import math
import logging
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
API_KEY = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Chaves de API ausentes no .env!")

try:
    client = Client(API_KEY, SECRET_KEY, testnet=True)
except Exception as e:
    logger.error(f"Erro ao conectar na Binance Testnet: {e}")
    client = None

POSITION_STATE = {
    "qty": 0.0, 
    "avg_price": 0.0,
    "highest_price": 0.0,
    "sl_price": 0.0,
    "last_buy_time": None,
    "atr_distance": 0.0,
    "tp_price": 0.0,
    "partial_tp_hit": False
}
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
        btc_balance_info = client.get_asset_balance(asset='BTC')
        btc_balance = float(btc_balance_info['free']) if btc_balance_info else 0.0
        if btc_balance > 0.001:
            POSITION_STATE["qty"] = btc_balance
            
            trades = client.get_my_trades(symbol=SYMBOL, limit=5)
            buyer_trades = [t for t in trades if t['isBuyer']]
            
            import pandas as pd
            if buyer_trades:
                last_buy_price = float(buyer_trades[-1]['price'])
                POSITION_STATE["avg_price"] = last_buy_price
                POSITION_STATE["highest_price"] = last_buy_price
                POSITION_STATE["atr_distance"] = last_buy_price * 0.015 # Fallback (1.5%) se reiniciado
                POSITION_STATE["sl_price"] = last_buy_price - POSITION_STATE["atr_distance"]
                POSITION_STATE["tp_price"] = last_buy_price + POSITION_STATE["atr_distance"]
                POSITION_STATE["partial_tp_hit"] = False
                POSITION_STATE["last_buy_time"] = pd.to_datetime(buyer_trades[-1]['time'], unit='ms')
                logger.info(f"[SYNC] Posição aberta detectada. Saldo BTC: {btc_balance:.5f} | Preço Médio Recuperado: ${last_buy_price:.2f}")
            else:
                current_market_price = float(client.get_ticker(symbol=SYMBOL)['lastPrice'])
                POSITION_STATE["avg_price"] = current_market_price
                POSITION_STATE["highest_price"] = current_market_price
                POSITION_STATE["atr_distance"] = current_market_price * 0.015
                POSITION_STATE["sl_price"] = current_market_price - POSITION_STATE["atr_distance"]
                POSITION_STATE["tp_price"] = current_market_price + POSITION_STATE["atr_distance"]
                POSITION_STATE["partial_tp_hit"] = False
                from datetime import datetime
                POSITION_STATE["last_buy_time"] = datetime.now()
                logger.warning(f"[SYNC] Posição detectada, mas sem histórico de compra recente. Preço médio ajustado para o mercado atual: ${current_market_price:.2f}")
        else:
            POSITION_STATE["qty"] = 0.0
            POSITION_STATE["avg_price"] = 0.0
            POSITION_STATE["highest_price"] = 0.0
            POSITION_STATE["sl_price"] = 0.0
            POSITION_STATE["last_buy_time"] = None
            POSITION_STATE["atr_distance"] = 0.0
            POSITION_STATE["tp_price"] = 0.0
            POSITION_STATE["partial_tp_hit"] = False
            logger.info(f"[SYNC] Nenhuma posição relevante detectada. Saldo BTC: {btc_balance:.5f}")
    except BinanceAPIException as e:
        logger.error(f"[SYNC ERRO] Falha ao sincronizar posição: {e.message}")
    except Exception as e:
        logger.error(f"[SYNC ERRO INESPERADO] {str(e)}")

sync_position_state()

def _arredondar_fracao(quantidade: float, decimais: int = 5) -> float:
    """Arredonda a quantidade de BTC para evitar erros de LOT_SIZE e precisão da Binance."""
    fator = 10 ** decimais
    return math.floor(quantidade * fator) / fator

async def execute_trade(decision: str, current_price: float, peso: float = 1.0, atr: float = 0.0):
    """
    Motor de execução de ordens a mercado baseado na decisão da estratégia.
    """
    global POSITION_STATE
    
    if not client:
        logger.error("[EXECUTION ERRO] Cliente Binance não inicializado.")
        return

    try:
        if decision.startswith("COMPRA_"):
            logger.info(f"[EXECUTION] Iniciando processo de {decision}... (Peso: {peso})")
            
            usdt_balance_info = client.get_asset_balance(asset='USDT')
            usdt_balance = float(usdt_balance_info['free']) if usdt_balance_info else 0.0
            
            # Fracionamento de Capital: MÁXIMO 30% do saldo total
            base_allocation = 0.30
            usdt_utilizado = usdt_balance * base_allocation * peso
            
            qty_btc = usdt_utilizado / current_price
            qty_btc = _arredondar_fracao(qty_btc, 5)
            
            if usdt_utilizado < 11.0 or qty_btc <= 0:
                logger.warning(f"[EXECUTION] Compra ignorada: Saldo de USDT alocado ({usdt_utilizado:.2f}) é menor que o lote mínimo exigido pela Binance (~11.00 USDT).")
                return

            order = client.order_market_buy(
                symbol=SYMBOL,
                quantity=qty_btc
            )
            
            qtd_antiga = POSITION_STATE["qty"]
            preco_antigo = POSITION_STATE["avg_price"]
            
            if qtd_antiga > 0:
                novo_preco_medio = ((preco_antigo * qtd_antiga) + (current_price * qty_btc)) / (qtd_antiga + qty_btc)
            else:
                novo_preco_medio = current_price
                
            from datetime import datetime
            POSITION_STATE["avg_price"] = novo_preco_medio
            POSITION_STATE["qty"] += qty_btc
            POSITION_STATE["highest_price"] = current_price
            
            # Stop Loss e Take Profit baseados no ATR
            distancia_atr = 2 * atr if atr > 0 else (current_price * 0.015)
            POSITION_STATE["atr_distance"] = distancia_atr
            POSITION_STATE["sl_price"] = current_price - distancia_atr
            POSITION_STATE["tp_price"] = current_price + distancia_atr
            POSITION_STATE["partial_tp_hit"] = False
            
            POSITION_STATE["last_buy_time"] = datetime.now()
            
            print(f"[EXECUTION] 🟢 COMPRA EXECUTADA ({decision}) | Quantidade: {qty_btc} BTC | Novo Preço Médio Ponderado: ${POSITION_STATE['avg_price']:.2f}")
            logger.info(f"Comprou com Sucesso! Order ID: {order.get('orderId')} | Status: {order.get('status')}")

        elif decision in ["VENDA_FORTE", "VENDA_STOP", "VENDA_TIME_STOP", "VENDA_PARCIAL"]:
            if POSITION_STATE["qty"] <= 0:
                logger.info(f"[EXECUTION] Venda ignorada: Sinal {decision} detectado, mas não há saldo.")
                return
                
            logger.info(f"[EXECUTION] Iniciando processo de {decision}...")
            
            if decision == "VENDA_FORTE":
                preco_minimo_venda = POSITION_STATE["avg_price"] * 1.0025
                if current_price < preco_minimo_venda:
                    logger.warning(f"[EXECUTION] Venda Bloqueada: Preço atual (${current_price:.2f}) não cobre o Preço Médio (${POSITION_STATE['avg_price']:.2f}) + Taxas.")
                    return
            else:
                logger.warning(f"[EXECUTION] Venda de Emergência acionada ({decision}). Ignorando lucro mínimo.")
            
            btc_balance_info = client.get_asset_balance(asset='BTC')
            btc_balance = float(btc_balance_info['free']) if btc_balance_info else 0.0
            
            if decision == "VENDA_PARCIAL":
                # Vende 50% da posição
                btc_qty = POSITION_STATE["qty"] / 2
                btc_qty = _arredondar_fracao(btc_qty, 5)
            else:
                btc_qty = _arredondar_fracao(btc_balance, 5)
            
            if btc_qty <= 0:
                 logger.warning(f"[EXECUTION] Saldo de BTC muito baixo ou insuficiente para venda. Saldo BTC: {btc_balance:.5f}")
                 POSITION_STATE["qty"] = 0.0
                 POSITION_STATE["avg_price"] = 0.0
                 return
            
            order = client.order_market_sell(
                symbol=SYMBOL,
                quantity=btc_qty
            )
            
            if decision == "VENDA_PARCIAL":
                POSITION_STATE["qty"] -= btc_qty
            else:
                POSITION_STATE["qty"] = 0.0
                POSITION_STATE["avg_price"] = 0.0
                POSITION_STATE["highest_price"] = 0.0
                POSITION_STATE["sl_price"] = 0.0
                POSITION_STATE["last_buy_time"] = None
                POSITION_STATE["atr_distance"] = 0.0
                POSITION_STATE["tp_price"] = 0.0
                POSITION_STATE["partial_tp_hit"] = False
            
            print(f"[EXECUTION] 🔴 VENDA EXECUTADA | Quantidade: {btc_qty} BTC | Preço de Venda: ${current_price:.2f}")
            logger.info(f"Vendeu com Sucesso! Order ID: {order.get('orderId')} | Status: {order.get('status')}")

    except BinanceAPIException as e:
        logger.error(f"[EXECUTION API ERRO] Código: {e.status_code} | Mensagem: {e.message}")
        print(f"❌ [API EXCEPTION] Falha ao executar a ordem na Binance: {e.message}")
    except Exception as e:
        logger.error(f"[EXECUTION ERRO INESPERADO] {str(e)}")
        print(f"❌ [ERRO INESPERADO] Falha inesperada no roteamento de ordem: {str(e)}")

async def check_risk_management(current_price: float, current_time):
    """
    Avalia constantemente a posição aberta para gerenciar o Stop Loss,
    Trailing Stop e Time Stop. Se atingir os limites de segurança,
    dispara uma ordem de venda a mercado.
    """
    global POSITION_STATE
    
    if POSITION_STATE["qty"] <= 0:
        return
        
    # 1. Atualiza Trailing Stop
    if current_price > POSITION_STATE["highest_price"]:
        POSITION_STATE["highest_price"] = current_price
        
    # Recalcula o limite do trailing stop mantendo a distância do ATR
    novo_sl = POSITION_STATE["highest_price"] - POSITION_STATE["atr_distance"]
    if novo_sl > POSITION_STATE["sl_price"]:
        POSITION_STATE["sl_price"] = novo_sl
        
    # 2. Hard Stop / Trailing Stop trigger
    if current_price <= POSITION_STATE["sl_price"]:
        logger.warning(f"[RISK] 🚨 Preço atingiu Stop Loss/Trailing Stop! Preço: ${current_price:.2f} <= SL: ${POSITION_STATE['sl_price']:.2f}")
        await execute_trade("VENDA_STOP", current_price, 1.0)
        return
        
    # 2.5 Scale-out (Partial TP)
    if current_price >= POSITION_STATE["tp_price"] and not POSITION_STATE["partial_tp_hit"]:
        logger.info(f"[RISK] 💰 Scale-out ativado! Preço alvo atingido (${POSITION_STATE['tp_price']:.2f}). Realizando lucro de 50%.")
        POSITION_STATE["partial_tp_hit"] = True
        await execute_trade("VENDA_PARCIAL", current_price, 1.0)
        
    # 3. Time Stop Trigger
    if POSITION_STATE["last_buy_time"] is not None:
        import pandas as pd
        if not isinstance(current_time, pd.Timestamp):
            current_time = pd.to_datetime(current_time)
            
        time_diff = current_time - POSITION_STATE["last_buy_time"]
        if time_diff.days >= 3:
            logger.warning(f"[RISK] ⏰ Time Stop atingido! Posição aberta há {time_diff.days} dias.")
            await execute_trade("VENDA_TIME_STOP", current_price, 1.0)
            return
