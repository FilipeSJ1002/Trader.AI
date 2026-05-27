import os
import math
import logging
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client
from binance.exceptions import BinanceAPIException
import database

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

from binance_stream import ATIVOS  # fonte única da lista de ativos

# Precisão de quantidade por ativo (casas decimais para floor)
# Baseado nos filtros LOT_SIZE da Binance
_QTY_DECIMALS: dict[str, int] = {
    "BTCUSDT":  5,
    "ETHUSDT":  4,
    "XRPUSDT":  1,
    "SOLUSDT":  2,
    "BNBUSDT":  3,
    "AVAXUSDT": 2,
}

# Saldo mínimo relevante por ativo base
_MIN_BALANCES: dict[str, float] = {
    "BTC":  0.0001,
    "ETH":  0.001,
    "XRP":  1.0,
    "SOL":  0.01,
    "BNB":  0.001,
    "AVAX": 0.01,
}


def get_base_asset(symbol: str) -> str:
    return symbol.replace("USDT", "")


def _decimais_para(symbol: str) -> int:
    return _QTY_DECIMALS.get(symbol, 5)

def sync_position_state():
    """
    Sincroniza as posições reais da Binance com o banco de dados para os ativos operados.
    """
    if not client:
        return

    try:
        for symbol in ATIVOS:
            base_asset = get_base_asset(symbol)
            balance_info = client.get_asset_balance(asset=base_asset)
            balance = float(balance_info['free']) if balance_info else 0.0

            min_bal = _MIN_BALANCES.get(base_asset, 0.001)

            pos_db = database.get_position(symbol)

            if balance > min_bal:
                if pos_db["qty"] == 0:
                    trades = client.get_my_trades(symbol=symbol, limit=5)
                    buyer_trades = [t for t in trades if t['isBuyer']]
                    
                    if buyer_trades:
                        last_buy_price = float(buyer_trades[-1]['price'])
                        pos_db["avg_price"] = last_buy_price
                        pos_db["highest_price"] = last_buy_price
                        pos_db["atr_value"] = last_buy_price * 0.002
                        pos_db["sl_price"] = last_buy_price - (1.2 * pos_db["atr_value"])
                        pos_db["tp_price"] = last_buy_price + (3.0 * pos_db["atr_value"])
                        pos_db["partial_tp_hit"] = 0
                        pos_db["last_buy_time"] = str(pd.to_datetime(buyer_trades[-1]['time'], unit='ms'))
                        logger.info(f"[SYNC] Posição aberta {symbol}. Saldo: {balance} | Preço: ${last_buy_price:.2f}")
                    else:
                        current_price = float(client.get_ticker(symbol=symbol)['lastPrice'])
                        pos_db["avg_price"] = current_price
                        pos_db["highest_price"] = current_price
                        pos_db["atr_value"] = current_price * 0.002
                        pos_db["sl_price"] = current_price - (1.2 * pos_db["atr_value"])
                        pos_db["tp_price"] = current_price + (3.0 * pos_db["atr_value"])
                        pos_db["partial_tp_hit"] = 0
                        pos_db["last_buy_time"] = str(datetime.now())
                        logger.warning(f"[SYNC] {symbol} posição sem histórico. Preço ajustado: ${current_price:.2f}")
                
                pos_db["qty"] = balance
                database.update_position(pos_db)
            else:
                database.clear_position(symbol)
                logger.info(f"[SYNC] {symbol} Nenhuma posição relevante. Saldo: {balance}")
    except BinanceAPIException as e:
        logger.error(f"[SYNC ERRO] Falha ao sincronizar posição: {e.message}")
    except Exception as e:
        logger.error(f"[SYNC ERRO INESPERADO] {str(e)}")

sync_position_state()

def _arredondar_fracao(quantidade: float, decimais: int = 5) -> float:
    fator = 10 ** decimais
    return math.floor(quantidade * fator) / fator

def calculate_account_exposure() -> tuple[float, float, float]:
    """Calcula a exposição total da conta. Retorna (usdt_free, total_equity, exposure_ratio)."""
    if not client:
        return 0.0, 0.0, 0.0
    account_info = client.get_account()
    usdt_free = 0.0
    total_equity = 0.0
    for asset in account_info['balances']:
        free = float(asset['free'])
        locked = float(asset['locked'])
        total = free + locked
        if total > 0:
            if asset['asset'] == 'USDT':
                usdt_free = free
                total_equity += total
            else:
                try:
                    ticker = client.get_ticker(symbol=f"{asset['asset']}USDT")
                    price = float(ticker['lastPrice'])
                    total_equity += total * price
                except:
                    pass
    exposure_ratio = (total_equity - usdt_free) / total_equity if total_equity > 0 else 0.0
    return usdt_free, total_equity, exposure_ratio

async def execute_trade(symbol: str, decision: str, current_price: float, peso: float = 1.0, atr: float = 0.0):
    """
    Motor de execução de ordens a mercado para Multi-Asset.
    """
    if not client:
        logger.error("[EXECUTION ERRO] Cliente Binance não inicializado.")
        return

    pos_db = database.get_position(symbol)

    try:
        if decision.startswith("COMPRA_"):
            logger.info(f"[EXECUTION] {symbol} - Iniciando {decision}... (Peso: {peso})")
            
            usdt_free, total_equity, exposure_ratio = calculate_account_exposure()
            
            if exposure_ratio >= 0.95:
                logger.warning(f"[EXECUTION] Compra {symbol} bloqueada: Exposição da conta está em {exposure_ratio*100:.1f}%. Máximo permitido é 95%.")
                return

            # Max 90% do equity total per trade
            max_usdt_per_trade = total_equity * 0.90
            usdt_utilizado = min(usdt_free, max_usdt_per_trade) * peso
            
            # Garantir que sobra pelo menos 5% do equity em USDT
            usdt_after = usdt_free - usdt_utilizado
            if usdt_after < total_equity * 0.05:
                usdt_utilizado = usdt_free - (total_equity * 0.05)
                
            if usdt_utilizado <= 0:
                logger.warning(f"[EXECUTION] Compra {symbol} bloqueada: Sem margem USDT para manter os 5% livres.")
                return

            qty_coin = usdt_utilizado / current_price
            qty_coin = _arredondar_fracao(qty_coin, _decimais_para(symbol))
            
            if usdt_utilizado < 11.0 or qty_coin <= 0:
                logger.warning(f"[EXECUTION] {symbol} Compra ignorada: Lote muito pequeno ({usdt_utilizado:.2f} USDT).")
                return

            order = client.order_market_buy(
                symbol=symbol,
                quantity=qty_coin
            )
            
            qtd_antiga = pos_db["qty"]
            preco_antigo = pos_db["avg_price"]
            
            if qtd_antiga > 0:
                novo_preco_medio = ((preco_antigo * qtd_antiga) + (current_price * qty_coin)) / (qtd_antiga + qty_coin)
            else:
                novo_preco_medio = current_price
                
            pos_db["avg_price"] = novo_preco_medio
            pos_db["qty"] += qty_coin
            pos_db["highest_price"] = current_price
            
            # Stop Loss e Take Profit baseados em ATR (Risk/Reward 1:2.5)
            # SL em 1.2×ATR: espaço suficiente para o ruído normal do mercado de 1M
            # TP em 3.0×ATR: manter R:R favorável de ~2.5:1
            atr_val = atr if atr > 0 else (current_price * 0.002)
            pos_db["atr_value"] = atr_val
            pos_db["sl_price"] = current_price - (1.2 * atr_val)
            pos_db["tp_price"] = current_price + (3.0 * atr_val)
            pos_db["partial_tp_hit"] = 0
            
            pos_db["last_buy_time"] = str(datetime.now())
            
            database.update_position(pos_db)

            # Registra a compra no log de performance (V4)
            fee_usdt = usdt_utilizado * 0.001  # 0.1% taker fee
            database.log_trade(
                symbol=symbol, side='BUY', reason=decision,
                price=current_price, qty=qty_coin, fee_usdt=fee_usdt
            )

            print(f"[EXECUTION] 🟢 COMPRA EXECUTADA {symbol} | Qtd: {qty_coin} | Preço Médio: ${pos_db['avg_price']:.2f}")
            logger.info(f"Sucesso! Order ID: {order.get('orderId')} | Status: {order.get('status')}")

        elif decision in ["VENDA_FORTE", "VENDA_STOP", "VENDA_TIME_STOP", "VENDA_PARCIAL", "VENDA_MODERADA", "VENDA_LEVE"]:
            if pos_db["qty"] <= 0:
                return
                
            logger.info(f"[EXECUTION] {symbol} - Iniciando {decision}...")
            
            base_asset = get_base_asset(symbol)
            balance_info = client.get_asset_balance(asset=base_asset)
            balance = float(balance_info['free']) if balance_info else 0.0
            
            if decision == "VENDA_LEVE":
                coin_qty = balance * 0.25
            elif decision in ["VENDA_MODERADA", "VENDA_PARCIAL"]:
                coin_qty = balance * 0.50
            else:
                coin_qty = balance
                
            coin_qty = _arredondar_fracao(coin_qty, _decimais_para(symbol))
            
            if coin_qty <= 0:
                 logger.warning(f"[EXECUTION] {symbol} Saldo muito baixo para venda.")
                 if decision in ["VENDA_FORTE", "VENDA_STOP", "VENDA_TIME_STOP"]:
                     database.clear_position(symbol)
                 return
            
            order = client.order_market_sell(
                symbol=symbol,
                quantity=coin_qty
            )
            
            # Calcula P&L antes de limpar a posição
            avg_price = pos_db.get("avg_price", 0.0)
            pnl_pct   = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0.0
            fee_usdt  = coin_qty * current_price * 0.001

            if decision in ["VENDA_PARCIAL", "VENDA_MODERADA", "VENDA_LEVE"]:
                pos_db["qty"] -= coin_qty
                if decision == "VENDA_PARCIAL":
                    pos_db["partial_tp_hit"] = 1
                    # Break-even com margem de segurança
                    break_even_price = pos_db["avg_price"] * 1.0025
                    pos_db["sl_price"] = max(pos_db["sl_price"], break_even_price)
                database.update_position(pos_db)
            else:
                database.clear_position(symbol)

            # Registra a venda no log de performance (V4)
            database.log_trade(
                symbol=symbol, side='SELL', reason=decision,
                price=current_price, qty=coin_qty,
                fee_usdt=fee_usdt, pnl_pct=pnl_pct
            )

            emoji = "🟢" if pnl_pct > 0 else "🔴"
            print(f"[EXECUTION] {emoji} VENDA EXECUTADA {symbol} ({decision}) | Qtd: {coin_qty} | Preço: ${current_price:.2f} | P&L: {pnl_pct:+.2f}%")
            logger.info(f"Sucesso! Order ID: {order.get('orderId')} | Status: {order.get('status')}")

    except BinanceAPIException as e:
        logger.error(f"[EXECUTION API ERRO] {symbol} | Código: {e.status_code} | Mensagem: {e.message}")
    except Exception as e:
        logger.error(f"[EXECUTION ERRO INESPERADO] {symbol} | {str(e)}")

async def check_risk_management(symbol: str, current_price: float, current_time):
    """
    Avalia constantemente a posição aberta para gerenciar Stop Loss, Trailing Stop e Scale-out TP.
    """
    pos_db = database.get_position(symbol)
    
    if pos_db["qty"] <= 0:
        return
        
    updated = False
    
    if current_price > pos_db["highest_price"]:
        pos_db["highest_price"] = current_price
        updated = True
        
    # Trailing Stop: Break-even após lucro de 1.5 * ATR
    # Ativa mais cedo que antes (era 2.0×ATR) para proteger ganhos mais rapidamente
    atr_val = pos_db.get("atr_value", current_price * 0.002)
    if pos_db["highest_price"] >= pos_db["avg_price"] + (1.5 * atr_val):
        break_even_lock = pos_db["avg_price"] * 1.002 # Cobre as taxas de compra + venda (0.2%)
        if break_even_lock > pos_db["sl_price"]:
            pos_db["sl_price"] = break_even_lock
            updated = True
            logger.info(f"[RISK] 🔒 {symbol} Trailing Stop → Break-even ativado em ${break_even_lock:.4f}")
            
    if updated:
        database.update_position(pos_db)
        
    if current_price <= pos_db["sl_price"]:
        logger.warning(f"[RISK] 🚨 {symbol} Stop Loss/Trailing Stop! Preço: ${current_price:.2f} <= SL: ${pos_db['sl_price']:.2f}")
        await execute_trade(symbol, "VENDA_STOP", current_price, 1.0)
        return
        
    if current_price >= pos_db["tp_price"]:
        logger.info(f"[RISK] 💰 {symbol} Take Profit Principal atingido (${pos_db['tp_price']:.2f}). Fechando posição completa.")
        await execute_trade(symbol, "VENDA_FORTE", current_price, 1.0)
        return
        
    if pos_db["last_buy_time"] is not None:
        if not isinstance(current_time, pd.Timestamp):
            current_time = pd.to_datetime(current_time)
            
        last_buy = pd.to_datetime(pos_db["last_buy_time"])
        time_diff = current_time - last_buy
        minutos_aberto = time_diff.total_seconds() / 60
        # Time Stop: 4 horas (240 minutos) — compatível com estratégia scalping de 1M
        if minutos_aberto >= 240:
            logger.warning(f"[RISK] ⏰ {symbol} Time Stop atingido! Posição aberta há {minutos_aberto:.0f} minutos.")
            await execute_trade(symbol, "VENDA_TIME_STOP", current_price, 1.0)
            return
