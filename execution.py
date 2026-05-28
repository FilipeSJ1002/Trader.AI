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
API_KEY    = os.getenv("BINANCE_API_KEY")
SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Chaves de API ausentes no .env!")

try:
    client = Client(API_KEY, SECRET_KEY, testnet=True)
except Exception as e:
    logger.error(f"Erro ao conectar na Binance Testnet: {e}")
    client = None

from binance_stream import ATIVOS  # fonte única da lista de ativos

# ─────────────────────────────────────────────────────────────────────────────
# Precisão de quantidade por ativo (casas decimais para floor)
# ─────────────────────────────────────────────────────────────────────────────
_QTY_DECIMALS: dict[str, int] = {
    "BTCUSDT":  5,
    "ETHUSDT":  4,
    "XRPUSDT":  1,
    "SOLUSDT":  2,
    "BNBUSDT":  3,
    "AVAXUSDT": 2,
}

_MIN_BALANCES: dict[str, float] = {
    "BTC":  0.0001,
    "ETH":  0.001,
    "XRP":  1.0,
    "SOL":  0.01,
    "BNB":  0.001,
    "AVAX": 0.01,
}

# ─────────────────────────────────────────────────────────────────────────────
# Fase 4.4 — Filtro de Correlação
# ─────────────────────────────────────────────────────────────────────────────
# Ativos do mesmo grupo têm alta correlação — limita a 1 posição por grupo
CORRELATION_GROUPS: dict[str, str] = {
    "BTCUSDT":  "MEGA_CAP",   # BTC e ETH movem juntos ~85% do tempo
    "ETHUSDT":  "MEGA_CAP",
    "SOLUSDT":  "ALT_L1",     # SOL e AVAX são Layer-1 alternativos correlacionados
    "AVAXUSDT": "ALT_L1",
    "BNBUSDT":  "EXCHANGE",   # BNB é independente (exchange token)
    "XRPUSDT":  "PAYMENTS",   # XRP segue dinâmica própria (pagamentos/regulatório)
}
MAX_PER_GROUP = 1  # máximo 1 posição por grupo de correlação

# ─────────────────────────────────────────────────────────────────────────────
# Fase 4.1 — Dimensionamento Dinâmico de Posição
# ─────────────────────────────────────────────────────────────────────────────
BASE_ALLOC_PCT   = 0.15   # 15% do equity como base por trade
MAX_ALLOC_PCT    = 0.25   # nunca mais que 25% por posição
MIN_USDT_TRADE   = 11.0   # mínimo da Binance

def _calcular_tamanho_posicao(
    usdt_free: float,
    total_equity: float,
    decision: str,
    ml_prob: float = 0.65,
    regime: str = "UNKNOWN",
    asset_strength: str = "NEUTRAL",
) -> float:
    """
    Calcula o tamanho USDT da posição de forma dinâmica.

    Fórmula:
      base = equity * 15%
      × multiplicador ML (confiança acima de 60%)
      × multiplicador de regime de mercado
      × multiplicador de força relativa do ativo
      × multiplicador da decisão (FORTE/MODERADA/LEVE)
      capped em 25% do equity e 90% do USDT livre
    """
    base = total_equity * BASE_ALLOC_PCT

    # Multiplicador do ML: 0.60 → 1.0x | 0.80 → 1.5x | 1.00 → 2.0x
    ml_prob = max(0.60, min(1.0, ml_prob))
    ml_mult = 1.0 + (ml_prob - 0.60) / 0.40

    # Multiplicador de regime
    regime_mult = {
        "BULL_TREND": 1.20,
        "SIDEWAYS":   0.85,
        "BEAR_TREND": 0.50,
        "UNKNOWN":    1.00,
    }.get(regime, 1.00)

    # Multiplicador de força relativa
    strength_mult = {
        "STRONG":  1.30,
        "NEUTRAL": 1.00,
        "WEAK":    0.70,
    }.get(asset_strength, 1.00)

    # Multiplicador da decisão
    decision_mult = {
        "COMPRA_FORTE":    1.00,
        "COMPRA_MODERADA": 0.75,
        "COMPRA_LEVE":     0.50,
    }.get(decision, 1.00)

    usdt = base * ml_mult * regime_mult * strength_mult * decision_mult

    # Hard caps
    usdt = min(usdt, total_equity * MAX_ALLOC_PCT)   # max 25% por posição
    usdt = min(usdt, usdt_free * 0.90)                # deixa 10% de USDT livre

    return usdt


def get_base_asset(symbol: str) -> str:
    return symbol.replace("USDT", "")


def _decimais_para(symbol: str) -> int:
    return _QTY_DECIMALS.get(symbol, 5)


def _correlation_ok(symbol: str) -> bool:
    """
    Retorna True se não há bloqueio de correlação para abrir nova posição.
    Verifica se já existe posição aberta em outro ativo do mesmo grupo.
    """
    my_group = CORRELATION_GROUPS.get(symbol)
    if my_group is None:
        return True
    for other in ATIVOS:
        if other == symbol:
            continue
        if CORRELATION_GROUPS.get(other) == my_group:
            pos = database.get_position(other)
            if pos["qty"] > 0:
                logger.info(f"[CORR] {symbol} bloqueado: {other} ({my_group}) ja esta aberto.")
                return False
    return True


def sync_position_state():
    """Sincroniza as posições reais da Binance com o banco de dados."""
    if not client:
        return
    try:
        for symbol in ATIVOS:
            base_asset   = get_base_asset(symbol)
            balance_info = client.get_asset_balance(asset=base_asset)
            balance      = float(balance_info['free']) if balance_info else 0.0
            min_bal      = _MIN_BALANCES.get(base_asset, 0.001)
            pos_db       = database.get_position(symbol)

            if balance > min_bal:
                if pos_db["qty"] == 0:
                    trades      = client.get_my_trades(symbol=symbol, limit=5)
                    buyer_trades = [t for t in trades if t['isBuyer']]
                    if buyer_trades:
                        last_buy_price           = float(buyer_trades[-1]['price'])
                        pos_db["avg_price"]      = last_buy_price
                        pos_db["highest_price"]  = last_buy_price
                        pos_db["atr_value"]      = last_buy_price * 0.002
                        pos_db["sl_price"]       = last_buy_price - (1.2 * pos_db["atr_value"])
                        pos_db["tp_price"]       = last_buy_price + (3.0 * pos_db["atr_value"])
                        pos_db["tp1_price"]      = last_buy_price + (1.5 * pos_db["atr_value"])
                        pos_db["tp1_hit"]        = 0
                        pos_db["partial_tp_hit"] = 0
                        pos_db["last_buy_time"]  = str(pd.to_datetime(buyer_trades[-1]['time'], unit='ms'))
                        logger.info(f"[SYNC] Posicao aberta {symbol}. Saldo: {balance} | Preco: ${last_buy_price:.2f}")
                    else:
                        current_price            = float(client.get_ticker(symbol=symbol)['lastPrice'])
                        pos_db["avg_price"]      = current_price
                        pos_db["highest_price"]  = current_price
                        pos_db["atr_value"]      = current_price * 0.002
                        pos_db["sl_price"]       = current_price - (1.2 * pos_db["atr_value"])
                        pos_db["tp_price"]       = current_price + (3.0 * pos_db["atr_value"])
                        pos_db["tp1_price"]      = current_price + (1.5 * pos_db["atr_value"])
                        pos_db["tp1_hit"]        = 0
                        pos_db["partial_tp_hit"] = 0
                        pos_db["last_buy_time"]  = str(datetime.now())
                        logger.warning(f"[SYNC] {symbol} posicao sem historico. Preco ajustado: ${current_price:.2f}")

                pos_db["qty"] = balance
                database.update_position(pos_db)
            else:
                database.clear_position(symbol)
                logger.info(f"[SYNC] {symbol} Nenhuma posicao relevante. Saldo: {balance}")
    except BinanceAPIException as e:
        logger.error(f"[SYNC ERRO] Falha ao sincronizar posicao: {e.message}")
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
    usdt_free    = 0.0
    total_equity = 0.0
    for asset in account_info['balances']:
        free   = float(asset['free'])
        locked = float(asset['locked'])
        total  = free + locked
        if total > 0:
            if asset['asset'] == 'USDT':
                usdt_free    = free
                total_equity += total
            else:
                try:
                    ticker        = client.get_ticker(symbol=f"{asset['asset']}USDT")
                    price         = float(ticker['lastPrice'])
                    total_equity += total * price
                except Exception:
                    pass
    exposure_ratio = (total_equity - usdt_free) / total_equity if total_equity > 0 else 0.0
    return usdt_free, total_equity, exposure_ratio


async def execute_trade(
    symbol: str,
    decision: str,
    current_price: float,
    peso: float = 1.0,
    atr: float = 0.0,
    ml_prob: float = 0.65,           # Fase 4.1 — confiança do ML
    asset_strength: str = "NEUTRAL", # Fase 4.1 — força relativa
    regime: str = "UNKNOWN",         # Fase 4.1 — regime de mercado
    features: dict = None,           # Fase 5 — vetor de features p/ loop de aprendizado
):
    """
    Motor de execução de ordens a mercado para Multi-Asset.
    Fase 4: dimensionamento dinâmico, proteção de drawdown, filtro de correlação,
            e suporte ao TP em escada (VENDA_TP1).
    """
    if not client:
        logger.error("[EXECUTION ERRO] Cliente Binance nao inicializado.")
        return

    pos_db = database.get_position(symbol)

    try:
        # ── COMPRA ───────────────────────────────────────────────────────────
        if decision.startswith("COMPRA_"):
            logger.info(f"[EXECUTION] {symbol} - Iniciando {decision}... (ML:{ml_prob:.2f} | Regime:{regime} | Forca:{asset_strength})")

            # Fase 4.2 — Proteção de drawdown diário
            if database.is_trading_paused():
                logger.warning(f"[EXECUTION] {symbol} Compra bloqueada: drawdown diario atingido. Aguardando reset as 00h UTC.")
                return

            usdt_free, total_equity, exposure_ratio = calculate_account_exposure()

            if exposure_ratio >= 0.95:
                logger.warning(f"[EXECUTION] Compra {symbol} bloqueada: exposicao da conta em {exposure_ratio*100:.1f}%.")
                return

            # Fase 4.4 — Filtro de correlação
            if not _correlation_ok(symbol):
                logger.warning(f"[EXECUTION] {symbol} Compra bloqueada: ativo correlacionado ja aberto.")
                return

            # Fase 4.1 — Dimensionamento dinâmico
            usdt_utilizado = _calcular_tamanho_posicao(
                usdt_free=usdt_free,
                total_equity=total_equity,
                decision=decision,
                ml_prob=ml_prob,
                regime=regime,
                asset_strength=asset_strength,
            )

            if usdt_utilizado < MIN_USDT_TRADE:
                logger.warning(f"[EXECUTION] {symbol} Compra ignorada: lote muito pequeno ({usdt_utilizado:.2f} USDT).")
                return

            qty_coin = _arredondar_fracao(usdt_utilizado / current_price, _decimais_para(symbol))

            if qty_coin <= 0:
                logger.warning(f"[EXECUTION] {symbol} qty calculada e zero. Ignorando.")
                return

            order = client.order_market_buy(symbol=symbol, quantity=qty_coin)

            # Atualiza posição
            qtd_antiga   = pos_db["qty"]
            preco_antigo = pos_db["avg_price"]
            if qtd_antiga > 0:
                novo_preco_medio = ((preco_antigo * qtd_antiga) + (current_price * qty_coin)) / (qtd_antiga + qty_coin)
            else:
                novo_preco_medio = current_price

            atr_val = atr if atr > 0 else (current_price * 0.002)

            pos_db["avg_price"]     = novo_preco_medio
            pos_db["qty"]          += qty_coin
            pos_db["highest_price"] = current_price
            pos_db["atr_value"]     = atr_val

            # SL: 1.2×ATR | TP1: 1.5×ATR (40%) | TP2/TP: 3.0×ATR (60%)
            pos_db["sl_price"]      = current_price - (1.2 * atr_val)
            pos_db["tp1_price"]     = current_price + (1.5 * atr_val)   # Fase 4.3
            pos_db["tp_price"]      = current_price + (3.0 * atr_val)
            pos_db["tp1_hit"]       = 0
            pos_db["partial_tp_hit"] = 0
            pos_db["last_buy_time"] = str(datetime.now())

            database.update_position(pos_db)

            fee_usdt = usdt_utilizado * 0.001
            database.log_trade(
                symbol=symbol, side='BUY', reason=decision,
                price=current_price, qty=qty_coin, fee_usdt=fee_usdt
            )

            # Fase 5 — salva features de entrada para o loop de aprendizado
            if features:
                try:
                    database.save_trade_features(
                        symbol=symbol,
                        entry_time=pos_db["last_buy_time"],
                        entry_price=current_price,
                        features=features,
                    )
                except Exception as e:
                    logger.debug(f"[LEARN] Falha ao salvar features de {symbol}: {e}")

            # Atualiza equity diário
            _, eq, _ = calculate_account_exposure()
            database.update_daily_equity(eq)

            print(f"[EXECUTION] COMPRA EXECUTADA {symbol} | Qtd: {qty_coin} | "
                  f"USDT: ${usdt_utilizado:.0f} ({usdt_utilizado/total_equity*100:.1f}% equity) | "
                  f"Preco Medio: ${pos_db['avg_price']:.2f} | "
                  f"SL:${pos_db['sl_price']:.2f} TP1:${pos_db['tp1_price']:.2f} TP2:${pos_db['tp_price']:.2f}")
            logger.info(f"Sucesso! Order ID: {order.get('orderId')} | Status: {order.get('status')}")

        # ── VENDA ────────────────────────────────────────────────────────────
        elif decision in [
            "VENDA_FORTE", "VENDA_STOP", "VENDA_TIME_STOP",
            "VENDA_PARCIAL", "VENDA_MODERADA", "VENDA_LEVE",
            "VENDA_TP1",   # Fase 4.3 — TP1 scale-out
        ]:
            if pos_db["qty"] <= 0:
                return

            logger.info(f"[EXECUTION] {symbol} - Iniciando {decision}...")

            base_asset   = get_base_asset(symbol)
            balance_info = client.get_asset_balance(asset=base_asset)
            balance      = float(balance_info['free']) if balance_info else 0.0

            # Proporção a vender conforme o tipo
            if decision == "VENDA_TP1":
                coin_qty = balance * 0.40     # Fase 4.3: 40% no primeiro TP
            elif decision == "VENDA_LEVE":
                coin_qty = balance * 0.25
            elif decision in ("VENDA_MODERADA", "VENDA_PARCIAL"):
                coin_qty = balance * 0.50
            else:
                coin_qty = balance             # FORTE, STOP, TIME_STOP — fecha tudo

            coin_qty = _arredondar_fracao(coin_qty, _decimais_para(symbol))

            if coin_qty <= 0:
                logger.warning(f"[EXECUTION] {symbol} Saldo muito baixo para venda.")
                if decision in ("VENDA_FORTE", "VENDA_STOP", "VENDA_TIME_STOP"):
                    database.clear_position(symbol)
                return

            order = client.order_market_sell(symbol=symbol, quantity=coin_qty)

            avg_price = pos_db.get("avg_price", 0.0)
            pnl_pct   = ((current_price - avg_price) / avg_price * 100) if avg_price > 0 else 0.0
            fee_usdt  = coin_qty * current_price * 0.001

            if decision in ("VENDA_PARCIAL", "VENDA_MODERADA", "VENDA_LEVE"):
                pos_db["qty"] -= coin_qty
                if decision == "VENDA_PARCIAL":
                    pos_db["partial_tp_hit"] = 1
                    pos_db["sl_price"] = max(pos_db["sl_price"], pos_db["avg_price"] * 1.0025)
                database.update_position(pos_db)

            elif decision == "VENDA_TP1":
                # Fase 4.3: escada — vende 40%, sobe SL para breakeven
                pos_db["qty"]      -= coin_qty
                pos_db["tp1_hit"]   = 1
                be_price = pos_db["avg_price"] * 1.002   # breakeven + taxas
                pos_db["sl_price"] = max(pos_db["sl_price"], be_price)
                logger.info(f"[RISK] {symbol} TP1 executado. SL movido para breakeven: ${be_price:.4f}")
                database.update_position(pos_db)

            else:
                # Fechamento total — Fase 5: rotula o trade para o loop de aprendizado.
                # "Ganho" se bateu TP1 (lucro travado) OU se o P&L final foi positivo.
                won = (pos_db.get("tp1_hit", 0) == 1) or (pnl_pct > 0)
                try:
                    database.close_trade_features(symbol, pnl_pct, won=won)
                except Exception as e:
                    logger.debug(f"[LEARN] Falha ao fechar features de {symbol}: {e}")
                database.clear_position(symbol)

            database.log_trade(
                symbol=symbol, side='SELL', reason=decision,
                price=current_price, qty=coin_qty,
                fee_usdt=fee_usdt, pnl_pct=pnl_pct
            )

            # Atualiza equity diário e verifica drawdown
            _, eq, _ = calculate_account_exposure()
            paused = database.update_daily_equity(eq)
            if paused:
                logger.warning(f"[RISK] Drawdown diario atingido (>{database.DAILY_DRAWDOWN_LIMIT*100:.0f}%). Compras pausadas ate 00h UTC.")

            emoji = "LUCRO" if pnl_pct > 0 else "PERDA"
            print(f"[EXECUTION] VENDA {symbol} ({decision}) [{emoji}] | "
                  f"Qtd: {coin_qty} | Preco: ${current_price:.2f} | P&L: {pnl_pct:+.2f}%")
            logger.info(f"Sucesso! Order ID: {order.get('orderId')} | Status: {order.get('status')}")

    except BinanceAPIException as e:
        logger.error(f"[EXECUTION API ERRO] {symbol} | Codigo: {e.status_code} | Mensagem: {e.message}")
    except Exception as e:
        logger.error(f"[EXECUTION ERRO INESPERADO] {symbol} | {str(e)}")


# ─────────────────────────────────────────────────────────────────────────────
# V6 — Motor de Rebalanceamento Trend-Following
# ─────────────────────────────────────────────────────────────────────────────
TREND_INVEST_FRACTION   = 0.98   # usa 98% do equity, 2% de caixa de seguranca
REBALANCE_MIN_DELTA_PCT = 0.03   # so movimenta se o ajuste for > 3% do equity


async def rebalance_portfolio(target_weights: dict, current_prices: dict):
    """
    Rebalanceia o portfolio para os pesos-alvo do trend_strategy (V6).
    Vende o que saiu da tendencia, compra/ajusta o que esta em tendencia.

    target_weights : {symbol: peso 0..1} (soma <= 1)
    current_prices : {symbol: preco atual de mercado}
    """
    if not client:
        logger.error("[REBAL] Cliente Binance nao inicializado.")
        return

    usdt_free, total_equity, _ = calculate_account_exposure()
    if total_equity <= 0:
        return

    ativos_alvo = {k: round(v, 3) for k, v in target_weights.items() if v > 0}
    logger.info(f"[REBAL] Equity=${total_equity:.2f} | Alvos={ativos_alvo}")

    thr = max(MIN_USDT_TRADE, total_equity * REBALANCE_MIN_DELTA_PCT)

    # Calcula os deltas (valor a ajustar) por ativo
    deltas = {}
    for symbol in ATIVOS:
        price = current_prices.get(symbol)
        if not price or price <= 0:
            continue
        tw            = target_weights.get(symbol, 0.0)
        target_value  = total_equity * tw * TREND_INVEST_FRACTION
        pos           = database.get_position(symbol)
        current_value = pos["qty"] * price
        deltas[symbol] = (target_value - current_value, price, pos)

    # 1) VENDAS primeiro (liberam USDT para as compras)
    for symbol, (dv, price, pos) in deltas.items():
        if dv >= -thr or pos["qty"] <= 0:
            continue
        try:
            base = get_base_asset(symbol)
            bal  = float(client.get_asset_balance(asset=base)["free"])
            full_exit = target_weights.get(symbol, 0.0) <= 0
            sell_qty  = bal if full_exit else min(bal, abs(dv) / price)
            sell_qty  = _arredondar_fracao(sell_qty, _decimais_para(symbol))
            if sell_qty <= 0:
                continue

            client.order_market_sell(symbol=symbol, quantity=sell_qty)
            avg = pos.get("avg_price", 0.0) or price
            pnl = (price - avg) / avg * 100 if avg > 0 else 0.0
            reason = "TREND_EXIT" if full_exit else "TREND_REBAL"
            database.log_trade(symbol, "SELL", reason, price, sell_qty,
                               fee_usdt=sell_qty * price * 0.001, pnl_pct=pnl)

            if full_exit:
                try:
                    database.close_trade_features(symbol, pnl, won=(pnl > 0))
                except Exception:
                    pass
                database.clear_position(symbol)
                print(f"[REBAL] SAIDA {symbol} | {sell_qty} @ ${price:.4f} | P&L {pnl:+.2f}%")
            else:
                pos["qty"] -= sell_qty
                database.update_position(pos)
                print(f"[REBAL] REDUZ {symbol} | -{sell_qty} @ ${price:.4f}")
        except BinanceAPIException as e:
            logger.error(f"[REBAL VENDA] {symbol}: {e.message}")

    # 2) COMPRAS (recalcula USDT livre apos as vendas)
    usdt_free, total_equity, _ = calculate_account_exposure()
    for symbol, (dv, price, pos) in deltas.items():
        if dv <= thr:
            continue
        try:
            invest = min(dv, usdt_free * 0.98)
            if invest < MIN_USDT_TRADE:
                continue
            qty = _arredondar_fracao(invest / price, _decimais_para(symbol))
            if qty <= 0:
                continue

            client.order_market_buy(symbol=symbol, quantity=qty)
            old_qty, old_avg = pos["qty"], pos.get("avg_price", 0.0)
            new_avg = ((old_avg * old_qty) + (price * qty)) / (old_qty + qty) if (old_qty + qty) > 0 else price
            pos["qty"]           = old_qty + qty
            pos["avg_price"]     = new_avg
            pos["highest_price"] = max(pos.get("highest_price", 0.0), price)
            pos["last_buy_time"] = str(datetime.now())
            database.update_position(pos)
            database.log_trade(symbol, "BUY", "TREND_ENTRY", price, qty, fee_usdt=invest * 0.001)
            usdt_free -= invest
            print(f"[REBAL] ENTRA {symbol} | {qty} @ ${price:.4f} | ${invest:.0f} "
                  f"({target_weights.get(symbol,0)*100:.0f}% alvo)")
        except BinanceAPIException as e:
            logger.error(f"[REBAL COMPRA] {symbol}: {e.message}")

    _, eq, _ = calculate_account_exposure()
    database.update_daily_equity(eq)


async def check_risk_management(
    symbol: str,
    current_price: float,
    current_time,
    exit_prob: float = 0.0,
):
    """
    Gerencia Stop Loss, Trailing Stop, Take Profit em escada e Exit Model.
    Fase 4.3: verifica TP1 (1.5×ATR) antes do TP2 (3.0×ATR).
    """
    pos_db = database.get_position(symbol)

    if pos_db["qty"] <= 0:
        return

    updated = False

    if current_price > pos_db["highest_price"]:
        pos_db["highest_price"] = current_price
        updated = True

    # ── Exit Model (Fase 3.3) — Saída Antecipada ─────────────────────────────
    if exit_prob >= 0.80 and pos_db["avg_price"] > 0:
        pnl_pct = (current_price - pos_db["avg_price"]) / pos_db["avg_price"]
        if pnl_pct > 0.001:
            logger.warning(f"[RISK] {symbol} Exit Model forte: prob={exit_prob:.2f} | P&L={pnl_pct*100:.2f}% -> VENDA_FORTE")
            await execute_trade(symbol, "VENDA_FORTE", current_price, 1.0)
            return
    elif exit_prob >= 0.72 and pos_db["avg_price"] > 0:
        pnl_pct = (current_price - pos_db["avg_price"]) / pos_db["avg_price"]
        if pnl_pct > 0.002:
            logger.info(f"[RISK] {symbol} Exit Model moderado: prob={exit_prob:.2f} -> VENDA_PARCIAL")
            await execute_trade(symbol, "VENDA_PARCIAL", current_price, 1.0)
            return

    # ── Trailing Stop: Break-even após 1.5×ATR ───────────────────────────────
    atr_val = pos_db.get("atr_value", current_price * 0.002)
    if pos_db["highest_price"] >= pos_db["avg_price"] + (1.5 * atr_val):
        break_even_lock = pos_db["avg_price"] * 1.002
        if break_even_lock > pos_db["sl_price"]:
            pos_db["sl_price"] = break_even_lock
            updated = True
            logger.info(f"[RISK] {symbol} Trailing Stop -> Break-even: ${break_even_lock:.4f}")

    if updated:
        database.update_position(pos_db)

    # ── Stop Loss / Trailing Stop ─────────────────────────────────────────────
    if current_price <= pos_db["sl_price"]:
        logger.warning(f"[RISK] {symbol} STOP LOSS! Preco: ${current_price:.2f} <= SL: ${pos_db['sl_price']:.2f}")
        await execute_trade(symbol, "VENDA_STOP", current_price, 1.0)
        return

    # ── Fase 4.3 — Take Profit em Escada ─────────────────────────────────────
    # TP1: 1.5×ATR — vende 40%, move SL para breakeven
    tp1_price = pos_db.get("tp1_price", 0.0)
    tp1_hit   = pos_db.get("tp1_hit", 0)
    if tp1_price > 0 and not tp1_hit and current_price >= tp1_price:
        logger.info(f"[RISK] {symbol} TP1 atingido! ${current_price:.2f} >= ${tp1_price:.2f} | Vendendo 40%...")
        await execute_trade(symbol, "VENDA_TP1", current_price, 1.0)
        return

    # TP2 (final): 3.0×ATR — fecha posição completa
    if current_price >= pos_db["tp_price"]:
        logger.info(f"[RISK] {symbol} TP2 FINAL atingido! ${current_price:.2f} >= ${pos_db['tp_price']:.2f} | Fechando posicao.")
        await execute_trade(symbol, "VENDA_FORTE", current_price, 1.0)
        return

    # ── Time Stop: 4 horas (240 minutos) ─────────────────────────────────────
    if pos_db["last_buy_time"] is not None:
        if not isinstance(current_time, pd.Timestamp):
            current_time = pd.to_datetime(current_time)
        last_buy       = pd.to_datetime(pos_db["last_buy_time"])
        minutos_aberto = (current_time - last_buy).total_seconds() / 60
        if minutos_aberto >= 240:
            logger.warning(f"[RISK] {symbol} Time Stop! Posicao aberta ha {minutos_aberto:.0f} minutos.")
            await execute_trade(symbol, "VENDA_TIME_STOP", current_price, 1.0)
            return
