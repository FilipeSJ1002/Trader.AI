"""
binance_stream.py — V6 (Trend-Following)
WebSocket multiplex para todos os ativos monitorados.

Ativos: BTC, ETH, XRP, SOL, BNB, AVAX
Streams: kline_1m + kline_1h por ativo

V6: a decisao deixou de ser scalping de 1M e passou a ser trend-following
diario (SMA200 + risk-on). A cada candle 1M atualizamos o preco/close diario
e rebalanceamos o portfolio quando o dia vira ou os sinais mudam.
"""
import asyncio
import pandas as pd
import logging
from binance import AsyncClient, BinanceSocketManager
import market_state
from execution import rebalance_portfolio
from trend_strategy import compute_target_weights

logger = logging.getLogger(__name__)

# Lista canônica de ativos — sincronizada com main.py e execution.py
ATIVOS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"]

# Estado do rebalanceamento (evita rebalancear toda hora)
_last_rebalance_day = None
_last_eligible      = {}
_last_heartbeat_min = None  # evita heartbeat duplicado por minuto


async def _maybe_rebalance(candle_time):
    """
    Recalcula os pesos-alvo do trend-following e rebalanceia quando:
      • muda o dia (a SMA200 diaria so muda 1x/dia), OU
      • muda o conjunto de ativos elegiveis (cruzamento de SMA200 intradia).
    Exibe heartbeat a cada minuto com o estado atual do mercado.
    """
    global _last_rebalance_day, _last_eligible, _last_heartbeat_min

    daily = market_state.get_daily_closes()
    if len(daily) < 2:
        return

    targets  = compute_target_weights(daily)
    eligible = {k: (v > 0) for k, v in targets.items()}
    day      = pd.to_datetime(candle_time).normalize()

    # Precos atuais
    prices = {}
    for s in ATIVOS:
        df = market_state.history_data.get(s)
        if df is not None and len(df) > 0:
            prices[s] = float(df['close'].iloc[-1])

    # Heartbeat uma vez por minuto (independente de quantos ativos fecham candle)
    current_min = pd.to_datetime(candle_time).strftime('%Y-%m-%d %H:%M')
    if current_min != _last_heartbeat_min:
        _last_heartbeat_min = current_min
        btc_price = prices.get("BTCUSDT", 0)
        alvo_str  = " | ".join(
            f"{k.replace('USDT','')}: {v*100:.0f}%"
            for k, v in targets.items() if v > 0
        ) or "CAIXA"
        estado = "BULL" if any(v > 0 for v in targets.values()) else "BEAR"
        logger.info(
            f"[BOT] {pd.to_datetime(candle_time).strftime('%H:%M')} | "
            f"BTC ${btc_price:,.0f} | {estado} | {alvo_str}"
        )

    if _last_rebalance_day == day and eligible == _last_eligible:
        return  # ja rebalanceou hoje e nada mudou

    if prices:
        try:
            await rebalance_portfolio(targets, prices)
            _last_rebalance_day = day
            _last_eligible      = eligible
        except Exception as e:
            logger.error(f"[REBAL] Falha no rebalanceamento: {e}")


async def start_stream():
    """
    Stream WebSocket multiplex para todos os ativos.
    Auto-reconnect automático em caso de queda de rede.
    """
    streams = []
    for symbol in ATIVOS:
        s = symbol.lower()
        streams.append(f"{s}@kline_1m")
        streams.append(f"{s}@kline_1h")

    while True:
        client = None
        try:
            print("[STREAM] Iniciando conexao WebSocket Multiplex com a Binance...")
            client = await AsyncClient.create()
            bm     = BinanceSocketManager(client, user_timeout=60)
            stream = bm.multiplex_socket(streams)

            async with stream as ts:
                print(f"[STREAM] Conexao estabelecida! Monitorando {len(ATIVOS)} ativos...")

                while True:
                    try:
                        res = await asyncio.wait_for(ts.recv(), timeout=60)
                    except asyncio.TimeoutError:
                        continue
                    except Exception as queue_e:
                        if "QueueOverflow" in str(queue_e) or "queue" in str(queue_e).lower():
                            logger.warning("[STREAM] Queue cheia — descartando mensagens antigas.")
                            await asyncio.sleep(0.1)
                            continue
                        raise

                    if res is None or 'data' not in res:
                        continue

                    try:
                        stream_name = res['stream']
                        vela        = res['data']['k']
                        symbol      = stream_name.split('@')[0].upper()

                        # ── Candle de 1H fechado → atualiza HTF ──────────────
                        if stream_name.endswith('@kline_1h') and vela['x']:
                            candle_time = pd.to_datetime(vela['t'], unit='ms')
                            market_state.update_1h_candle(symbol, {
                                "date":   candle_time,
                                "open":   float(vela['o']),
                                "high":   float(vela['h']),
                                "low":    float(vela['l']),
                                "close":  float(vela['c']),
                                "volume": float(vela['v'])
                            })

                        # ── Candle de 1M fechado → atualiza preco + rebalanceia ──
                        elif stream_name.endswith('@kline_1m') and vela['x']:
                            candle_time      = pd.to_datetime(vela['t'], unit='ms')
                            preco_fechamento = float(vela['c'])

                            new_candle = {
                                "date":   candle_time,
                                "open":   float(vela['o']),
                                "high":   float(vela['h']),
                                "low":    float(vela['l']),
                                "close":  preco_fechamento,
                                "volume": float(vela['v'])
                            }

                            # Atualiza buffer de preco + close diario (leve, sem scalping)
                            market_state.update_price_only(symbol, new_candle)

                            # Trend-following: rebalanceia quando o dia vira ou os sinais mudam
                            await _maybe_rebalance(candle_time)

                    except Exception as loop_e:
                        logger.error(f"[STREAM] Erro ao processar candle: {loop_e}")

        except Exception as str_e:
            logger.error(f"[STREAM] Erro critico no WebSocket: {str_e}")
            print(f"[STREAM] Queda de rede! Reconectando em 5s...")

        finally:
            try:
                if client is not None:
                    await client.close_connection()
            except Exception:
                pass

        await asyncio.sleep(5)
