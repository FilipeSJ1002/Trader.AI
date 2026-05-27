"""
binance_stream.py — V4
WebSocket multiplex para todos os ativos monitorados.

Ativos: BTC, ETH, XRP, SOL, BNB, AVAX
Streams: kline_1m + kline_1h por ativo
"""
import asyncio
import pandas as pd
import logging
from binance import AsyncClient, BinanceSocketManager
import market_state
from execution import execute_trade, check_risk_management

logger = logging.getLogger(__name__)

# Lista canônica de ativos — sincronizada com main.py e execution.py
ATIVOS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT"]


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
            bm     = BinanceSocketManager(client)
            stream = bm.multiplex_socket(streams)

            async with stream as ts:
                print(f"[STREAM] Conexao estabelecida! Monitorando {len(ATIVOS)} ativos...")

                while True:
                    res = await ts.recv()

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

                        # ── Candle de 1M fechado → análise principal ──────────
                        elif stream_name.endswith('@kline_1m') and vela['x']:
                            candle_time    = pd.to_datetime(vela['t'], unit='ms')
                            preco_fechamento = float(vela['c'])
                            volume           = float(vela['v'])

                            new_candle = {
                                "date":   candle_time,
                                "open":   float(vela['o']),
                                "high":   float(vela['h']),
                                "low":    float(vela['l']),
                                "close":  preco_fechamento,
                                "volume": volume
                            }

                            # Ignora símbolo sem histórico carregado
                            if symbol not in market_state.history_data:
                                continue

                            analysis = market_state.append_new_candle(symbol, new_candle)
                            decision = analysis["decision"]

                            analise   = analysis.get("analysis", {})
                            rsi       = analise.get("rsi", "N/A")
                            atr       = analise.get("atr", 0.0)
                            buy_score = analise.get("buy_score", 0)
                            regime    = analise.get("market_regime", "?")
                            sig_5m    = analise.get("signal_5m", "?")
                            vol_ratio = analise.get("vol_ratio", 0.0)
                            ml_ok     = analise.get("ml_status", "?")

                            print(
                                f"[TRADER.AI] {symbol} | ${preco_fechamento:.4f} | "
                                f"RSI:{rsi} ATR:{atr:.4f} Vol:{vol_ratio:.2f}x | "
                                f"Score:{buy_score} | Regime:{regime} 5M:{sig_5m} | "
                                f"ML:{ml_ok} | => {decision}"
                            )

                            # ── Gestão de risco sempre ────────────────────────
                            await check_risk_management(symbol, preco_fechamento, candle_time)

                            # ── Execução de ordens ────────────────────────────
                            if decision.startswith("COMPRA_"):
                                pesos = {
                                    "COMPRA_LEVE":     0.25,
                                    "COMPRA_MODERADA": 0.50,
                                    "COMPRA_FORTE":    1.0
                                }
                                peso = pesos.get(decision, 1.0)
                                await execute_trade(symbol, decision, preco_fechamento, peso, atr)

                            elif decision in ["VENDA_FORTE", "VENDA_MODERADA", "VENDA_LEVE"]:
                                await execute_trade(symbol, decision, preco_fechamento, 1.0, atr)

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
