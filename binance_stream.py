import asyncio
import pandas as pd
import logging
from binance import AsyncClient, BinanceSocketManager
import market_state
from execution import execute_trade, check_risk_management

logger = logging.getLogger(__name__)

async def start_stream():
    """
    Inicia o stream de conexão assíncrona com o WebSocket da Binance.
    Implementa um loop robusto de Auto-Reconnect em caso de desconexão.
    """
    streams = [
        'btcusdt@kline_1m', 'btcusdt@kline_1h',
        'ethusdt@kline_1m', 'ethusdt@kline_1h',
        'xrpusdt@kline_1m', 'xrpusdt@kline_1h'
    ]
    
    while True:
        client = None
        try:
            print("⏳ Iniciando (ou re-iniciando) conexão WebSocket com a Binance...")
            client = await AsyncClient.create()
            bm = BinanceSocketManager(client)
            
            stream = bm.multiplex_socket(streams)
            
            async with stream as ts:
                print("✅ Conexão WebSocket (Multiplex) estabelecida com sucesso! Aguardando velas...")
                
                while True:
                    res = await ts.recv()
                    
                    if res is None or 'data' not in res:
                        continue
                        
                    try:
                        stream_name = res['stream']
                        vela = res['data']['k']
                        
                        symbol = stream_name.split('@')[0].upper()
                        
                        if stream_name.endswith('@kline_1h'):
                            if vela['x']:
                                candle_time = pd.to_datetime(vela['t'], unit='ms')
                                new_mtf = {
                                    "date": candle_time,
                                    "open": float(vela['o']),
                                    "high": float(vela['h']),
                                    "low": float(vela['l']),
                                    "close": float(vela['c']),
                                    "volume": float(vela['v'])
                                }
                                market_state.update_1h_candle(symbol, new_mtf)
                                
                        elif stream_name.endswith('@kline_1m'):
                            if vela['x']:
                                candle_time = pd.to_datetime(vela['t'], unit='ms')
                                preco_abertura = float(vela['o'])
                                preco_maxima = float(vela['h'])
                                preco_minima = float(vela['l'])
                                preco_fechamento = float(vela['c'])
                                volume = float(vela['v'])
                                
                                new_candle = {
                                    "date": candle_time,
                                    "open": preco_abertura,
                                    "high": preco_maxima,
                                    "low": preco_minima,
                                    "close": preco_fechamento,
                                    "volume": volume
                                }
                                
                                analysis = market_state.append_new_candle(symbol, new_candle)
                                decision = analysis["decision"]
                                
                                analise_dados = analysis.get("analysis", {})
                                rsi = analise_dados.get("rsi", "N/A")
                                buy_score = analise_dados.get("buy_score", 0)
                                sell_score = analise_dados.get("sell_score", 0)
                                htf_weakness = analise_dados.get("htf_weakness", False)
                                atr = analise_dados.get("atr", 0.0)
                                
                                print(f"[TRADER.AI] {symbol} | Preço: ${preco_fechamento:.4f} | RSI: {rsi} | ATR: {atr:.4f} | Score C: {buy_score} | Decisão: {decision} | MTF Fraqueza: {htf_weakness}")
                                
                                await check_risk_management(symbol, preco_fechamento, candle_time)
                                
                                if decision.startswith("COMPRA_"):
                                    pesos = {
                                        "COMPRA_LEVE": 0.25,
                                        "COMPRA_MODERADA": 0.50,
                                        "COMPRA_FORTE": 1.0
                                    }
                                    peso = pesos.get(decision, 1.0)
                                    await execute_trade(symbol, decision, preco_fechamento, peso, atr)
                                elif decision == "VENDA_FORTE":
                                    await execute_trade(symbol, decision, preco_fechamento, 1.0, atr)
                                
                    except Exception as loop_e:
                        logger.error(f"Falha ao processar dados da vela/sinais: {loop_e}")
                        print(f"⚠️ [ISOLAMENTO] Erro ao extrair dados quantitativos: {loop_e}. Ignorando vela...")
                        
        except Exception as str_e:
             logger.error(f"Erro Crítico no WebSocket: {str_e}")
             print(f"❌ Queda de Rede! Erro: {str(str_e)}. Tentando se recuperar...")
             
        finally:
             print("Limpando buffer/cliente antes de tentar religar o sistema...")
             try:
                 if client is not None:
                     await client.close_connection()
             except Exception:
                 pass
             
        print("Aguardando 5 segundos para nova tentativa de Auto-Reconnect...\n")
        await asyncio.sleep(5)
