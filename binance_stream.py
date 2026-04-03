import asyncio
import pandas as pd
import logging
from binance import AsyncClient, BinanceSocketManager
import market_state
from execution import execute_trade

# Configuração de Logger para Isolamento de Erros
logger = logging.getLogger(__name__)

async def start_stream():
    """
    Inicia o stream de conexão assíncrona com o WebSocket da Binance.
    Implementa um loop robusto de Auto-Reconnect em caso de desconexão.
    """
    # 1. Auto-Reconnect: Loop Externo principal
    while True:
        try:
            print("⏳ Iniciando (ou re-iniciando) conexão WebSocket com a Binance...")
            client = await AsyncClient.create()
            bm = BinanceSocketManager(client)
            
            # Inicia o socket para BTCUSDT no tempo gráfico de 1 minuto
            stream = bm.kline_socket(symbol='BTCUSDT', interval='1m')
            
            async with stream as ts:
                print("🟢 Conexão WebSocket estabelecida com sucesso! Aguardando fechamento de velas...")
                
                while True:
                    # Recebe a mensagem do WebSocket
                    res = await ts.recv()
                    
                    # 2. Validação do Payload (Fail-Safe contra mensagens de ping/sistema da Binance)
                    if res is None or 'k' not in res:
                        continue
                        
                    # 3. Isolamento de Erros: Falhas em uma vela não derrubam a conexão
                    try:
                        vela = res['k']
                        
                        # Verifica se a vela fechou (x == True)
                        if vela['x']:
                            # O timestamp 't' vem em milisegundos, converter p/ datetime
                            candle_time = pd.to_datetime(vela['t'], unit='ms')
                            preco_abertura = float(vela['o'])
                            preco_maxima = float(vela['h'])
                            preco_minima = float(vela['l'])
                            preco_fechamento = float(vela['c'])
                            volume = float(vela['v'])
                            
                            # Atualiza estado e aciona estratégia logicamente
                            new_candle = {
                                "date": candle_time,
                                "open": preco_abertura,
                                "high": preco_maxima,
                                "low": preco_minima,
                                "close": preco_fechamento,
                                "volume": volume
                            }
                            
                            analysis = market_state.append_new_candle(new_candle)
                            decision = analysis["decision"]
                            
                            # Extração de Métricas do Sistema de Scoring (evitando KeyErrors)
                            analise_dados = analysis.get("analysis", {})
                            rsi = analise_dados.get("rsi", "N/A")
                            buy_score = analise_dados.get("buy_score", 0)
                            sell_score = analise_dados.get("sell_score", 0)
                            
                            print(f"[TRADER.AI] 📊 Preço: ${preco_fechamento:.2f} | RSI: {rsi} | Score C: {buy_score} | Score V: {sell_score} | Decisão: {decision}")
                            
                            # Transmite a decisão ao motor de Execução / Gerenciamento de Risco
                            if decision.startswith("COMPRA_"):
                                pesos = {
                                    "COMPRA_LEVE": 0.25,
                                    "COMPRA_MODERADA": 0.50,
                                    "COMPRA_FORTE": 1.0
                                }
                                peso = pesos.get(decision, 1.0)
                                await execute_trade(decision, preco_fechamento, peso)
                            elif decision == "VENDA_FORTE":
                                await execute_trade(decision, preco_fechamento, 1.0)
                                
                    except Exception as loop_e:
                        logger.error(f"Falha ao processar dados da vela/sinais: {loop_e}")
                        print(f"⚠️ [ISOLAMENTO] Erro ao extrair dados quantitativos: {loop_e}. Ignorando vela...")
                        
        except Exception as str_e:
             # Ocorreu um ConnectionClosedError, Timeout, ou queda de servidor Binance
             logger.error(f"Erro Crítico no WebSocket: {str_e}")
             print(f"🔴 Queda de Rede! Erro: {str(str_e)}. Tentando se recuperar...")
             
        finally:
             print("🧹 Limpando buffer/cliente antes de tentar religar o sistema...")
             try:
                 if 'client' in locals():
                     await client.close_connection()
             except Exception:
                 pass
             
        # Pausa obrigatória para não ser banido pela Binance em caso de looping desenfreado
        print("⏳ Aguardando 5 segundos para nova tentativa de Auto-Reconnect...\n")
        await asyncio.sleep(5)
