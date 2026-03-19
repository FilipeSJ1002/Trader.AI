import asyncio
import pandas as pd
from binance import AsyncClient, BinanceSocketManager
import market_state

async def start_stream():
    """
    Inicia o stream de conexão assíncrona com o WebSocket da Binance.
    """
    print("⏳ Iniciando conexão com a Binance...")
    client = await AsyncClient.create()
    bm = BinanceSocketManager(client)
    
    # Inicia o socket para BTCUSDT no tempo gráfico de 1 minuto
    stream = bm.kline_socket(symbol='BTCUSDT', interval='1m')
    
    try:
        async with stream as ts:
            print("🟢 Conexão WebSocket estabelecida com sucesso! Aguardando fechamento de velas...")
            while True:
                res = await ts.recv()
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
                    
                    try:
                        analysis = market_state.append_new_candle(new_candle)
                        decision = analysis["decision"]
                        # Prevenção: a estratégia retona um dicionário em analysis["analysis"]
                        rsi = analysis["analysis"]["rsi"]
                        
                        print(f"[TRADER.AI] 📊 Preço: ${preco_fechamento:.2f} | RSI: {rsi} | Decisão: {decision}")
                    except Exception as loop_e:
                         print(f"Erro na analise do stream: {loop_e}")
                         
    except Exception as e:
        print(f"Erro na conexão WebSocket: {e}")
    finally:
        await client.close_connection()
        print("🔴 Conexão com a Binance encerrada.")
