# -*- coding: utf-8 -*-
"""
v7_coleta_alternativa.py — dados que NAO sao preco nem volume
==============================================================

Por que existe (15/08/2026): a sessao de medicao provou que preco e volume em
candles de 1 minuto nao contem sinal suficiente para pagar 0,08% de custo por
operacao — em regras, em padroes de candle e em rede neural, com amostras de
centenas a dezenas de milhares de operacoes. Ver [[Limite de Edge]].

Sobrou uma porta: informacao sobre o COMPORTAMENTO DOS OUTROS PARTICIPANTES,
que o preco nao mostra.

O que a Binance entrega (levantado em 15/08/2026):

    funding rate     historico COMPLETO desde 2019  -> da para backtestar
    open interest    so ~21 dias                    -> coletar daqui pra frente
    long/short ratio so ~21 dias                    -> idem
    taker buy/sell   so ~21 dias                    -> idem
    liquidacoes      endpoint publico removido      -> so websocket ao vivo

Por isso este script faz duas coisas diferentes:

  1. BAIXA o historico completo de funding (backtestavel hoje)
  2. TIRA UM SNAPSHOT dos demais e ACUMULA em disco, para que daqui a alguns
     meses exista historico proprio. Rodando isto 1x/dia via agendador, em
     ~3 meses ha amostra suficiente para testar os outros sinais.

Uso:
  python v7_coleta_alternativa.py --funding        # historico completo
  python v7_coleta_alternativa.py --snapshot       # acumula os de 21 dias
  python v7_coleta_alternativa.py --funding --snapshot
"""
import os
import sys
import time
import glob
import argparse
import requests
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_s, "reconfigure", None)
    if _reconfigure is not None:
        _reconfigure(encoding="utf-8", errors="replace")

BASE = "https://fapi.binance.com"
DEST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_alt")
os.makedirs(DEST, exist_ok=True)


# Universo padrao — usado quando data/ nao existe (ex.: no servidor, que roda
# o coletor mas nao guarda os parquets de preco, que ocupam GBs)
ATIVOS_PADRAO = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
                 "AVAXUSDT", "DOGEUSDT", "LINKUSDT", "ADAUSDT", "DOTUSDT",
                 "LTCUSDT"]


def ativos_disponiveis():
    """Pares de data/ quando existem; senao, o universo padrao."""
    encontrados = sorted(os.path.basename(p).replace("_1m.parquet", "")
                         for p in glob.glob("data/*_1m.parquet"))
    return encontrados or ATIVOS_PADRAO


def baixa_funding(symbol, inicio_ms=1500000000000):
    """
    Historico completo de funding rate, paginado.
    A Binance devolve no maximo 1000 registros por chamada; o funding ocorre
    a cada 8 horas, entao 1000 registros ~ 333 dias.
    """
    linhas, cursor = [], inicio_ms
    while True:
        try:
            r = requests.get(f"{BASE}/fapi/v1/fundingRate",
                             params={"symbol": symbol, "startTime": cursor,
                                     "limit": 1000}, timeout=20)
            r.raise_for_status()
            lote = r.json()
        except Exception as e:
            print(f"    [erro] {symbol}: {e}")
            break
        if not lote:
            break
        linhas += lote
        novo = lote[-1]["fundingTime"] + 1
        if novo <= cursor or len(lote) < 1000:
            break
        cursor = novo
        time.sleep(0.25)          # respeita o limite de requisicoes

    if not linhas:
        return None
    df = pd.DataFrame(linhas)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms")
    df["fundingRate"] = df["fundingRate"].astype(float)
    # Deduplica pelo indice (mesmo idioma usado em v5_live.py e v6_ciclo.py)
    df = df[["fundingTime", "fundingRate"]].set_index("fundingTime").sort_index()
    return df[~df.index.duplicated(keep="last")]


def snapshot(symbol):
    """Uma foto dos indicadores de janela curta, para acumular ao longo do tempo."""
    saidas = {}
    fontes = [
        ("open_interest", "/futures/data/openInterestHist",
         {"period": "1h", "limit": 500}),
        ("long_short", "/futures/data/globalLongShortAccountRatio",
         {"period": "1h", "limit": 500}),
        ("taker", "/futures/data/takerlongshortRatio",
         {"period": "1h", "limit": 500}),
    ]
    for nome, rota, extra in fontes:
        try:
            r = requests.get(BASE + rota, params={"symbol": symbol, **extra},
                             timeout=20)
            r.raise_for_status()
            dados = r.json()
            if isinstance(dados, list) and dados:
                df = pd.DataFrame(dados)
                col_ts = "timestamp" if "timestamp" in df.columns else "time"
                df[col_ts] = pd.to_datetime(df[col_ts], unit="ms")
                saidas[nome] = df.set_index(col_ts).sort_index()
        except Exception as e:
            print(f"    [erro] {symbol}/{nome}: {e}")
        time.sleep(0.25)
    return saidas


def salva_acumulando(df_novo, caminho):
    """Mescla com o que ja existe em disco, sem duplicar timestamps."""
    if os.path.exists(caminho):
        try:
            antigo = pd.read_parquet(caminho)
            df_novo = pd.concat([antigo, df_novo])
            df_novo = df_novo[~df_novo.index.duplicated(keep="last")].sort_index()
        except Exception as e:
            print(f"    [aviso] nao consegui ler {caminho}: {e}")
    df_novo.to_parquet(caminho)
    return len(df_novo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--funding", action="store_true",
                    help="baixa o historico completo de funding rate")
    ap.add_argument("--snapshot", action="store_true",
                    help="acumula open interest / long-short / taker")
    ap.add_argument("--assets", default=None,
                    help="lista separada por virgula (padrao: os de data/)")
    a = ap.parse_args()

    if not (a.funding or a.snapshot):
        ap.error("escolha --funding, --snapshot ou os dois")

    ativos = ([s.strip().upper() for s in a.assets.split(",")] if a.assets
              else ativos_disponiveis())
    print(f"\nColetando para {len(ativos)} ativos -> {DEST}\n")

    if a.funding:
        print("FUNDING RATE (historico completo)")
        for sym in ativos:
            df = baixa_funding(sym)
            if df is None or df.empty:
                print(f"  {sym}: sem dados")
                continue
            caminho = os.path.join(DEST, f"funding_{sym}.parquet")
            n = salva_acumulando(df, caminho)
            print(f"  {sym}: {n} registros | {df.index[0]:%Y-%m-%d} -> "
                  f"{df.index[-1]:%Y-%m-%d} | media {df['fundingRate'].mean()*100:+.4f}%")

    if a.snapshot:
        print("\nSNAPSHOT (open interest / long-short / taker)")
        for sym in ativos:
            saidas = snapshot(sym)
            for nome, df in saidas.items():
                caminho = os.path.join(DEST, f"{nome}_{sym}.parquet")
                n = salva_acumulando(df, caminho)
                print(f"  {sym}/{nome}: +{len(df)} agora, {n} acumulados")

    print(f"\nPronto. Arquivos em {DEST}")
    if a.snapshot:
        print("Rode --snapshot 1x por dia para acumular historico proprio:")
        print("  daqui a ~3 meses havera amostra suficiente para testar esses sinais.")


if __name__ == "__main__":
    main()
