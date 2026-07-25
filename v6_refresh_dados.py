# -*- coding: utf-8 -*-
"""
v6_refresh_dados.py — Trader.AI V6: completa os parquets ate AGORA via REST
============================================================================

O ETL mensal (Binance Vision) so publica meses COMPLETOS — em 25/07 os dados
param em 30/06. Este script le a ultima data de cada parquet e busca o que
falta pela API publica de klines (sem API key), candle a candle, ate o minuto
fechado mais recente.

Uso:
  python v6_refresh_dados.py            # todos os pares do processar_dados
  python v6_refresh_dados.py --pair DOGEUSDT
"""
import os
import sys
import time
import argparse
import requests
import pandas as pd

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from processar_dados import CAMINHOS_PASTAS   # lista canonica de pares

API = "https://api.binance.com/api/v3/klines"
DATA_DIR = "data"


def topup(sym: str) -> None:
    path = os.path.join(DATA_DIR, f"{sym}_1m.parquet")
    if not os.path.exists(path):
        print(f"  [{sym}] parquet nao existe — rode processar_dados.py antes. Pulando.")
        return

    df = pd.read_parquet(path)
    if "date" not in df.columns:
        print(f"  [{sym}] formato inesperado (sem coluna 'date'). Pulando.")
        return

    last = pd.to_datetime(df["date"].max())
    start_ms = int((last.value // 10**6) + 60_000)   # proximo minuto
    print(f"  [{sym}] parquet ate {last} — buscando o restante...", flush=True)

    rows = []
    while True:
        r = requests.get(API, params={
            "symbol": sym, "interval": "1m",
            "startTime": start_ms, "limit": 1000,
        }, timeout=20)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 60_000
        if len(batch) < 1000:
            break
        time.sleep(0.15)   # gentileza com o rate limit

    if not rows:
        print(f"  [{sym}] ja estava atualizado.")
        return

    novo = pd.DataFrame(rows).iloc[:, :6]
    novo.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    novo["date"] = pd.to_datetime(novo["timestamp"], unit="ms")
    novo = novo.drop(columns=["timestamp"])
    for c in ["open", "high", "low", "close", "volume"]:
        novo[c] = pd.to_numeric(novo[c], errors="coerce")
    novo = novo[["date", "open", "high", "low", "close", "volume"]].dropna()
    novo = novo.iloc[:-1]   # descarta o candle em formacao

    combinado = pd.concat([df, novo], ignore_index=True)
    combinado = (combinado.drop_duplicates(subset="date", keep="last")
                          .sort_values("date").reset_index(drop=True))
    combinado.to_parquet(path, index=False)
    print(f"  [{sym}] +{len(novo)} candles -> agora ate {combinado['date'].max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default=None, help="Par especifico (padrao: todos)")
    a = ap.parse_args()

    pares = [a.pair.upper()] if a.pair else list(CAMINHOS_PASTAS.keys())
    print(f"=== Refresh via REST — {len(pares)} par(es) ===")
    for sym in pares:
        try:
            topup(sym)
        except Exception as e:
            print(f"  [{sym}] ERRO: {e} — seguindo para o proximo.")
    print("=== Refresh concluido ===")


if __name__ == "__main__":
    main()
