# -*- coding: utf-8 -*-
"""
app/semear.py — cria o histórico local a partir da corretora
=============================================================

O servidor não guarda os parquets de preço — eles ocupam gigabytes e existem
para pesquisa, não para operação. Mas as features precisam de 200 barras
diárias (a SMA200), o que significa ~260 dias de histórico.

Este script baixa esse mínimo e grava em data/, uma vez. Depois disso,
dados/atualizar.py completa incrementalmente a cada execução.

Uso:
  python -m app.semear              # 260 dias, o minimo para a SMA200
  python -m app.semear --dias 400   # com folga
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta

for _s in (sys.stdout, sys.stderr):
    _r = getattr(_s, "reconfigure", None)
    if _r is not None:
        _r(encoding="utf-8", errors="replace")

from dotenv import load_dotenv

from dados.atualizar import baixar_desde
from dados.fonte import RAIZ_PADRAO, ler_config

MINIMO_DIAS = 260      # SMA200 diaria + folga de aquecimento


def main() -> None:
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), ".env"))

    ap = argparse.ArgumentParser(prog="v9-semear")
    ap.add_argument("--dias", type=int, default=MINIMO_DIAS)
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()

    if a.dias < MINIMO_DIAS:
        print(f"AVISO: {a.dias} dias e menos que os {MINIMO_DIAS} necessarios "
              f"para a SMA200. As features vao sair incompletas.")

    from execucao.carteira import CarteiraBinance
    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]

    carteira = CarteiraBinance(ativos, testnet=not a.real, armado=False)
    carteira.conectar()

    os.makedirs(RAIZ_PADRAO, exist_ok=True)
    inicio = datetime.utcnow() - timedelta(days=a.dias)
    print(f"\nBaixando {a.dias} dias de {len(ativos)} ativos "
          f"(desde {inicio:%Y-%m-%d}) -> {RAIZ_PADRAO}\n", flush=True)

    for symbol in ativos:
        destino = os.path.join(RAIZ_PADRAO, f"{symbol}_1m.parquet")
        if os.path.exists(destino):
            print(f"  {symbol}: ja existe, pulando "
                  f"(apague para rebaixar)", flush=True)
            continue

        t0 = time.time()
        df = baixar_desde(carteira.api, symbol, inicio,
                          log=lambda m: print(f"  {m}", flush=True))
        if df is None or df.is_empty():
            print(f"  {symbol}: FALHOU — nada retornado")
            continue

        # Grava no formato que as V1..V8 usam, para o carregador ler igual.
        (df.rename({"ts": "date", "abertura": "open", "maxima": "high",
                    "minima": "low", "fechamento": "close"})
           .write_parquet(destino))
        print(f"  {symbol}: {len(df):,} barras "
              f"({df['ts'][0]:%Y-%m-%d} -> {df['ts'][-1]:%Y-%m-%d}) "
              f"em {time.time()-t0:.0f}s", flush=True)

    print(f"\nPronto. Agora rode:  python -m app.treinar_oraculo")


if __name__ == "__main__":
    main()
