# -*- coding: utf-8 -*-
"""
app/semear.py — cria o histórico local a partir da corretora
=============================================================

O servidor não guarda os parquets de preço — eles ocupam gigabytes e existem
para pesquisa, não para operação. Mas as features precisam de 200 barras
diárias (a SMA200), o que significa ~260 dias de histórico.

Por que barras de 4 horas, e não de 1 minuto
--------------------------------------------
As features do oráculo saem exclusivamente das visões diária e de 4h — a série
fina nunca é lida. Verificado em 04/09/2026: reconstruindo o histórico a partir
de barras de 4h, as 16 features saem IDÊNTICAS (diferença relativa zero), com
240 vezes menos dados. Em 1 minuto, 2.600 dias levariam 3,7 horas de download;
em 4h, são 11 requisições por ativo.

Isso importa porque a alternativa — semear pouco — não é aceitável: treinar com
257 dias derruba a acurácia para 50,07%, que é o mesmo que jogar moeda. Com o
histórico completo ela é 53,69%.

Este script grava em data/, uma vez. Depois disso, dados/atualizar.py completa
incrementalmente a cada execução, no mesmo intervalo.

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

MINIMO_DIAS = 2600     # o suficiente para o modelo valer 53,69%
INTERVALO = "4h"       # ver a nota no cabecalho


def main() -> None:
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), ".env"))

    ap = argparse.ArgumentParser(prog="v9-semear")
    ap.add_argument("--dias", type=int, default=MINIMO_DIAS)
    ap.add_argument("--intervalo", default=INTERVALO,
                    choices=["1m", "15m", "1h", "4h"])
    ap.add_argument("--real", action="store_true")
    a = ap.parse_args()

    if a.dias < MINIMO_DIAS:
        print(f"AVISO: {a.dias} dias. Medido em 04/09/2026, treinar com 257\n"
              f"dias derruba a acuracia para 50,07% — moeda. Com o historico\n"
              f"completo ela e 53,69%. Use pelo menos {MINIMO_DIAS}.")

    from execucao.carteira import CarteiraBinance
    cfg = ler_config()
    ativos = cfg["universo"]["ativos"]

    carteira = CarteiraBinance(ativos, testnet=not a.real, armado=False)
    carteira.conectar()

    os.makedirs(RAIZ_PADRAO, exist_ok=True)
    inicio = datetime.utcnow() - timedelta(days=a.dias)
    print(f"\nBaixando {a.dias} dias em barras de {a.intervalo}, "
          f"{len(ativos)} ativos (desde {inicio:%Y-%m-%d}) -> {RAIZ_PADRAO}\n",
          flush=True)

    for symbol in ativos:
        destino = os.path.join(RAIZ_PADRAO, f"{symbol}_1m.parquet")
        if os.path.exists(destino):
            print(f"  {symbol}: ja existe, pulando "
                  f"(apague para rebaixar)", flush=True)
            continue

        t0 = time.time()
        df = baixar_desde(carteira.api, symbol, inicio,
                          intervalo=a.intervalo,
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
