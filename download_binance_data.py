"""
download_binance_data.py — V4
ETL automatizado de dados históricos do Binance Vision.

Pares suportados: BTC, ETH, XRP, SOL, BNB, AVAX
Intervalos:       1m (padrão), 5m, 15m (para backtesting MTF)

Como usar:
  python download_binance_data.py              # baixa 1m de todos os pares
  python download_binance_data.py --interval 5m
  python download_binance_data.py --pair SOLUSDT --interval 1m
"""
import os
import re
import io
import zipfile
import argparse
import requests
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuração dos pares e pastas de destino
# ─────────────────────────────────────────────────────────────────────────────
DATASETS_ROOT = r"C:\Users\filip\Downloads\datasets_brutos"

PAIRS_CONFIG = {
    "BTCUSDT":  os.path.join(DATASETS_ROOT, "BTC-USDT_datasets_brutos"),
    "ETHUSDT":  os.path.join(DATASETS_ROOT, "ETH-USDT_datasets_brutos"),
    "XRPUSDT":  os.path.join(DATASETS_ROOT, "XRP-USDT_datasets_brutos"),
    "SOLUSDT":  os.path.join(DATASETS_ROOT, "SOL-USDT_datasets_brutos"),
    "BNBUSDT":  os.path.join(DATASETS_ROOT, "BNB-USDT_datasets_brutos"),
    "AVAXUSDT": os.path.join(DATASETS_ROOT, "AVAX-USDT_datasets_brutos"),
    # V6: +5 pares liquidos para ampliar o volume de oportunidades
    "DOGEUSDT": os.path.join(DATASETS_ROOT, "DOGE-USDT_datasets_brutos"),
    "LINKUSDT": os.path.join(DATASETS_ROOT, "LINK-USDT_datasets_brutos"),
    "ADAUSDT":  os.path.join(DATASETS_ROOT, "ADA-USDT_datasets_brutos"),
    "DOTUSDT":  os.path.join(DATASETS_ROOT, "DOT-USDT_datasets_brutos"),
    "LTCUSDT":  os.path.join(DATASETS_ROOT, "LTC-USDT_datasets_brutos"),
}

# Data de início disponível no Binance Vision por par
# V5: estendido para cobrir mais ciclos de mercado (bull 2019-2021 + bear 2022)
PAIR_START = {
    "BTCUSDT":  (2019, 1),   # BTC desde 2019 — inclui bull 2020-2021 + bear 2022
    "ETHUSDT":  (2019, 1),   # ETH desde 2019
    "XRPUSDT":  (2019, 1),   # XRP desde 2019
    "SOLUSDT":  (2020, 9),   # SOL listado em Set/2020
    "BNBUSDT":  (2019, 1),   # BNB desde 2019
    "AVAXUSDT": (2020, 9),   # AVAX listado em Set/2020
    # V6
    "DOGEUSDT": (2019, 7),   # DOGE listado em Jul/2019
    "LINKUSDT": (2019, 1),   # LINK desde Jan/2019
    "ADAUSDT":  (2019, 1),   # ADA desde 2018 (Vision a partir de 2019)
    "DOTUSDT":  (2020, 8),   # DOT listado em Ago/2020
    "LTCUSDT":  (2019, 1),   # LTC desde 2017 (padronizado em 2019)
}


def get_latest_downloaded_month(folder: str, symbol: str, interval: str) -> tuple[int, int]:
    """Detecta o mês mais recente já baixado para evitar redownload."""
    os.makedirs(folder, exist_ok=True)

    pattern = re.compile(
        rf"^{symbol}-{interval}-(?P<year>\d{{4}})-(?P<month>\d{{2}})\.csv$"
    )

    latest_year, latest_month = PAIR_START.get(symbol, (2021, 12))
    # recua um mês para forçar o próximo download começar deste
    if latest_month == 1:
        latest_year -= 1
        latest_month = 12
    else:
        latest_month -= 1

    has_files = False
    for f in os.listdir(folder):
        m = pattern.match(f)
        if m:
            has_files = True
            y, mo = int(m.group("year")), int(m.group("month"))
            if y > latest_year or (y == latest_year and mo > latest_month):
                latest_year, latest_month = y, mo

    if not has_files:
        start_y, start_m = PAIR_START.get(symbol, (2021, 12))
        # Recua um mês para que download_pair inicie exatamente no mês de início
        if start_m > 1:
            return start_y, start_m - 1
        else:
            return start_y - 1, 12

    return latest_year, latest_month


def download_pair(symbol: str, interval: str = "1m"):
    """Baixa todos os meses faltantes para um par/intervalo."""
    folder = PAIRS_CONFIG[symbol]
    os.makedirs(folder, exist_ok=True)

    now = datetime.now()
    # Limite: mês anterior (dados do mês atual não estão completos)
    if now.month == 1:
        target_year, target_month = now.year - 1, 12
    else:
        target_year, target_month = now.year, now.month - 1

    last_y, last_m = get_latest_downloaded_month(folder, symbol, interval)

    # Próximo mês a baixar
    if last_m == 12:
        curr_y, curr_m = last_y + 1, 1
    else:
        curr_y, curr_m = last_y, last_m + 1

    print(f"\n[{symbol}/{interval}] Ultimo arquivo: {last_y}-{last_m:02d} | Baixando a partir de {curr_y}-{curr_m:02d}...")

    baixados = 0
    while (curr_y < target_year) or (curr_y == target_year and curr_m <= target_month):
        url = (
            f"https://data.binance.vision/data/spot/monthly/klines/"
            f"{symbol}/{interval}/{symbol}-{interval}-{curr_y}-{curr_m:02d}.zip"
        )

        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    csvs = [f for f in z.namelist() if f.endswith('.csv')]
                    if csvs:
                        for csv_file in csvs:
                            z.extract(csv_file, path=folder)
                        print(f"  [OK] {symbol}-{interval}-{curr_y}-{curr_m:02d}.csv extraido")
                        baixados += 1
                    else:
                        print(f"  [WARN] ZIP sem .csv: {url}")
            elif r.status_code == 404:
                print(f"  [404] {curr_y}-{curr_m:02d} ainda nao disponivel na Binance Vision.")
            else:
                print(f"  [HTTP {r.status_code}] {url}")
        except Exception as e:
            print(f"  [ERRO] {e}")

        if curr_m == 12:
            curr_y += 1
            curr_m = 1
        else:
            curr_m += 1

    print(f"[{symbol}/{interval}] Concluido. {baixados} arquivo(s) novos baixados.")
    return baixados


def run_etl(pairs: list[str] = None, interval: str = "1m"):
    """Executa o ETL para os pares e intervalo especificados."""
    if pairs is None:
        pairs = list(PAIRS_CONFIG.keys())

    now = datetime.now()
    print("=" * 60)
    print(f"ETL Trader.AI V4 — Binance Vision")
    print(f"Intervalo: {interval} | Pares: {', '.join(pairs)}")
    print(f"Data limite: {now.year}-{now.month - 1:02d}")
    print("=" * 60)

    total = 0
    for symbol in pairs:
        if symbol not in PAIRS_CONFIG:
            print(f"[WARN] Par desconhecido: {symbol}. Ignorando.")
            continue
        total += download_pair(symbol, interval)

    print(f"\nETL concluido. Total de arquivos baixados: {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download de dados historicos Binance Vision — V4")
    parser.add_argument(
        "--pair", type=str, default=None,
        help="Par especifico (ex: SOLUSDT). Padrao: todos os pares."
    )
    parser.add_argument(
        "--interval", type=str, default="1m",
        choices=["1m", "5m", "15m", "1h"],
        help="Intervalo de tempo. Padrao: 1m"
    )
    args = parser.parse_args()

    pairs = [args.pair.upper()] if args.pair else None
    run_etl(pairs=pairs, interval=args.interval)
