import os
import re
import io
import zipfile
import requests
from datetime import datetime

def get_latest_downloaded_month(folder_path: str, symbol: str) -> tuple[int, int]:
    """
    Verifica a pasta do par e retorna o ano e mês do arquivo mais recente baixado.
    Se a pasta estiver vazia ou não existir, retorna (2021, 12) para que
    o próximo download inicie em (2022, 1).
    """
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Diretório criado: {folder_path}")
        except Exception as e:
            print(f"Erro ao criar o diretório {folder_path}: {e}")
            return 2021, 12

    latest_year = 2021
    latest_month = 12

    # Padrão esperado do nome do arquivo (ex: BTCUSDT-1m-2025-12.csv)
    pattern = re.compile(rf"^{symbol}-1m-(?P<year>\d{{4}})-(?P<month>\d{{2}})\.csv$")

    has_files = False
    try:
        for file in os.listdir(folder_path):
            match = pattern.match(file)
            if match:
                has_files = True
                y = int(match.group("year"))
                m = int(match.group("month"))
                if y > latest_year or (y == latest_year and m > latest_month):
                    latest_year = y
                    latest_month = m
    except Exception as e:
        print(f"Erro ao ler o diretório {folder_path}: {e}")

    if not has_files:
        return 2021, 12

    return latest_year, latest_month

def run_etl():
    """
    Função principal que itera sobre os pares, verifica o último mês baixado,
    e faz o download/extração dos próximos meses até o mês passado.
    """
    base_folders = {
        "BTCUSDT": r"C:\Users\filip\Downloads\datasets_brutos\BTC-USDT_datasets_brutos",
        "ETHUSDT": r"C:\Users\filip\Downloads\datasets_brutos\ETH-USDT_datasets_brutos",
        "XRPUSDT": r"C:\Users\filip\Downloads\datasets_brutos\XRP-USDT_datasets_brutos"
    }

    current_date = datetime.now()
    # O limite é o mês anterior ao atual
    if current_date.month == 1:
        target_year = current_date.year - 1
        target_month = 12
    else:
        target_year = current_date.year
        target_month = current_date.month - 1

    print("=== ETL de Dados Históricos Binance Vision ===")
    print(f"Data limite para download: {target_year}-{target_month:02d}")

    for symbol, folder in base_folders.items():
        print(f"\n--- Processando Par: {symbol} ---")
        
        last_y, last_m = get_latest_downloaded_month(folder, symbol)
        
        # Calcular o próximo mês a ser baixado
        if last_m == 12:
            curr_y = last_y + 1
            curr_m = 1
        else:
            curr_y = last_y
            curr_m = last_m + 1
            
        if last_y == 2021 and last_m == 12:
            print(f"Nenhum arquivo encontrado em {folder}. Iniciando download em 2022-01.")
        else:
            print(f"Último arquivo encontrado: {last_y}-{last_m:02d}. Iniciando download em {curr_y}-{curr_m:02d}.")

        # Iterar mês a mês até a data alvo
        while (curr_y < target_year) or (curr_y == target_year and curr_m <= target_month):
            year_str = str(curr_y)
            month_str = f"{curr_m:02d}"
            
            url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/1m/{symbol}-1m-{year_str}-{month_str}.zip"
            print(f"[*] Baixando: {symbol} de {year_str}-{month_str}...")
            
            try:
                response = requests.get(url, timeout=30)
                
                if response.status_code == 200:
                    try:
                        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                            # Filtra apenas os arquivos CSV
                            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                            if csv_files:
                                for csv_file in csv_files:
                                    # Extrai o arquivo diretamente para a pasta destino
                                    z.extract(csv_file, path=folder)
                                print(f"  [+] Sucesso! Arquivo extraído em: {folder}")
                            else:
                                print("  [-] O ZIP foi baixado, mas não continha arquivos .csv.")
                    except zipfile.BadZipFile:
                        print("  [!] Erro: O arquivo baixado não é um ZIP válido.")
                        
                elif response.status_code == 404:
                    print("  [-] Erro 404: Arquivo não encontrado. A Binance provavelmente não o disponibilizou ainda.")
                else:
                    print(f"  [!] Erro HTTP {response.status_code} ao tentar acessar a URL.")
            except Exception as e:
                print(f"  [!] Erro na requisição (timeout/conexão): {e}")
                
            # Avançar para o próximo mês
            if curr_m == 12:
                curr_y += 1
                curr_m = 1
            else:
                curr_m += 1

if __name__ == "__main__":
    run_etl()
