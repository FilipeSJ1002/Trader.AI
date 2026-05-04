import pandas as pd
import os
import glob

CAMINHOS_PASTAS = {
    "BTCUSDT": r"C:\Users\filip\Downloads\datasets_brutos\BTC-USDT_datasets_brutos",
    "ETHUSDT": r"C:\Users\filip\Downloads\datasets_brutos\ETH-USDT_datasets_brutos",
    "XRPUSDT": r"C:\Users\filip\Downloads\datasets_brutos\XRP-USDT_datasets_brutos"
}

def processar_dados():
    os.makedirs("data", exist_ok=True)

    for moeda, caminho_pasta in CAMINHOS_PASTAS.items():
        print(f"\n--- Processando Par: {moeda} ---")
        print(f"Buscando arquivos CSV em: {caminho_pasta}")
        
        arquivos = glob.glob(os.path.join(caminho_pasta, "*.csv"))
        
        if not arquivos:
            print(f"AVISO: Nenhum arquivo .csv encontrado para {moeda}!")
            continue

        print(f"Encontrados {len(arquivos)} arquivos para {moeda}. Iniciando leitura...")
        
        lista_dfs = []
        
        for arq in arquivos:
            try:
                df = pd.read_csv(arq, header=None)
                
                df = df.iloc[:, :6]
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                
                col_numerica = pd.to_numeric(df['timestamp'], errors='coerce')
                mask_num = col_numerica.notna()
                
                df['date'] = pd.NaT
                
                mask_valid_us = mask_num & (col_numerica > 100000000000000)
                df.loc[mask_valid_us, 'date'] = pd.to_datetime(col_numerica[mask_valid_us], unit='us', errors='coerce')
                
                mask_valid_ms = mask_num & (col_numerica <= 100000000000000) & (col_numerica > 100000000000)
                df.loc[mask_valid_ms, 'date'] = pd.to_datetime(col_numerica[mask_valid_ms], unit='ms', errors='coerce')
                
                mask_valid_s = mask_num & (col_numerica <= 100000000000) & (col_numerica > 0)
                df.loc[mask_valid_s, 'date'] = pd.to_datetime(col_numerica[mask_valid_s], unit='s', errors='coerce')
                
                df.loc[~mask_num, 'date'] = pd.to_datetime(df.loc[~mask_num, 'timestamp'], errors='coerce')
                
                df = df.dropna(subset=['date'])
                
                df = df.drop(columns=['timestamp'])
                
                cols_float = ['open', 'high', 'low', 'close', 'volume']
                for col in cols_float:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                lista_dfs.append(df)
                print(".", end="")
            except Exception as e:
                print(f"\nErro ao ler arquivo {os.path.basename(arq)}: {e}")
        
        if not lista_dfs:
            print(f"\nNenhum dado válido extraído para {moeda}.")
            continue

        print("\n\nConcatenando DataFrames...")
        df_final = pd.concat(lista_dfs, ignore_index=True)
        
        print(f"Linhas brutas ({moeda}): {len(df_final)}")
        
        print("Tratamento de Datas e Ordenação...")
        df_final = df_final.sort_values('date').reset_index(drop=True)
        
        df_final = df_final[df_final['date'].dt.year < 2100]
        
        df_final = df_final[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        arquivo_saida = f"data/{moeda}_1m.parquet"
        print(f"Salvando arquivo otimizado em: {arquivo_saida}")
        
        try:
            df_final.to_parquet(arquivo_saida, index=False)
            print(f"SUCESSO! Base de dados de {moeda} pronta e limpa.")
            print(f"Total de candles válidos: {len(df_final)}")
            print(f"Período: de {df_final['date'].min()} até {df_final['date'].max()}")
        except ImportError:
            print("ERRO: Biblioteca 'pyarrow' ou 'fastparquet' não encontrada.")
            print("Tentando salvar em CSV como alternativa...")
            arquivo_csv = arquivo_saida.replace('.parquet', '.csv')
            df_final.to_csv(arquivo_csv, index=False)
            print(f"Salvo em CSV: {arquivo_csv}")

if __name__ == "__main__":
    processar_dados()