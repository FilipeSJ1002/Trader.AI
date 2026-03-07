import pandas as pd
import os
import glob

# --- CONFIGURAÇÃO ---
# Dicionário mapeando as moedas para suas pastas locais de CSVs
CAMINHOS_PASTAS = {
    "BTCUSDT": r"C:\Users\filip\Downloads\datasets_brutos\BTC-USDT_datasets_brutos",
    "ETHUSDT": r"C:\Users\filip\Downloads\datasets_brutos\ETH-USDT_datasets_brutos",
    "XRPUSDT": r"C:\Users\filip\Downloads\datasets_brutos\XRP-USDT_datasets_brutos"
}

def processar_dados():
    # Cria a pasta data se não existir
    os.makedirs("data", exist_ok=True)

    for moeda, caminho_pasta in CAMINHOS_PASTAS.items():
        print(f"\n--- Processando Par: {moeda} ---")
        print(f"Buscando arquivos CSV em: {caminho_pasta}")
        
        # Busca recursiva para garantir
        arquivos = glob.glob(os.path.join(caminho_pasta, "*.csv"))
        
        if not arquivos:
            print(f"AVISO: Nenhum arquivo .csv encontrado para {moeda}!")
            continue

        print(f"Encontrados {len(arquivos)} arquivos para {moeda}. Iniciando leitura...")
        
        lista_dfs = []
        
        for arq in arquivos:
            try:
                # header=None assume que não tem cabeçalho.
                df = pd.read_csv(arq, header=None)
                
                # Pega as 6 primeiras colunas
                df = df.iloc[:, :6]
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                
        # --- LIMPEZA PREVENTIVA ---
                # Tenta converter o que for número (Unix ms)
                col_numerica = pd.to_numeric(df['timestamp'], errors='coerce')
                mask_num = col_numerica.notna()
                
                # Inicializa a coluna date oficial
                df['date'] = pd.NaT
                
                # 1. Onde for número e for um valor válido no tempo, converte assumindo a unidade adequada
                # Microsegundos (16+ dígitos - Binance a partir de 2025)
                mask_valid_us = mask_num & (col_numerica > 100000000000000)
                df.loc[mask_valid_us, 'date'] = pd.to_datetime(col_numerica[mask_valid_us], unit='us', errors='coerce')
                
                # Milissegundos (Binance tradicional - ~13 dígitos)
                mask_valid_ms = mask_num & (col_numerica <= 100000000000000) & (col_numerica > 100000000000)
                df.loc[mask_valid_ms, 'date'] = pd.to_datetime(col_numerica[mask_valid_ms], unit='ms', errors='coerce')
                
                # Para números pequenos (já em segundos) ou outras sujeiras
                mask_valid_s = mask_num & (col_numerica <= 100000000000) & (col_numerica > 0)
                df.loc[mask_valid_s, 'date'] = pd.to_datetime(col_numerica[mask_valid_s], unit='s', errors='coerce')
                
                # 2. Onde NÃO for número (ex: strings de 2025 ou cabeçalhos), tenta o parse padrão
                df.loc[~mask_num, 'date'] = pd.to_datetime(df.loc[~mask_num, 'timestamp'], errors='coerce')
                
                # 3. Removemos apenas o que sobrou como NaT (lixos reais e cabeçalhos inúteis)
                df = df.dropna(subset=['date'])
                
                # Após isso, pode remover a coluna 'timestamp' antiga
                df = df.drop(columns=['timestamp'])
                
                # Otimização: converte colunas de preço para float para economizar memória
                cols_float = ['open', 'high', 'low', 'close', 'volume']
                for col in cols_float:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                lista_dfs.append(df)
                print(".", end="") # Feedback visual de progresso
            except Exception as e:
                print(f"\nErro ao ler arquivo {os.path.basename(arq)}: {e}")
        
        if not lista_dfs:
            print(f"\nNenhum dado válido extraído para {moeda}.")
            continue

        print("\n\nConcatenando DataFrames...")
        df_final = pd.concat(lista_dfs, ignore_index=True)
        
        print(f"Linhas brutas ({moeda}): {len(df_final)}")
        
        print("Tratamento de Datas e Ordenação...")
        # Ordena e reseta o índice
        df_final = df_final.sort_values('date').reset_index(drop=True)
        
        # Filtro de segurança extra para evitar datas absurdas > ano 2100 e NaN
        df_final = df_final[df_final['date'].dt.year < 2100]
        
        # Reorganiza colunas
        df_final = df_final[['date', 'open', 'high', 'low', 'close', 'volume']]
        
        arquivo_saida = f"data/{moeda}_1m.parquet"
        print(f"Salvando arquivo otimizado em: {arquivo_saida}")
        
        # Salva em Parquet (requer pyarrow ou fastparquet instalado)
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