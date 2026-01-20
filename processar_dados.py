import pandas as pd
import os
import glob

# --- CONFIGURAÇÃO ---
# Verifique se o caminho continua correto
CAMINHO_PASTA_CSVS = r"C:\Users\filip\Downloads\datasets_brutos\BTC-USDT_datasets_brutos"
ARQUIVO_SAIDA = "data/BTCUSDT_real_1m_2022_2025.parquet"

def processar_dados():
    print(f"Buscando arquivos CSV em: {CAMINHO_PASTA_CSVS}")
    # Busca recursiva para garantir
    arquivos = glob.glob(os.path.join(CAMINHO_PASTA_CSVS, "*.csv"))
    
    if not arquivos:
        print("ERRO: Nenhum arquivo .csv encontrado!")
        return

    print(f"Encontrados {len(arquivos)} arquivos. Iniciando leitura...")
    
    lista_dfs = []
    
    for arq in arquivos:
        try:
            # header=None assume que não tem cabeçalho.
            df = pd.read_csv(arq, header=None)
            
            # Pega as 6 primeiras colunas
            df = df.iloc[:, :6]
            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # --- LIMPEZA PREVENTIVA ---
            # Força a coluna timestamp a ser numérica. Se tiver texto (cabeçalho), vira NaN
            df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            
            # Remove linhas onde o timestamp virou NaN (ou seja, eram lixo/cabeçalho)
            df = df.dropna(subset=['timestamp'])
            
            # Otimização: converte colunas de preço para float para economizar memória
            cols_float = ['open', 'high', 'low', 'close', 'volume']
            for col in cols_float:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            lista_dfs.append(df)
            print(".", end="") # Feedback visual de progresso
        except Exception as e:
            print(f"\nErro ao ler arquivo {os.path.basename(arq)}: {e}")
    
    print("\n\nConcatenando DataFrames...")
    df_final = pd.concat(lista_dfs, ignore_index=True)
    
    print(f"Linhas brutas: {len(df_final)}")
    
    # --- FILTRO DE SEGURANÇA ---
    print("Filtrando datas inválidas...")
    # Remove qualquer timestamp que seja absurdamente grande (maior que o ano 2100)
    # 4102444800000 é aprox o ano 2100 em milissegundos
    df_final = df_final[df_final['timestamp'] < 4102444800000]
    
    print("Convertendo Timestamps...")
    df_final['date'] = pd.to_datetime(df_final['timestamp'], unit='ms')
    
    # Ordena e reseta o índice
    df_final = df_final.sort_values('date').reset_index(drop=True)
    
    # Reorganiza colunas
    df_final = df_final[['date', 'open', 'high', 'low', 'close', 'volume']]
    
    print(f"Salvando arquivo otimizado em: {ARQUIVO_SAIDA}")
    
    # Cria a pasta data se não existir
    os.makedirs("data", exist_ok=True)
    
    # Salva em Parquet (requer pyarrow ou fastparquet instalado)
    try:
        df_final.to_parquet(ARQUIVO_SAIDA, index=False)
        print("SUCESSO! Base de dados pronta e limpa.")
        print(f"Total de candles válidos: {len(df_final)}")
        print(f"Período: de {df_final['date'].min()} até {df_final['date'].max()}")
    except ImportError:
        print("ERRO: Biblioteca 'pyarrow' ou 'fastparquet' não encontrada.")
        print("Tentando salvar em CSV como alternativa...")
        ARQUIVO_CSV = ARQUIVO_SAIDA.replace('.parquet', '.csv')
        df_final.to_csv(ARQUIVO_CSV, index=False)
        print(f"Salvo em CSV: {ARQUIVO_CSV}")

if __name__ == "__main__":
    processar_dados()