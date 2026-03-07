# 📈 TRADER.AI

> **Sistema de Trading Algorítmico Autônomo Baseado em Análise Técnica Quantitativa.**
>
> *Projeto de Trabalho de Conclusão de Curso (TCC) - Ciência da Computação.*

![Status](https://img.shields.io/badge/Status-Etapa%202%20(Dados%20Reais%20e%20ETL)-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Pandas](https://img.shields.io/badge/Data-Pandas%20%7C%20PyArrow-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📖 Visão Geral do Projeto

O **TRADER.AI** é uma solução de software desenvolvida para automatizar o processo de tomada de decisão no mercado de criptoativos. 

Diferente de sistemas tradicionais baseados em regras simples, esta inteligência quantitativa utiliza uma arquitetura modular que visa integrar **Estratégias de Confluência** (união de múltiplos indicadores técnicos) e, futuramente, modelos de **Machine Learning** para mitigar o viés emocional humano e explorar ineficiências de mercado em operações 24/7.

Atualmente, o projeto encontra-se na **Fase 2**, onde o sistema abandonou dados simulados e passou a consumir **dados históricos reais de 1 minuto** do mercado SPOT (BTC/USDT, ETH/USDT, XRP/USDT). O projeto agora conta com um Pipeline de Dados (ETL) próprio e automatizado.

## 🚀 Funcionalidades da Etapa 2

Nesta segunda etapa, a arquitetura foi expandida para suportar grandes volumes de dados reais:

* **Pipeline ETL Automatizado:** Scripts integrados para extração, transformação e carregamento de dados históricos diretamente da Binance Vision.
* **Suporte Multi-Moedas:** O sistema agora processa e padroniza o histórico do Bitcoin (BTC), Ethereum (ETH) e Ripple (XRP).
* **Armazenamento Otimizado:** Transição de arquivos `.csv` brutos para o formato `.parquet` via `pyarrow`, garantindo leitura em milissegundos pela API.
* **Tratamento de Anomalias:** Lógica de limpeza avançada para lidar com inconsistências de *timestamps* (conversão de milissegundos e microssegundos) na base de dados da corretora.
* **Segurança e Isolamento:** Implementação de variáveis de ambiente (`.env`) para proteção de dados sensíveis e adoção de ambiente virtual (`venv`).

## 📂 Estrutura do Projeto

A organização do código segue os padrões de *Clean Code* e modularidade:

```text
Trader.AI/
├── main.py                    # Ponto de entrada da API e rotas
├── strategy.py                # Core: Cálculos dos indicadores e Regras de Trade
├── data_loader.py             # Módulo de carregamento otimizado de arquivos Parquet
├── download_binance_data.py   # Script de Extração (Download progressivo da Binance)
├── processar_dados.py         # Script de Transformação (Limpeza e conversão para Parquet)
├── data/                      # Diretório de armazenamento dos dados refinados (.parquet)
├── .env.example               # Template seguro de variáveis de ambiente
├── requirements.txt           # Lista de dependências do projeto
├── README.md                  # Documentação oficial
└── .gitignore                 # Arquivos ignorados pelo controle de versão
```

---

## 💻 Instalação e Execução

Siga os passos abaixo para preparar a infraestrutura do sistema.

### 1. Pré-requisitos

* Python 3.10 ou superior instalado.
* Git instalado.

### 2. Clonar o Repositório

```bash
git clone https://github.com/SEU_USUARIO/Trader.AI.git
cd Trader.AI
```

### 3. Configurar o Ambiente Virtual (Obrigatório)

Isolar as dependências é fundamental para o funcionamento do pipeline de dados.

No Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```

No Linux ou macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Configurar Variáveis de Ambiente

Duplique o arquivo de exemplo e renomeie-o:

1. Copie o arquivo `.env.example`.
2. Renomeie a cópia para `.env`.
3. O sistema já vem configurado para ler o arquivo `data/BTCUSDT_1m.parquet` por padrão.

### 5. Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## ⚙️ Alimentando a Base de Dados (ETL)

Antes de iniciar o servidor, você precisa baixar e processar o histórico real de mercado.

**Passo 1: Extração (Download)**
Este comando varre os arquivos locais e baixa automaticamente os meses faltantes da corretora.

```bash
python download_binance_data.py
```

**Passo 2: Transformação (Limpeza e Parquet)**
Este comando limpa os CSVs baixados, corrige formatações de data e gera os arquivos `.parquet` finais na pasta `data/`.

```bash
python processar_dados.py
```

---

## 🧪 Como Realizar Testes na API

Com os dados processados, inicie a inteligência quantitativa:

```bash
uvicorn main:app --reload
```

✅ Sucesso: O servidor estará rodando em `http://127.0.0.1:8000`.

**Exemplo de Requisição (Input)**
Faça uma requisição do tipo `POST` para o endpoint `/analisar_mercado`.

Corpo (JSON): Simulando a chegada de um novo preço de mercado.

```json
{
  "price": 95000.00,
  "volume": 3500.50
}
```

**Exemplo de Resposta (Output)**
O TRADER.AI processará o novo preço contra todo o histórico real carregado e retornará a decisão:

```json
{
	"decision": "VENDA FORTE",
	"analysis": {
		"rsi": 88.5,
		"bollinger_position": "UPPER_BAND_BREAKOUT",
		"macd_signal": "BULLISH",
		"trend": "BULLISH (Price > EMA200)",
        "close_price": 95000.0,
        "ema_200": 93826.81
	},
	"timestamp": "2026-03-05T20:45:00"
}
```

---

## 📅 Roadmap de Desenvolvimento

O desenvolvimento do TRADER.AI segue um cronograma incremental:

* [x] **Etapa 1:** Arquitetura Base, API e Estratégia de Confluência.
* [x] **Etapa 2:** Integração com Dataset Histórico Real e ETL Multi-Moedas automatizado.
* [ ] **Etapa 3:** Conexão com API Binance (Leitura de mercado em Tempo Real).
* [ ] **Etapa 4:** Execução de Ordens (Live Trading e Gestão de Risco Automatizada).

## ✒️ Autor

Desenvolvido por **Filipe Spirlandeli Junqueira**.

---

> Este projeto é estritamente educacional e experimental. O autor não se responsabiliza por perdas financeiras decorrentes do uso deste software em contas reais.
