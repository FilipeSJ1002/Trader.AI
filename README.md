
# 📈 TRADER.AI

> **Sistema de Trading Algorítmico Autônomo Baseado em Análise Técnica Quantitativa.**
>
> *Projeto de Trabalho de Conclusão de Curso (TCC) - Ciência da Computação.*

![Status](https://img.shields.io/badge/Status-Etapa%204%20(Live%20Trading%20&%20Execução)-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![WebSockets](https://img.shields.io/badge/Data-WebSockets%20%7C%20Binance-F3BA2F?style=for-the-badge&logo=binance&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📖 Visão Geral do Projeto

O **TRADER.AI** é uma solução de software desenvolvida para automatizar o processo de tomada de decisão no mercado de criptoativos. 

Diferente de sistemas tradicionais, esta inteligência quantitativa utiliza uma arquitetura modular que visa integrar **Estratégias de Confluência** (união de múltiplos indicadores técnicos) e, futuramente, modelos de **Machine Learning** para mitigar o viés emocional humano e explorar ineficiências de mercado em operações 24/7.

Atualmente, o projeto encontra-se na **Fase 4**, marcando sua transição final para uma inteligência artificial de "Live Trading". Além da conexão persistente com a corretora para avaliar o mercado em tempo real, a IA agora conta com um motor de execução e gerenciamento de risco. Ela pontua a força dos sinais matemáticos e aloca o capital de forma inteligente e autônoma, operando inicialmente no ambiente seguro da Binance Testnet.

---

## 🚀 Funcionalidades da Etapa 4

Nesta quarta etapa, a inteligência artificial adquiriu autonomia operacional para atuar de forma ativa no mercado através da execução automatizada de ordens:

* **Integração Binance Testnet:** Autenticação segura via variáveis de ambiente (`.env`) e operações em ambiente de simulação real sem expor capital verdadeiro.
* **Sistema de Scoring Dinâmico:** O antigo algoritmo de decisões fixas foi substituído por um motor de confluência matricial com base em pesos. O sistema gera uma pontuação de 0 a 100, classificando os sinais em escalas de força (`COMPRA_FORTE`, `COMPRA_MODERADA`, `VENDA_FORTE`, etc.).
* **Execução Autônoma e Gestão de Risco:** Um gerenciamento interno que consome as decisões computacionais e emite ordens de compra/venda inteligentemente baseada na força do sinal, protegendo o saldo.
* **Monitoramento de Saldo:** Capacidade utilitária extraída para auditar saldos da carteira antes e depois das operações na Testnet.

---

## 📂 Estrutura do Projeto

A organização do código segue rigorosamente os padrões de *Clean Code* e separação de responsabilidades:

```text
Trader.AI/
├── main.py                    # Ponto de entrada (API REST e inicialização do WebSocket)
├── execution.py               # Motor autônomo de execução e gestão de risco
├── strategy.py                # Core Matemático: Cálculos e Regras de Confluência com Scoring
├── market_state.py            # Gerenciador de Estado: Memória RAM e histórico
├── binance_stream.py          # Conexão WebSocket em Tempo Real com a Binance
├── teste_saldo.py             # Script utilitário para checagem rápida de saldo
├── data_loader.py             # Carregamento otimizado de arquivos Parquet
├── download_binance_data.py   # ETL: Script de Extração (Download progressivo)
├── processar_dados.py         # ETL: Script de Transformação (Limpeza e Parquet)
├── data/                      # Armazenamento dos datasets históricos (.parquet)
├── .env                       # Variáveis de ambiente seguras (API Keys da Testnet)
├── requirements.txt           # Lista de dependências e bibliotecas
└── README.md                  # Documentação oficial
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
Isolar as dependências é fundamental para o funcionamento dos WebSockets.

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

1. Copie o arquivo `.env.example` e renomeie a cópia para `.env`.
2. Adicione suas credenciais da **Binance Testnet** dentro do `.env` (obrigatório para testar operações de ordens e consultar saldos):
   - `BINANCE_API_KEY=sua_chave_aqui`
   - `BINANCE_SECRET_KEY=sua_chave_secreta_aqui`

### 5. Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## ⚙️ Alimentando a Memória Inicial (ETL)

Antes de iniciar a escuta em tempo real, o sistema precisa carregar o passado para ter contexto matemático.

**Passo 1: Baixar Histórico**
```bash
python download_binance_data.py
```

**Passo 2: Processar para Parquet**
```bash
python processar_dados.py
```

---

## 🧪 Como Rodar a Inteligência em Tempo Real

Com os dados base processados, inicie a aplicação principal:

```bash
uvicorn main:app --reload
```

✅ **Comportamento Esperado:**
1. O servidor carregará o arquivo Parquet na memória RAM (`market_state.py`).
2. O servidor REST ficará disponível em `http://127.0.0.1:8000`.
3. O WebSocket iniciará em segundo plano. Assim que o relógio virar o minuto, você verá as decisões do sistema brotando automaticamente no terminal, e quaisquer execuções de live trading configuradas atuarão de imediato:
   `[TRADER.AI] 📊 Preço: $74434.37 | RSI: 65.20 | Decisão: NEUTRO`

### 💰 Verificando o Saldo da Conta
Para confirmar se a sua conexão com a Binance Testnet foi bem sucedida e acompanhar seu saldo antes e depois do bot operar, você pode rodar (inclusive em outro terminal) o comando utilitário:

```bash
python teste_saldo.py
```

*Nota: O endpoint manual `/analisar_mercado` continua funcional via Insomnia, comunicando-se diretamente com o mesmo gerenciador de estado unificado.*

---

## 📅 Roadmap de Desenvolvimento

O desenvolvimento do TRADER.AI segue um cronograma incremental:

- [x] **Etapa 1:** Arquitetura Base, API e Estratégia de Confluência.
- [x] **Etapa 2:** Integração com Dataset Histórico Real e ETL automatizado.
- [x] **Etapa 3:** Conexão com API Binance via WebSockets e Decisões em Tempo Real.
- [x] **Etapa 4:** Execução de Ordens (Integração de Contas, Scoring e Live Trading).

---

## ✒️ Autor

Desenvolvido por **Filipe Spirlandeli Junqueira**.

---
> Este projeto é estritamente educacional e experimental. O autor não se responsabiliza por perdas financeiras decorrentes do uso deste software em contas reais.
