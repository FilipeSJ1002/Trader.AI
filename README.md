
# 📈 TRADER.AI

> **Sistema de Trading Algorítmico Autônomo Baseado em Análise Técnica Quantitativa.**
>
> *Projeto de Trabalho de Conclusão de Curso (TCC) - Ciência da Computação.*

![Status](https://img.shields.io/badge/Status-Etapa%203%20(Tempo%20Real%20&%20WebSockets)-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![WebSockets](https://img.shields.io/badge/Data-WebSockets%20%7C%20Binance-F3BA2F?style=for-the-badge&logo=binance&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

## 📖 Visão Geral do Projeto

O **TRADER.AI** é uma solução de software desenvolvida para automatizar o processo de tomada de decisão no mercado de criptoativos. 

Diferente de sistemas tradicionais, esta inteligência quantitativa utiliza uma arquitetura modular que visa integrar **Estratégias de Confluência** (união de múltiplos indicadores técnicos) e, futuramente, modelos de **Machine Learning** para mitigar o viés emocional humano e explorar ineficiências de mercado em operações 24/7.

Atualmente, o projeto encontra-se na **Fase 3**, onde o sistema atingiu seu status de "tempo real". Ele agora possui uma conexão persistente e assíncrona com a corretora Binance, permitindo que a inteligência avalie o mercado global e tome decisões instantâneas a cada fechamento de vela (1 minuto).

---

## 🚀 Funcionalidades da Etapa 3

Nesta terceira etapa, a arquitetura foi refatorada para suportar fluxos de dados contínuos sem perda de performance:

* **Conexão WebSocket Assíncrona:** Integração em tempo real com a Binance utilizando a biblioteca `python-binance` e `asyncio`, rodando em paralelo ao servidor FastAPI sem bloqueios de *Event Loop*.
* **Processamento Contínuo:** A cada fechamento de vela (`1m`), o sistema extrai o novo preço e volume, recalculando instantaneamente indicadores complexos (RSI, MACD, Bollinger Bands, EMA 200).
* **Gerenciamento de Estado Centralizado:** Implementação do módulo `market_state.py`, que mantém um histórico fixo e seguro em memória (exatamente 10.000 velas), prevenindo vazamentos de memória e importações circulares.
* **Decisões ao Vivo:** O motor de confluência agora imprime decisões matemáticas (COMPRA FORTE, VENDA FORTE, NEUTRO) diretamente no terminal em milissegundos após o fechamento do mercado.

---

## 📂 Estrutura do Projeto

A organização do código segue rigorosamente os padrões de *Clean Code* e separação de responsabilidades:

```text
Trader.AI/
├── main.py                    # Ponto de entrada (API REST e inicialização do WebSocket)
├── strategy.py                # Core Matemático: Cálculos e Regras de Confluência
├── market_state.py            # Gerenciador de Estado: Memória RAM e histórico
├── binance_stream.py          # Módulo de Conexão WebSocket em Tempo Real
├── data_loader.py             # Carregamento otimizado de arquivos Parquet
├── download_binance_data.py   # ETL: Script de Extração (Download progressivo)
├── processar_dados.py         # ETL: Script de Transformação (Limpeza e Parquet)
├── data/                      # Armazenamento dos datasets históricos (.parquet)
├── .env.example               # Template de variáveis de ambiente seguras
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

Duplique o arquivo de exemplo e renomeie-o:

1. Copie o arquivo `.env.example`.
2. Renomeie a cópia para `.env`.

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

Com os dados base processados, inicie a aplicação:

```bash
uvicorn main:app --reload
```

✅ **Comportamento Esperado:**
1. O servidor carregará o arquivo Parquet na memória RAM (`market_state.py`).
2. O servidor REST ficará disponível em `http://127.0.0.1:8000`.
3. O WebSocket iniciará em segundo plano. Assim que o relógio virar o minuto, você verá as decisões do sistema brotando automaticamente no terminal:
   `[TRADER.AI] 📊 Preço: $74434.37 | RSI: 65.20 | Decisão: NEUTRO`

*Nota: O endpoint manual `/analisar_mercado` continua funcional via Insomnia, comunicando-se diretamente com o mesmo gerenciador de estado unificado.*

---

## 📅 Roadmap de Desenvolvimento

O desenvolvimento do TRADER.AI segue um cronograma incremental:

- [x] **Etapa 1:** Arquitetura Base, API e Estratégia de Confluência.
- [x] **Etapa 2:** Integração com Dataset Histórico Real e ETL automatizado.
- [x] **Etapa 3:** Conexão com API Binance via WebSockets e Decisões em Tempo Real.
- [ ] **Etapa 4:** Execução de Ordens (Integração de Contas, Live Trading e Gestão de Risco).

---

## ✒️ Autor

Desenvolvido por **Filipe Spirlandeli Junqueira**.

---
> Este projeto é estritamente educacional e experimental. O autor não se responsabiliza por perdas financeiras decorrentes do uso deste software em contas reais.
