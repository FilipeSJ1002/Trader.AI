
#  TRADER.AI

> **Sistema de Trading Algorítmico Autônomo Baseado em Análise Técnica Quantitativa e Redes Neurais.**
>
> *Projeto de Trabalho de Conclusão de Curso (TCC) - Ciência da Computação.*

![Status](https://img.shields.io/badge/Status-Etapa%205%20(IA%20Neural%20%26%20Estrat%C3%A9gia%20H%C3%ADbrida)-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/IA-PyTorch%20%7C%20BiLSTM%20%2B%20Attention-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![WebSockets](https://img.shields.io/badge/Data-WebSockets%20%7C%20Binance-F3BA2F?style=for-the-badge&logo=binance&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

##  Visão Geral do Projeto

O **TRADER.AI** é uma solução de software desenvolvida para automatizar o processo de tomada de decisão no mercado de criptoativos.

Diferente de sistemas tradicionais, esta inteligência quantitativa utiliza uma arquitetura modular que integra **Estratégias de Confluência** (união de múltiplos indicadores técnicos) e modelos de **Deep Learning** para mitigar o viés emocional humano e explorar ineficiências de mercado em operações 24/7.

Atualmente, o projeto encontra-se na **Fase 5**, que introduz a **estratégia híbrida bidirecional**: os sinais matemáticos clássicos (RSI, MACD, Bandas de Bollinger) definem *quando* operar, enquanto uma rede neural **BiLSTM + Attention** define *em qual direção* — comprando em mercados de alta (LONG) e vendendo a descoberto em mercados de queda (SHORT), com alavancagem proporcional à confiança do modelo.

---

##  Funcionalidades da Etapa 5

* **Classificador Direcional Neural (V5):** Rede BiLSTM bidirecional com mecanismo de atenção (~2.17M parâmetros, PyTorch) que classifica cada janela de 120 minutos em **QUEDA / NEUTRO / ALTA** para 6 criptoativos simultaneamente. O NEUTRO atua como sinal de "não operar" — essencial para não corroer o capital com taxas.
* **Estratégia Híbrida V1+V5:** O score de confluência clássico gera o gatilho de entrada; a rede neural confirma a direção via *confiança direcional* (`p_direção / (p_alta + p_queda)`). Só entra quando ambos concordam.
* **Operações Bidirecionais (LONG + SHORT):** Em regime de alta opera comprado nos *dips*; em regime de queda abre posições vendidas (futuros) e lucra com a desvalorização.
* **Filtro de Regime Diário:** Média móvel de 24h decide o lado permitido do mercado — elimina compras em tendência de baixa (o erro fatal das versões anteriores) e shorts em tendência de alta.
* **Gestão de Risco Completa:** Take Profit (+1.0%), Stop Loss (-0.5%, verificado minuto a minuto), saída por sinal técnico contrário, tempo máximo de posição (6h) e simulação de liquidação de futuros.
* **Alavancagem por Confiança:** 1x a 5x conforme a convicção do modelo (faixas superiores foram testadas e desativadas por descalibração estatística).
* **Pipeline de Experimentos Automatizado:** Preparação de dados, treino de múltiplos modelos e backtests comparativos executados em sequência sem supervisão (`v5_run_experiments.py`).

###  Resultados do Backtest (V5.9 — modelo "especialista em queda")

Avaliação *walk-forward* honesta: treino até jun/2025, validação jul-dez/2025, teste jan-mai/2026 (dados nunca vistos).

| Período | Estratégia | Hold de BTC | Resultado |
|---|---|---|---|
| Teste (jan-mai/2026, *bear market*) | **+1.8%** | -15.9% | ✅ Lucro operando na queda |
| Validação (jul-dez/2025, mercado lateral) | -2.0% | -18.2% | ✅ Capital preservado |

*Zero liquidações em 11 meses simulados. Taxas de futuros (0.04%/lado) incluídas. Modelo treinado em GTX 1650.*

---

##  Estrutura do Projeto

```text
Trader.AI/
├── main.py                    # Ponto de entrada (API REST e inicialização do WebSocket)
├── execution.py               # Motor autônomo de execução e gestão de risco
├── strategy.py                # Core Matemático: Confluência com Scoring (V1)
├── market_state.py            # Gerenciador de Estado: Memória RAM e histórico
├── binance_stream.py          # Conexão WebSocket em Tempo Real com a Binance
│
├── v5_data_prep.py            # IA: Features multi-timeframe e rótulos direcionais (--dual p/ experimentos)
├── v5_model.py                # IA: Arquitetura BiLSTM + Attention (3 classes)
├── v5_train.py                # IA: Treino com Focal Loss e early stopping
├── v5_backtest.py             # IA: Backtest híbrido bidirecional (LONG/SHORT + TP/SL + regime)
├── v5_run_experiments.py      # IA: Pipeline automático (prep -> treinos -> backtests)
├── v5_model.pth               # Modelo V5.8 baseline (local, não versionado)
├── v5_model_b.pth             # Modelo V5.9-B vencedor (local, não versionado)
│
├── download_binance_data.py   # ETL: Script de Extração (Download progressivo)
├── processar_dados.py         # ETL: Script de Transformação (Limpeza e Parquet)
├── data/                      # Datasets históricos .parquet (local, não versionado)
├── .env                       # Variáveis de ambiente seguras (API Keys da Testnet)
├── requirements.txt           # Lista de dependências e bibliotecas
└── README.md                  # Documentação oficial
```

---

##  Instalação e Execução

### 1. Pré-requisitos
* Python 3.10 ou superior instalado.
* Git instalado.
* (Opcional, recomendado para treino) GPU NVIDIA com CUDA.

### 2. Clonar o Repositório
```bash
git clone https://github.com/SEU_USUARIO/Trader.AI.git
cd Trader.AI
```

### 3. Configurar o Ambiente Virtual (Obrigatório)
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
2. Adicione suas credenciais da **Binance Testnet** dentro do `.env`:
   - `BINANCE_API_KEY=sua_chave_aqui`
   - `BINANCE_SECRET_KEY=sua_chave_secreta_aqui`

### 5. Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## Alimentando a Memória Inicial (ETL)

**Passo 1: Baixar Histórico**
```bash
python download_binance_data.py
```

**Passo 2: Processar para Parquet**
```bash
python processar_dados.py
```

---

##  Treinando e Avaliando a IA (V5)

**Preparar os dados de treino** (features + rótulos direcionais):
```bash
python v5_data_prep.py            # dataset padrão
python v5_data_prep.py --dual     # datasets dos experimentos A e B numa passada
```

**Treinar o modelo** (~12h em GTX 1650; acompanhe com `Get-Content v5_training.log -Wait`):
```bash
python v5_train.py --data data_v5a --y-dir data_v5b --model-out v5_model_b.pth --label V5.9-B
```

**Rodar o backtest híbrido**:
```bash
python v5_backtest.py --model v5_model_b.pth          # período de teste (2026+)
python v5_backtest.py --model v5_model_b.pth --val    # período de validação
python v5_backtest.py --model v5_model_b.pth --realop # diário detalhado de operações
```
Flags úteis: `--sl`, `--tp`, `--max-lev`, `--no-short`, `--skip BTCUSDT`, `--from/--to AAAA-MM-DD`.

**Pipeline completo de experimentos** (roda tudo sozinho):
```bash
python v5_run_experiments.py
```

---

##  Como Rodar a Inteligência em Tempo Real

```bash
uvicorn main:app --reload
```

**Comportamento Esperado:**
1. O servidor carregará o arquivo Parquet na memória RAM (`market_state.py`).
2. O servidor REST ficará disponível em `http://127.0.0.1:8000`.
3. O WebSocket iniciará em segundo plano e as decisões aparecerão no terminal:
   `[TRADER.AI]  Preço: $74434.37 | RSI: 65.20 | Decisão: NEUTRO`

###  Verificando o Saldo da Conta
```bash
python teste_saldo.py
```

---

##  Roadmap de Desenvolvimento

- [x] **Etapa 1:** Arquitetura Base, API e Estratégia de Confluência.
- [x] **Etapa 2:** Integração com Dataset Histórico Real e ETL automatizado.
- [x] **Etapa 3:** Conexão com API Binance via WebSockets e Decisões em Tempo Real.
- [x] **Etapa 4:** Execução de Ordens (Integração de Contas, Scoring e Live Trading).
- [x] **Etapa 5:** Rede Neural Direcional (BiLSTM + Attention) e Estratégia Híbrida Bidirecional (LONG/SHORT).
- [ ] **Etapa 6:** *Walk-forward retraining* trimestral e integração do modelo híbrido ao motor de execução em tempo real.

---

## Autor

Desenvolvido por **Filipe Spirlandeli Junqueira**.

---
> Este projeto é estritamente educacional e experimental. O autor não se responsabiliza por perdas financeiras decorrentes do uso deste software em contas reais.
