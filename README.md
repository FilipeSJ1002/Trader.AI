
#  TRADER.AI

> **Sistema de Trading Algorítmico Autônomo Baseado em Análise Técnica Quantitativa e Redes Neurais.**
>
> *Projeto de Trabalho de Conclusão de Curso (TCC) - Ciência da Computação.*

![Status](https://img.shields.io/badge/Status-Etapa%207%20(V6%20%7C%20Abla%C3%A7%C3%A3o%20%26%20Calibra%C3%A7%C3%A3o)-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/IA-PyTorch%20%7C%20BiLSTM%20%2B%20Attention-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![WebSockets](https://img.shields.io/badge/Data-WebSockets%20%7C%20Binance-F3BA2F?style=for-the-badge&logo=binance&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

##  Visão Geral do Projeto

O **TRADER.AI** é uma solução de software desenvolvida para automatizar o processo de tomada de decisão no mercado de criptoativos.

Diferente de sistemas tradicionais, esta inteligência quantitativa utiliza uma arquitetura modular que integra **Estratégias de Confluência** (união de múltiplos indicadores técnicos) e modelos de **Deep Learning** para mitigar o viés emocional humano e explorar ineficiências de mercado em operações 24/7.

O projeto encontra-se na **Etapa 7 (V6)**, que consolida a **estratégia híbrida bidirecional**: os sinais matemáticos clássicos (RSI, MACD, Bandas de Bollinger) definem *quando* operar, enquanto uma rede neural **BiLSTM + Attention** define *em qual direção* — comprando em mercados de alta (LONG) e vendendo a descoberto em mercados de queda (SHORT), com alavancagem proporcional à confiança do modelo.

---

##  Arquitetura da Estratégia

* **Classificador Direcional Neural:** Rede BiLSTM bidirecional com mecanismo de atenção (~2.17M parâmetros, PyTorch) que classifica cada janela de 120 minutos em **QUEDA / NEUTRO / ALTA**. O NEUTRO atua como sinal de "não operar" — essencial para não corroer o capital com taxas.
* **Estratégia Híbrida V1+V5:** O score de confluência clássico gera o gatilho de entrada; a rede neural confirma a direção via *confiança direcional* (`p_direção / (p_alta + p_queda)`). Só entra quando ambos concordam.
* **Operações Bidirecionais (LONG + SHORT):** Em regime de alta opera comprado nos *dips*; em regime de queda abre posições vendidas (futuros) e lucra com a desvalorização.
* **Filtro de Regime Diário:** Média móvel de 24h decide o lado permitido do mercado — elimina compras em tendência de baixa (o erro fatal das versões anteriores) e shorts em tendência de alta.
* **Gestão de Risco Completa:** Take Profit (+1.0%), Stop Loss (-0.5%, verificado minuto a minuto), saída por sinal técnico contrário, tempo máximo de posição (6h) e simulação de liquidação de futuros.
* **Alavancagem por Confiança:** 1x a 5x conforme a convicção do modelo (faixas superiores foram testadas e desativadas por descalibração estatística).
* **Pipeline de Experimentos Automatizado:** Preparação de dados, treino de múltiplos modelos e backtests comparativos executados em sequência sem supervisão.

###  Resultados do Backtest

Avaliação *walk-forward* honesta: treino até jun/2025, validação jul–dez/2025, teste 2026 (dados nunca vistos), mais um **holdout virgem** (jun–jul/2026) baixado *depois* de todas as decisões de projeto.

| Período | Estratégia | Hold de BTC | Vantagem |
|---|---|---|---|
| **Teste jan–jul/2026** (*bear market*) | **+2,0%** | −26,8% | **+28,8 p.p.** |
| **Holdout virgem jun–jul/2026** | **+0,2%** | −13,0% | **+13,2 p.p.** |
| Validação jul–dez/2025 (lateral) | −2,0% | −18,2% | +16,2 p.p. |

O *win rate* foi **idêntico (41,2%)** no teste e no holdout virgem — evidência de estabilidade, não de calibração afortunada. **Zero liquidações** em todos os períodos. Taxas de futuros (0,04%/lado) incluídas.

###  A rede neural agrega valor? (estudo de ablação)

O experimento mais importante do projeto: rodar a estratégia **idêntica**, com e sem o filtro neural.

| Período | COM rede neural | SEM rede neural | Contribuição da IA |
|---|---|---|---|
| Validação H2-2025 | −2,0% (55 ops) | −4,1% (232 ops) | **+2,1 p.p.** |
| **Teste jan–jul/2026** | **+2,0%** (68 ops) | **−6,0%** (246 ops) | **+8,0 p.p.** |
| Holdout virgem | +0,2% (17 ops) | −0,6% (57 ops) | **+0,8 p.p.** |

**Sem a rede neural o sistema perde dinheiro em todos os períodos.** Ela descarta ~75% dos sinais do componente determinístico e eleva o *win rate* de 36,6% para 41,2% — comprovando empiricamente a tese central do trabalho: **a arquitetura híbrida supera cada componente isolado**.

---

##  Estrutura do Projeto

```text
Trader.AI/
│  ── Bot em tempo real (API + WebSocket) ──────────────────────────────
├── main.py                    # Ponto de entrada (API REST e inicialização do WebSocket)
├── execution.py               # Motor autônomo de execução e gestão de risco
├── strategy.py                # Core Matemático: Confluência com Scoring (V1)
├── market_state.py            # Gerenciador de Estado: Memória RAM e histórico
├── binance_stream.py          # Conexão WebSocket em Tempo Real com a Binance
│
│  ── Núcleo de IA ────────────────────────────────────────────────────
├── v5_model.py                # Arquitetura BiLSTM + Attention (3 classes)
├── v5_data_prep.py            # Features (18) e rótulos direcionais
├── v6_data_prep.py            # Features enriquecidas (26): Bollinger, regime, sazonalidade
├── v5_train.py                # Treino: Focal Loss, early stopping, --resume, pausa cooperativa
├── v5_backtest.py             # Backtest híbrido: LONG/SHORT, TP/SL intrabar, regime, curvas de alavancagem
├── v5_live.py                 # Motor híbrido em tempo real (paper trading)
│
│  ── Ferramentas de análise (V6) ─────────────────────────────────────
├── v6_ablacao.py              # Mede a contribuição real da rede neural (com vs sem)
├── v6_edge_por_ativo.py       # Edge direcional por ativo
├── v6_edge_por_faixa.py       # Edge por faixa de confiança (calibração)
├── v6_calibracao.py           # Distribuição de confiança e precisão por faixa
├── v6_exp_ativos.py           # Experimento: ampliar universo de ativos
├── v6_exp_atr.py              # Experimento: stops adaptativos por volatilidade
├── v6_exp_curvas.py           # Experimento: curvas de alavancagem
├── v6_exp_regime.py           # Experimento: alavancagem condicionada ao regime
├── v6_sweep_k.py              # Sweep de parâmetro na validação
├── v6_veredito.py             # Comparação final V6 vs V5.9 com critério pré-definido
├── v5_walkforward.py          # Walk-forward retraining (3 folds trimestrais)
├── v5_run_experiments.py      # Pipeline de experimentos A/B
│
│  ── Controles operacionais (Windows, 1 clique) ──────────────────────
├── treino_pausar.cmd          # Pausa o treino e libera a GPU
├── treino_retomar.cmd         # Retoma o treino de onde parou
├── religar_trader.cmd         # Religa treino + live após reiniciar o PC
│
│  ── ETL ─────────────────────────────────────────────────────────────
├── download_binance_data.py   # Extração do Binance Vision (11 pares)
├── processar_dados.py         # Transformação: CSV -> Parquet
├── v6_refresh_dados.py        # Completa os parquets até "agora" via API REST
│
│  ── Documentação ────────────────────────────────────────────────────
├── README.md                  # Este arquivo
├── DOCUMENTACAO_TCC.md        # Documentação técnica completa (base do artigo)
├── TRAJETORIA_VERSOES_TCC.md  # Evolução V1 -> V6 (seção do artigo)
├── METODOLOGIA_EXPERIMENTAL.md# Protocolo científico e experimentos executados
│
│  ── Gerados em runtime (não versionados) ────────────────────────────
├── relatorios/                # TODA saída: logs, relatórios de backtest, resultados
├── data/                      # Datasets históricos .parquet
├── data_v6/                   # Dataset de treino (26 features)
├── *.pth                      # Modelos treinados
├── v5_live_state.json         # Estado do paper trading
└── .env                       # Chaves da API (Testnet)
```

> **Convenção:** toda saída gerada em execução vai para `relatorios/` — uma pasta, uma regra no `.gitignore`. O conteúdo é 100% regenerável, pois o código que o produz está versionado.
             
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

**Passo 1: Baixar Histórico** (11 pares, desde 2019)
```bash
python download_binance_data.py
```

**Passo 2: Processar para Parquet**
```bash
python processar_dados.py
```

**Passo 3 (opcional): Completar até agora** — o Binance Vision só publica meses fechados
```bash
python v6_refresh_dados.py
```

---

##  Treinando e Avaliando a IA

**Preparar os dados de treino:**
```bash
python v5_data_prep.py --dual     # 18 features, experimentos A e B numa passada
python v6_data_prep.py            # 26 features (V6)
```

**Treinar o modelo** (~13–16h em GTX 1650):
```bash
python v5_train.py --data data_v6 --model-out v6_model.pth --log v6_training.log --label V6.0 --resume --batch 192
Get-Content relatorios/v6_training.log -Wait   # acompanhar ao vivo
```
O treino salva checkpoint a cada melhora e **retoma de onde parou** com `--resume` — resistente a quedas de energia, crashes de cuDNN e reinícios.

**Rodar o backtest híbrido:**
```bash
python v5_backtest.py --model v5_model_b.pth          # teste (2026+)
python v5_backtest.py --model v5_model_b.pth --val    # validação (H2-2025)
python v5_backtest.py --model v5_model_b.pth --realop # diário detalhado de operações
```

Flags disponíveis:

| Flag | Função |
|---|---|
| `--from / --to AAAA-MM-DD` | Período customizado |
| `--assets ALL` ou lista | Universo de ativos (padrão: os 6 do treino) |
| `--featset v5\|v6` | Conjunto de features (deve casar com o modelo) |
| `--sl / --tp` | Stop loss / take profit percentuais |
| `--sl-mode atr --atr-k N` | Stops proporcionais à volatilidade do ativo |
| `--lev-curve {v59,edge,pico,flat1,flat2,regime}` | Curva de alavancagem |
| `--forca-min N` | Força mínima da tendência para alavancar (curvas `regime`) |
| `--ablacao sem_nn` | **Neutraliza a rede neural** (mede sua contribuição) |
| `--max-lev`, `--no-short`, `--skip`, `--eval-step`, `--max-hold` | Ajustes finos |

**Experimentos automatizados** (cada um roda dezenas de backtests e emite veredito):
```bash
python v6_ablacao.py        # a rede neural agrega valor?
python v6_exp_regime.py     # alavancagem condicionada ao regime
python v6_exp_curvas.py     # comparação de curvas de alavancagem
python v6_edge_por_faixa.py # calibração: o edge cresce com a confiança?
python v5_walkforward.py    # retreino trimestral vs modelo congelado
```

---

##  Paper Trading em Tempo Real

Executa a estratégia com **preços reais da Binance** e ordens **simuladas** — a validação final antes de qualquer capital.

```bash
python v5_live.py                 # loop contínuo (ciclo de 15 min)
python v5_live.py --once          # um único ciclo (teste)
python v5_live.py --reset         # zera capital e posições
Get-Content relatorios/v5_live.log -Wait
```

Estado persistente em `v5_live_state.json` (sobrevive a reinícios) e diário de operações em `v5_live_trades.csv`. Tolerante a quedas de rede e suspensão do PC.

> **Atenção:** sinal raro é o design — a estratégia faz ~1 operação a cada 3 dias. Dias sem trades são normais.

---

##  Bot V4 em Tempo Real (API + WebSocket)

```bash
uvicorn main:app --reload
```

**Comportamento Esperado:**
1. O servidor carregará o arquivo Parquet na memória RAM (`market_state.py`).
2. O servidor REST ficará disponível em `http://127.0.0.1:8000`.
3. O WebSocket iniciará em segundo plano e as decisões aparecerão no terminal:
   `[TRADER.AI]  Preço: $74434.37 | RSI: 65.20 | Decisão: NEUTRO`

> **Nota:** este motor ainda usa a estratégia de *trend-following* da V4. A integração da estratégia híbrida V5/V6 com execução real é o objetivo da **Etapa 8**.

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
- [x] **Etapa 6:** Validação *walk-forward* trimestral e motor de *paper trading* em tempo real com dados reais da Binance.
- [x] **Etapa 7 (V6):** Investigação sistemática dos limites do sistema — ablação da rede neural, calibração de confiança, expansão de universo, stops adaptativos e curvas de alavancagem.
- [ ] **Etapa 8:** Ponte para execução real — integração da estratégia híbrida ao motor de ordens e operação na Binance Futures (Testnet).

###  Achados da Etapa 6 (walk-forward)

Comparação honesta entre **retreinar o modelo a cada trimestre** vs. usar um **modelo congelado**, em 3 trimestres de dados nunca vistos (Q4-2025 a Q2-2026):

| Estratégia | Lucro acumulado (3 trimestres) |
|---|---|
| Modelo retreinado a cada trimestre | +2,0% |
| Modelo congelado (jun/2025) | +2,0% |

**Conclusão:** o retreino trimestral não trouxe ganho — o modelo congelado generaliza bem por ~11 meses. Retreinar por calendário foi descartado; o gatilho correto é a divergência entre desempenho ao vivo e esperado.

###  Achados da Etapa 7 (V6)

Cinco hipóteses de melhoria testadas com rigor — **quatro refutadas, uma confirmada**:

| Hipótese | Veredito | Evidência |
|---|---|---|
| A rede neural agrega valor? | ✅ **Confirmada** | **+8,0 p.p.** no teste; sem ela o sistema perde em todos os períodos |
| Ampliar universo (6 → 11 ativos) | ❌ Refutada | Pior nos 3 splits (−0,7% vs +2,0% no teste) |
| Stops adaptativos por volatilidade (ATR) | ❌ Refutada | Stop fixo vence com k de 6 a 12 |
| Realinhar curva de alavancagem | ⚠️ Inconclusiva | Trade-off dependente do regime |
| Enriquecer features (18 → 26) | ⚠️ Em avaliação | `val_loss` ainda atrás do modelo campeão |

**Descoberta transversal:** a alavancagem só compensa em **tendência forte**. Com *edge* estatístico modesto, alavancar em mercado lateral apenas multiplica o custo de fricção (taxas escalam com o notional) e a variância — sem melhorar a expectativa.

**Lição metodológica:** o *edge* direcional da rede medido em janelas aleatórias do mercado é ~0,50 (aleatório), o que sugeriria descartá-la. O estudo de ablação provou o contrário: ela não prevê o mercado do zero — **discrimina entre candidatos já filtrados pelo componente determinístico**. Métricas devem ser medidas no contexto real de uso, não em abstrato.

Detalhamento completo em [`METODOLOGIA_EXPERIMENTAL.md`](METODOLOGIA_EXPERIMENTAL.md).

---

## Autor

Desenvolvido por **Filipe Spirlandeli Junqueira**.

---
> Este projeto é estritamente educacional e experimental. O autor não se responsabiliza por perdas financeiras decorrentes do uso deste software em contas reais.
