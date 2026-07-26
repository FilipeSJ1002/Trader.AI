# Trader.AI — Documentação Técnica Completa

> Documento-base para o artigo e a apresentação do Trabalho de Conclusão de Curso (TCC).
> Descreve o sistema de ponta a ponta: arquitetura, dados, modelo de IA, estratégia,
> metodologia de avaliação, resultados e ferramentas.
>
> Autor: **Filipe Spirlandeli Junqueira** — Ciência da Computação.

---

## 1. Resumo (Abstract)

O **Trader.AI** é um sistema autônomo de *day trading* para o mercado de criptomoedas que
combina **análise técnica quantitativa** (regras matemáticas determinísticas) com uma
**rede neural profunda** (classificador direcional) numa **estratégia híbrida**. A premissa
central é a separação de responsabilidades: os indicadores técnicos clássicos decidem
**QUANDO** existe uma oportunidade de operação, e a rede neural decide **EM QUAL DIREÇÃO**
operar (comprar ou vender). O sistema opera de forma **bidirecional** — lucra tanto na alta
(posições *long*) quanto na queda (posições *short* alavancadas) — sob um arcabouço rígido
de gestão de risco. O modelo é avaliado por **validação walk-forward**, metodologia padrão
em finanças quantitativas que impede o vazamento de informação do futuro.

---

## 2. Objetivo e Motivação

**Objetivo geral:** construir um agente de negociação que tome decisões de forma autônoma,
sem viés emocional humano, e que **preserve e valorize capital inclusive em mercados de baixa**.

**Motivação:** estratégias puramente baseadas em regras ("compre quando o RSI estiver baixo")
funcionam apenas em mercados de alta — elas compram quedas que, num *bear market*, continuam
caindo. A hipótese do projeto é que uma rede neural treinada para reconhecer o **regime e a
direção** do mercado pode atuar como um **filtro inteligente** sobre as regras clássicas,
permitindo operar com segurança nos dois sentidos do mercado.

---

## 3. Evolução do Projeto

O Trader.AI passou por três gerações de inteligência, todas preservadas no histórico:

| Geração | Inteligência | Tecnologia | Característica |
|---------|--------------|------------|---------------|
| **V1** | Regras determinísticas | Indicadores técnicos (RSI, MACD, Bollinger) | Sistema de *scoring* de compra/venda. Lucrativo só em alta. |
| **V4** | Aprendizado de máquina clássico | **LightGBM** (árvores de decisão / *gradient boosting*) | Filtro preditivo de probabilidade de sucesso da operação. |
| **V5** *(atual)* | Aprendizado profundo | **Rede Neural BiLSTM + Attention** (PyTorch) | Classificador direcional sequencial + estratégia híbrida bidirecional. |

A versão atual (V5.9) **não descarta** o conhecimento anterior: ela usa a lógica de regras da
V1 como gatilho de entrada e a rede neural como confirmador de direção.

---

## 4. Arquitetura Geral do Sistema

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FONTE DE DADOS                              │
│              Binance (candles de 1 minuto, 6 ativos)                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │  ETL (download + processamento)
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ENGENHARIA DE FEATURES (18)                       │
│   Geometria de candle · retornos multi-timeframe · RSI · MACD ·      │
│   ATR · EMAs · Stochastic · z-score de volume · contexto do BTC      │
└───────────────────────────────┬─────────────────────────────────────┘
                                 │  janelas de 120 minutos
                                 ▼
┌──────────────────────────────┐     ┌──────────────────────────────────┐
│   REDE NEURAL (V5)           │     │   REGRAS TÉCNICAS (V1)            │
│   BiLSTM + Attention         │     │   buy_score / sell_score          │
│   → QUEDA / NEUTRO / ALTA    │     │   → gatilho de entrada            │
└──────────────┬───────────────┘     └────────────────┬─────────────────┘
               │                                       │
               └──────────────┬────────────────────────┘
                              ▼
            ┌─────────────────────────────────────────┐
            │       ESTRATÉGIA HÍBRIDA                 │
            │  Regra dispara + NN confirma direção     │
            │  + filtro de regime + gestão de risco    │
            └──────────────────┬──────────────────────┘
                               ▼
            ┌─────────────────────────────────────────┐
            │  EXECUÇÃO: Backtest · Paper · (Testnet)  │
            └─────────────────────────────────────────┘
```

---

## 5. Dados

- **Fonte:** API da **Binance** (dados públicos de mercado).
- **Granularidade:** candles de **1 minuto** (OHLCV — abertura, máxima, mínima, fechamento, volume).
- **Período:** **janeiro/2019 a maio/2026** (~3,9 milhões de candles por ativo no caso do BTC).
- **Ativos operados (6):** BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, XRPUSDT, AVAXUSDT.
- **Ativos disponíveis (11):** os seis acima mais DOGEUSDT, LINKUSDT, ADAUSDT, DOTUSDT,
  LTCUSDT. A expansão do universo operado foi testada e rejeitada (seção 11.4) — os cinco
  pares adicionais permanecem na base para experimentos futuros.
- **Armazenamento:** arquivos **Parquet** (formato colunar comprimido, leitura rápida).
- **ETL:** `download_binance_data.py` (extração paginada) e `processar_dados.py`
  (limpeza e conversão para Parquet).

O BTC tem papel duplo: é um ativo operável **e** serve de **contexto de mercado** — seus
retornos são injetados como features em todos os outros ativos (quando o BTC cai, o mercado
inteiro tende a cair).

---

## 6. Engenharia de Features (18 variáveis)

Cada instante de tempo é descrito por **18 features normalizadas** (todas adimensionais, em
escala relativa ao preço — o que permite ao modelo generalizar entre ativos de preços muito
diferentes, como BTC a $70.000 e XRP a $1,30). Computadas em `v5_data_prep.py::_add_features`.

| # | Feature | Significado |
|---|---------|-------------|
| 1 | `body` | Tamanho do corpo do candle (fechamento − abertura) / preço |
| 2 | `upper_wick` | Pavio superior (rejeição de alta) |
| 3 | `lower_wick` | Pavio inferior (rejeição de baixa) |
| 4 | `vol_norm` | Volume relativo à média de 20 períodos |
| 5–8 | `ret_5m`, `ret_15m`, `ret_60m`, `ret_240m` | Retornos em 5/15/60/240 min (visão *multi-timeframe*) |
| 9 | `rsi` | Índice de Força Relativa (normalizado em torno de 0) |
| 10 | `macdh` | Histograma do MACD (momentum) |
| 11 | `atr_pct` | *Average True Range* (volatilidade) como % do preço |
| 12 | `ema20_dist` | Distância à média móvel exponencial de 20 |
| 13 | `ema50_dist` | Distância à EMA de 50 |
| 14 | `ema200_dist` | Distância à EMA de 200 (tendência de longo prazo) |
| 15 | `vol_zscore` | Z-score do volume (detecta picos anômalos) |
| 16 | `stoch_k` | Oscilador estocástico %K (momentum complementar ao RSI) |
| 17 | `btc_ret_60` | Retorno do BTC em 60 min (contexto de mercado) |
| 18 | `btc_ret_240` | Retorno do BTC em 240 min (contexto de mercado) |

> **Nota técnica honesta:** o design original previa também duas features de Bandas de
> Bollinger (`dist_bbu`, `dist_bbl`). Na versão `pandas-ta 0.4.71b0` utilizada, o nome
> interno das colunas mudou, de modo que essas duas features **não são geradas** — o modelo
> opera, na prática, com 18 variáveis. Isso é documentado como limitação conhecida (Seção 16).

---

## 7. O Modelo de IA (Rede Neural V5)

Arquivo: `v5_model.py`. Implementado em **PyTorch**.

### 7.1 Arquitetura

```
Entrada: (lote, 120 timesteps, 18 features)
   │
   ▼
LayerNorm  ........................ normaliza as features de entrada
   │
   ▼
LSTM Bidirecional .................. hidden_size=256, 2 camadas, dropout=0.4
   │                                  (lê a sequência nos dois sentidos)
   ▼
Camada de Atenção (Attention) ...... pondera quais dos 120 minutos
   │                                  são mais relevantes para a decisão
   ▼
Cabeça densa: Linear(512→64) → ReLU → Dropout → Linear(64→3)
   │
   ▼
Saída: 3 logits → softmax → [P(QUEDA), P(NEUTRO), P(ALTA)]
```

- **Tipo:** classificador sequencial (rede recorrente).
- **Por que BiLSTM?** O preço é uma **série temporal**; uma LSTM bidirecional captura
  dependências de curto e médio prazo ao longo dos 120 minutos da janela — algo que modelos
  de árvore (LightGBM da V4) não fazem bem.
- **Por que Attention?** Permite ao modelo "focar" nos minutos mais informativos da janela
  (ex.: o instante de um pico de volume) em vez de tratar todos igualmente.
- **Parâmetros treináveis:** **2.175.784** (~2,17 milhões).

### 7.2 Saída e interpretação

O modelo classifica cada janela em **3 classes**:

| Classe | Rótulo | Uso na estratégia |
|--------|--------|-------------------|
| 0 | **QUEDA** | Sinal para operar *short* (vender) |
| 1 | **NEUTRO** | Sinal de **não operar** (essencial para não pagar taxas à toa) |
| 2 | **ALTA** | Sinal para operar *long* (comprar) |

A existência da classe **NEUTRO** é deliberada: sem ela, o sistema seria forçado a operar o
tempo todo e as taxas de corretagem consumiriam o capital.

---

## 8. Rotulagem dos Dados (Targets)

O alvo de treino é a **direção futura do preço** num horizonte de **120 minutos (2 horas)**:

- Se o preço subir mais que um limiar → **ALTA**
- Se cair mais que um limiar → **QUEDA**
- Caso contrário → **NEUTRO**

Os limiares foram objeto de **experimentação controlada** (Seção 11.1). A configuração
vencedora (modelo "B", especialista em queda) usa:
- **ALTA:** subida > **0,8%**
- **QUEDA:** queda > **0,4%** (mais sensível — detecta quedas mais cedo)

Parâmetros de janela: `WINDOW_SIZE = 120` (entrada), `HORIZON_H = 120` (previsão),
`SUBSAMPLE = 15` (1 amostra a cada 15 candles, para reduzir redundância e tamanho do dataset).

---

## 9. Metodologia de Treinamento

Arquivo: `v5_train.py`.

| Item | Configuração | Justificativa |
|------|--------------|---------------|
| **Função de perda** | **Focal Loss** (γ=2) | Penaliza menos os exemplos fáceis (NEUTRO, maioria) e força o modelo a aprender os sinais raros (ALTA/QUEDA) |
| **Pesos por classe** | Inversos à frequência | Compensa o desbalanceamento (o mercado é majoritariamente NEUTRO) |
| **Otimizador** | AdamW (lr=1e-4, weight_decay=2e-4) | Regularização L2 contra *overfitting* |
| **Agendador de LR** | Cosine Annealing (até 5e-6) | Reduz a taxa de aprendizado suavemente |
| **Batch size** | 256 (192 no walk-forward) | Limitado pela VRAM da GPU (4 GB) |
| **Épocas** | até 60, com **early stopping** (paciência 15) | Para quando a validação para de melhorar |
| **Regularização** | Dropout 0,4 + *gradient clipping* (1.0) | Estabilidade e generalização |

**Divisão dos dados (walk-forward base):**
- **Treino:** até 30/06/2025
- **Validação:** 2º semestre de 2025
- **Teste:** 2026 em diante (dados nunca vistos)

---

## 10. A Estratégia Híbrida (V1 + V5)

Arquivo: `v5_backtest.py`. É o coração do sistema.

### 10.1 Entrada (abertura de posição)

Uma posição só é aberta quando **as duas inteligências concordam**:

1. **Gatilho técnico (V1):** o `buy_score` ou `sell_score` (somatório de pontos de RSI,
   MACD, volume e filtro EMA200) atinge o limiar **≥ 60**.
2. **Confirmação neural (V5):** a **confiança direcional** do modelo é **≥ 52%**, onde:

   ```
   confiança_direcional = P(direção) / (P(ALTA) + P(QUEDA))
   ```

   Essa razão ignora o peso do NEUTRO e mede **o quanto o modelo favorece uma direção sobre
   a outra**. Foi uma solução-chave do projeto: como o modelo prevê NEUTRO na maioria do
   tempo, usar a probabilidade absoluta quase nunca dispararia operações — a razão relativa
   resolve isso.

### 10.2 Filtro de Regime Diário

Uma média móvel de **24 horas (1440 minutos)** define o regime e **restringe o lado** das
operações:
- Preço **acima** da média de 24h → mercado em alta → **somente LONG** (compra os *dips*).
- Preço **abaixo** da média de 24h → mercado em queda → **somente SHORT** (vende os repiques).

Esse filtro corrige o erro fatal da V1 (comprar quedas no *bear market*).

### 10.3 Saída (fechamento de posição)

A posição é encerrada pelo **primeiro** dos eventos abaixo:
- **Take Profit (TP):** +1,0% de lucro.
- **Stop Loss (SL):** −0,5% de perda (verificado **minuto a minuto**, *intrabar*).
- **Sinal técnico contrário** (a V1 sinaliza reversão).
- **Tempo máximo:** 6 horas de posição.

A relação risco/retorno é **2:1** (ganha 1% ou perde 0,5%) — desenhada para ser lucrativa
mesmo com taxa de acerto abaixo de 50%.

### 10.4 Alavancagem por Confiança

A alavancagem é proporcional à convicção do modelo:

| Confiança direcional | Alavancagem |
|----------------------|-------------|
| ≥ 52% | 1x (spot) |
| ≥ 57% | 2x |
| ≥ 62% | **5x (teto atual)** |

Faixas superiores (10x, 20x) foram **testadas e desativadas**: estatisticamente, a
confiança acima de 67% do modelo não é **calibrada** (poucos exemplos no treino), e operá-las
gerava prejuízo. O teto de 5x é uma decisão baseada em evidência.

### 10.5 Parâmetros financeiros

- Capital inicial simulado: **$10.000**.
- Margem por operação: **20% do capital**.
- Taxa (futuros Binance): **0,04% por lado**.
- Margem de manutenção (liquidação): **0,5%**.

---

## 11. Experimentação Científica

### 11.1 Experimento A vs. B (qual rotulagem aprende melhor?)

Dois modelos foram treinados com a **mesma arquitetura**, mudando apenas os limiares de rótulo:

| Modelo | Rótulos | Resultado no teste (2026) |
|--------|---------|---------------------------|
| **A** — "mais sinais" | ALTA/QUEDA simétricos em 0,4% | +0,3% |
| **B** — "especialista em queda" | ALTA 0,8% / QUEDA 0,4% | **+11,1%** ✅ |

O modelo **B venceu** — confirmando a hipótese de que detectar quedas com sensibilidade é
mais valioso, dado que o período de teste foi um *bear market*.

> **Ressalva:** esse +11,1% foi obtido em configuração ainda sem gestão de risco
> estruturada, e o mesmo modelo perdia 19,6% na validação. O número **não** representa
> o sistema final (+2,0%); serve apenas como comparação entre rotulagens.

### 11.2 Walk-Forward (retreinar vale a pena?)

Comparação entre **retreinar o modelo a cada trimestre** vs. **modelo congelado**, em três
trimestres de dados nunca vistos (`v5_walkforward.py`):

| Trimestre | Modelo retreinado | Modelo congelado |
|-----------|-------------------|------------------|
| Q4-2025 | +0,8% | +0,4% |
| Q1-2026 | +0,8% | +1,1% |
| Q2-2026 | +0,4% | +0,5% |
| **Total** | **+2,0%** | **+2,0%** |

**Conclusão (resultado negativo, cientificamente valioso):** o retreino trimestral **não
trouxe ganho** — o modelo congelado generaliza bem por ~11 meses. Isso simplifica a operação
em produção e indica que a próxima alavanca de melhoria **não** é o cronograma de treino.

### 11.3 Ablação — a contribuição da rede neural

Executar a estratégia com e sem o filtro neural, mantendo todo o resto constante:

| Período | Com rede neural | Sem rede neural | Contribuição |
|---|---|---|---|
| Validação H2-2025 | −2,0% (55 ops) | −4,1% (232 ops) | **+2,1 p.p.** |
| Teste jan–jul/2026 | **+2,0%** (68 ops) | **−6,0%** (246 ops) | **+8,0 p.p.** |
| *Holdout* virgem | +0,2% (17 ops) | −0,6% (57 ops) | **+0,8 p.p.** |

**Resultado: a rede neural é essencial** — sem ela o sistema perde dinheiro em todos os
períodos.

**Lição metodológica relevante:** uma medição anterior avaliou a capacidade preditiva da
rede isoladamente (taxa de acerto direcional em janelas arbitrárias do mercado) e obteve
valor próximo de 0,50 — equivalente ao acaso. Essa métrica sugeriria que o componente era
dispensável. A ablação demonstrou o contrário. A explicação está na natureza da tarefa: a
rede não prevê a direção do mercado a partir do zero; ela **discrimina entre candidatos
previamente filtrados pelos indicadores técnicos**. Métricas devem ser aferidas no contexto
real de aplicação.

### 11.4 Hipóteses de melhoria refutadas

Três hipóteses foram formuladas e rejeitadas com evidência nos três *splits*:

| Hipótese | Resultado |
|---|---|
| Ampliar o universo de 6 para 11 ativos | −0,7% no teste (contra +2,0% com 6 ativos) |
| Stops adaptativos por volatilidade (ATR) | Stop fixo venceu com multiplicadores de 6 a 12 |
| Enriquecer features (18 → 26 variáveis) | Erro de validação superior ao modelo vigente |

### 11.5 Achado transversal — alavancagem e regime de mercado

Comparação de curvas de alavancagem na validação:

| Curva | Comportamento | Validação |
|---|---|---|
| Concentrada na faixa de melhor discriminação | Seletiva | +0,5% |
| **Sem alavancagem** | Exposição 1x | −0,8% |
| Histórica (proporcional à confiança) | Até 5x | −2,0% |

Operar **sem alavancagem alguma** superou a configuração vigente em 1,2 ponto percentual em
mercado lateral. A explicação é econômica: os custos de transação escalam com o valor
nocional. Com margem preditiva modesta, a alavancagem não melhora o valor esperado —
amplifica o custo de fricção e a variância.

Uma variante condicionada à força da tendência apresentou soma superior (+1,0% contra +0,2%)
e variância substancialmente menor, porém desempenho inferior em mercado de forte tendência.
Trata-se de um compromisso entre retorno e consistência, não de melhoria absoluta.

> O detalhamento completo da metodologia e de todos os experimentos está em
> `METODOLOGIA_EXPERIMENTAL.md`.

---

## 12. Backtesting (Metodologia de Avaliação)

O backtest (`v5_backtest.py`) simula a operação real candle a candle, com realismo:
- Verificação de sinais a cada **15 minutos**.
- Saídas TP/SL checadas **minuto a minuto** (não só no fechamento).
- **Taxas** de corretagem descontadas em cada operação.
- Simulação de **liquidação** de futuros (se o preço cruza o nível de liquidação).
- Comparação contra dois *benchmarks*: **comprar e segurar BTC** e **manter em dólar**.

### Resultados principais (modelo B, configuração final)

| Período | Trader.AI | Comprar e segurar BTC | Vantagem | Liquidações |
|---------|-----------|------------------------|----------|-------------|
| **Teste jan–jul/2026** (*bear*) | **+2,0%** | −26,8% | **+28,8 p.p.** | 0 |
| **Holdout virgem jun–jul/2026** | **+0,2%** | −13,0% | **+13,2 p.p.** | 0 |
| Validação jul–dez/2025 (lateral) | −2,0% | −18,2% | +16,2 p.p. | 0 |

O sistema **supera o BTC em 13 a 29 pontos percentuais** em todos os períodos e **nunca
foi liquidado** — ou seja, protege capital em mercados adversos, que era o objetivo central.

A **taxa de acerto foi idêntica (41,2%)** no período de teste e no *holdout* virgem —
evidência de estabilidade comportamental, e não de calibração afortunada.

> **Nota sobre o *holdout* virgem:** os dados de junho e julho de 2026 foram obtidos
> em 25/07/2026, *após* o congelamento de toda a arquitetura, parâmetros e limiares.
> Nenhuma decisão de projeto pôde ser influenciada por eles — trata-se da aproximação
> mais fiel de operação real sem exposição de capital.

### Estudo de ablação: qual a contribuição da rede neural?

O experimento mais importante do projeto executa a estratégia de forma **idêntica**,
com e sem o filtro neural, isolando sua contribuição:

| Período | Com rede neural | Sem rede neural | Contribuição |
|---|---|---|---|
| Validação H2-2025 | −2,0% (55 ops) | −4,1% (232 ops) | **+2,1 p.p.** |
| Teste jan–jul/2026 | **+2,0%** (68 ops) | **−6,0%** (246 ops) | **+8,0 p.p.** |
| *Holdout* virgem | +0,2% (17 ops) | −0,6% (57 ops) | **+0,8 p.p.** |

**Sem a rede neural o sistema é deficitário em todos os períodos.** Ela descarta cerca
de 75% dos sinais gerados pelos indicadores técnicos e eleva a taxa de acerto de 36,6%
para 41,2%. Este resultado sustenta empiricamente a tese central do trabalho: **a
arquitetura híbrida supera cada componente isolado**.

---

## 13. Operação em Tempo Real (Paper Trading)

Arquivo: `v5_live.py`. Executa a estratégia **ao vivo com preços reais da Binance**, mas com
ordens **simuladas** (*paper trading*) — a validação final antes de qualquer dinheiro real.
- Busca dados via **API REST** pública da Binance a cada 15 minutos.
- Aplica exatamente as mesmas regras do backtest (reutiliza o código, garantindo paridade).
- Persiste estado em `v5_live_state.json` (sobrevive a reinícios) e registra cada operação
  em `v5_live_trades.csv`.
- Tolerante a falhas: reconecta sozinho após quedas de rede ou suspensão do PC.

---

## 13.1 Arquitetura de Execução Real (Binance Futures)

Arquivos: `v6_executor.py` (interface com a corretora) e `v6_ciclo.py` (ponte entre
estratégia e execução).

### Separação de responsabilidades

| Módulo | Responsabilidade |
|---|---|
| `v5_backtest.py` | Define a estratégia — funções de decisão |
| `v6_ciclo.py` | **Importa** essas funções e as aplica a dados ao vivo |
| `v6_executor.py` | Traduz decisões em ordens; não conhece a estratégia |

Decisão de projeto relevante: `v6_ciclo.py` **importa** `v1_scores` e `leverage_for`
de `v5_backtest.py`, em vez de reimplementá-las. A lógica que movimenta capital é,
portanto, o mesmo objeto de código validado nos backtests. Reimplementação seria a
via mais provável para divergência silenciosa entre o sistema simulado e o real.

### Modelo de segurança em três camadas

| Camada | Estado padrão | Liberação |
|---|---|---|
| **Modo de simulação** (*dry-run*) | Registra ordens sem enviá-las | Flag explícita `--armar` |
| **Ambiente** | Testnet (capital fictício) | Flag `--real` + confirmação textual |
| **Limites operacionais** | 3 posições, alavancagem 5x, exposição 60% | Constantes no código |

As camadas são independentes: operar com dinheiro real exige três ações
deliberadas e distintas.

### Proteção residente na corretora

Ao abrir posição, o sistema registra *take profit* e *stop loss* como **ordens
condicionais *reduce-only* na própria Binance**. A consequência é relevante para
confiabilidade: se o processo for interrompido — falha de software, queda de
energia, perda de conectividade —, a proteção do capital permanece ativa, pois
a execução da saída passa a ser responsabilidade da corretora.

A alternativa (manter os stops apenas na lógica do bot) cria dependência entre a
integridade do capital e a disponibilidade contínua do processo, o que constitui
fragilidade estrutural em sistemas de negociação automatizada.

### Conformidade com as regras da corretora

Cada par negociado possui restrições próprias de granularidade e valor mínimo,
obtidas dinamicamente via `futures_exchange_info`:

| Par | Incremento de quantidade | Notional mínimo |
|---|---|---|
| BTCUSDT | 0,0001 | US$ 50 |
| ETHUSDT | 0,001 | US$ 20 |
| AVAXUSDT | 1 (apenas inteiros) | US$ 5 |
| XRPUSDT | 0,1 | US$ 5 |

O executor arredonda quantidades para baixo respeitando o incremento e verifica o
notional resultante **após** o arredondamento — verificação necessária, pois o
arredondamento pode reduzir a ordem abaixo do mínimo exigido.

### Estado e reconciliação

A fonte de verdade sobre posições abertas é a **corretora**, consultada a cada
ciclo via `futures_position_information`. O arquivo de estado local registra
metadados complementares (instante de abertura, confiança do modelo na decisão),
mas não é tratado como autoridade — o que permite reinício do processo sem
inconsistência de estado.

---

## 14. Robustez Operacional

Como o treino leva ~12–16h por modelo numa GPU modesta (GTX 1650, 4 GB, compartilhada com o
uso normal do PC), foram implementados mecanismos de resiliência:

- **Checkpoint + resume:** o treino salva o melhor modelo a cada melhora e **retoma de onde
  parou** após qualquer interrupção (crash de cuDNN, falta de memória, desligamento).
- **Pausar/retomar:** scripts de um clique (`treino_pausar.cmd` / `treino_retomar.cmd`)
  liberam a GPU instantaneamente para uso pessoal sem perder progresso.
- **Religação pós-reboot:** `religar_trader.cmd` reinicia treino e *paper trading* do ponto salvo.
- **Re-tentativas automáticas:** o pipeline de walk-forward tenta cada etapa até 3 vezes.

---

## 15. Stack Tecnológico

| Categoria | Ferramenta | Versão |
|-----------|-----------|--------|
| Linguagem | **Python** | 3.13.1 |
| Deep Learning | **PyTorch** | 2.6.0 (CUDA 12.4) |
| Hardware (treino) | **NVIDIA GeForce GTX 1650** | 4 GB VRAM |
| Manipulação de dados | **pandas** / **NumPy** | 3.0.1 / 2.2.6 |
| Indicadores técnicos | **pandas-ta** | 0.4.71b0 |
| ML clássico (legado V4) | **scikit-learn** / **LightGBM** | 1.8.0 |
| Armazenamento | **PyArrow** (Parquet) / **SQLite** | 23.0.1 |
| API / servidor | **FastAPI** + **Uvicorn** | 0.135.1 |
| Tempo real | **WebSockets** / **python-binance** | 1.0.35 |
| Persistência de modelo | **joblib** (.pkl) / `torch.save` (.pth) | — |

---

## 16. Limitações Conhecidas e Trabalhos Futuros

**Limitações metodológicas:**
1. **Ausência de validação em execução real.** Todos os resultados provêm de simulação.
   *Slippage*, latência, rejeição de ordens e custos de *funding* não estão modelados.
2. **Amostras pequenas em recortes específicos.** O *holdout* virgem contém 17 operações;
   configurações mais seletivas chegam a 5–8. Conclusões nesses recortes carregam incerteza
   estatística elevada.
3. **Sobreposição parcial entre períodos.** O teste (jan–jul/2026) contém o *holdout*
   (jun–jul/2026); comparações estritamente independentes exigem separar jan–mai de jun–jul.
4. **Período de avaliação predominantemente baixista.** O desempenho em mercado altista
   sustentado permanece não caracterizado.

**Limitações técnicas:**
5. **Duas features projetadas não são geradas** (Bandas de Bollinger) por incompatibilidade de
   nomes na versão beta do `pandas-ta` — o modelo em produção opera com 18 das 20 variáveis
   previstas. A correção foi implementada na V6, mas o modelo resultante ainda não superou o
   vigente.
6. **Calibração imperfeita da confiança.** O *edge* medido não cresce monotonicamente com a
   confiança declarada pelo modelo, o que limita o uso da alavancagem proporcional.
7. **Retorno modesto:** o sistema preserva capital e supera consistentemente o BTC, mas não
   atinge a meta de lucro diário expressivo.
8. **A execução real ainda não foi exercitada.** A ponte foi implementada (seção 13.1) e
   validada nos componentes que independem de autenticação, mas nenhuma ordem foi
   efetivamente emitida — pendente de credenciais da Futures Testnet. O bot legado
   (`main.py`) segue operando com a estratégia de *trend-following* da V4.

**Trabalhos futuros (Etapa 8, em andamento):**
- **Primeiro ciclo com ordens reais** na Testnet, iniciando em modo de simulação para
  auditoria das decisões antes de qualquer emissão.
- **Validação em Testnet** por período prolongado, comparando o desempenho observado com o
  previsto pelo backtest — em particular quanto a *slippage* e latência, não modelados na
  simulação.
- **Fortalecer a capacidade discriminativa** do modelo — identificada como o gargalo real
  após a série de experimentos da V6.
- **Monitoramento de divergência** entre desempenho ao vivo e esperado, como gatilho para
  retreinamento (o retreino em calendário fixo foi descartado empiricamente).

---

## 17. Conclusão

O Trader.AI demonstra que a combinação de **regras técnicas determinísticas** com um
**classificador neural sequencial** produz um sistema de negociação que **opera nos dois
sentidos do mercado e preserva capital em condições adversas** — superando a estratégia
passiva de comprar e segurar BTC em 13 a 29 pontos percentuais, sem jamais ter sido
liquidado.

A contribuição da rede neural, frequentemente assumida em trabalhos da área, foi aqui
**quantificada por ablação**: sua remoção torna o sistema deficitário em todos os períodos
analisados, o que atribui a ela **+8,0 pontos percentuais** no período de teste. Este
resultado sustenta empiricamente a tese central — a arquitetura híbrida supera cada
componente isolado.

Do ponto de vista do rigor acadêmico, o projeto adota **avaliação walk-forward** sem
vazamento temporal, incorpora um **conjunto de validação virgem** obtido após o congelamento
das decisões, define **critérios de promoção anteriores à observação dos resultados** e
**documenta seus resultados negativos** com a mesma ênfase dos positivos. Das nove hipóteses
formuladas ao longo do desenvolvimento, quatro foram refutadas — e essas refutações foram
determinantes para localizar o gargalo real do sistema.

A investigação também produziu achados de valor transferível: a **alavancagem comporta-se
como multiplicador de regime, não de competência preditiva**; e **métricas de desempenho
devem ser aferidas no contexto real de uso**, sob pena de conduzirem à conclusão oposta — como
quase ocorreu com o componente mais valioso da arquitetura.

O sistema constitui, portanto, uma base sólida e cientificamente defensável, com limitações
explicitadas e um caminho de evolução claramente delimitado pela evidência acumulada.

---

*Documento atualizado a partir do estado do código em julho/2026. Projeto estritamente
educacional e experimental; não constitui recomendação de investimento.*
