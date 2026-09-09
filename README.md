
#  TRADER.AI

> **Sistema de negociação algorítmica autônoma para criptoativos, e o instrumento de medição construído para avaliá-lo.**
>
> *Trabalho de Conclusão de Curso — Ciência da Computação.*

![Status](https://img.shields.io/badge/Status-V9%20em%20opera%C3%A7%C3%A3o%20(testnet)-2E7D5B?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.12-yellow?style=for-the-badge&logo=python&logoColor=white)
![Testes](https://img.shields.io/badge/Testes-111%20passando-2E7D5B?style=for-the-badge)
![Dados](https://img.shields.io/badge/Dados-Polars%20%7C%20Binance-F3BA2F?style=for-the-badge&logo=binance&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

---

##  O que este projeto é — e o que ele não é

Nove gerações de sistemas de negociação foram construídas e avaliadas entre 2025 e 2026.
A conclusão do trabalho, sustentada por medições reproduzíveis, é dupla:

**Existe sinal preditivo real.** O classificador de regime alcança **53,69% ± 0,90** de
acurácia balanceada na direção do mercado em horizonte de três dias (estatística t = 3,40),
sob validação *walk-forward* com embargo de 210 dias. É o primeiro resultado preditivo do
projeto a sobreviver a esse critério.

**E ele é insuficiente.** O limiar para igualar a estratégia passiva de comprar e manter,
medido para a mesma arquitetura, é de **58%**. O sistema resultante gera retorno positivo e
inferior ao da alternativa que não requer sistema algum.

> **O Trader.AI não é um sistema de geração de lucro.** A evidência acumulada em nove etapas
> de avaliação indica que preço e volume, isoladamente, não sustentam essa finalidade no
> universo e no horizonte investigados. O que se construiu, e que se oferece como
> contribuição, é um **instrumento de medição**.

---

##  As duas contribuições metodológicas

Ambos os instrumentos são computacionalmente triviais e estavam ausentes de todo o
desenvolvimento anterior. Cada um reclassificou um resultado que o projeto considerava
estabelecido.

### 1. O referencial do passeio aleatório — estabelece o **piso**

Quando uma posição é encerrada por barreiras fixas de ganho e perda, um ativo sem
previsibilidade alguma atinge o alvo numa proporção conhecida das vezes. Sem esse
referencial, uma taxa de acerto não distingue seleção competente de exposição favorável
ao mercado.

**O que reclassificou:** o retorno de ~16% obtido em operação real pela V1, que motivou
todo o projeto, era exposição ao mercado — não capacidade de seleção. Medido sobre 8.702
operações, o *edge* da estratégia determinística é de 0,000% ± 0,007%.

### 2. O teste de deslocamento da grade — estabelece a **dispersão**

A mesma configuração, sobre os mesmos dados e o mesmo intervalo de datas, alterando
exclusivamente o minuto em que a grade de avaliação começa:

| Início da avaliação | Retorno (jan/2025 – jul/2026) |
|---|---|
| 00:00 | **−4,90%** |
| 00:07 | **+8,24%** |

Desvio padrão entre execuções que deveriam ser idênticas: **5,32 a 16,54 pontos
percentuais** — maior que a diferença entre as configurações que o ranking pretendia ordenar.

**O que reclassificou:** quatorze configurações avaliadas em oito deslocamentos cada, sobre
5,5 anos. Sob correção de Bonferroni, **nenhuma apresentou retorno positivo distinguível
de zero**. A única que atravessou o limiar o fez por ser consistentemente negativa.

> **Consequência prática:** resultados de *backtest* reportados como valor único não
> constituem evidência de desempenho. Isso inclui os resultados das gerações V1 a V7 deste
> próprio repositório — ver a seção **Trajetória das versões**.

---

##  Estado atual — V9 em operação

Em produção na *testnet* da Binance desde **05/09/2026**, sob agendamento diário
(`systemd timer`, 03:30 UTC).

| | |
|---|---|
| **Estratégia** | Comprado nos 6 ativos em peso igual a 1x, **ou** em caixa |
| **Decisão** | Uma a cada 3 dias, pelo classificador de regime |
| **Sem** | Alavancagem, venda a descoberto, *stop*, seleção de ativo |
| **Universo** | BTC, ETH, SOL, BNB, XRP, AVAX |

Cada ausência corresponde a um mecanismo medido e refutado, ou a um incidente:

| Mecanismo ausente | Motivo |
|---|---|
| Alavancagem | De 1x para 2x o retorno **cai**; em 20x, 6 de 6 execuções zeraram a conta |
| Venda a descoberto | O motor Bear nunca superou o custo de transação em medição alguma |
| *Stop* na corretora | Falhou em 19/08/2026, deixando posições descobertas |
| Seleção de ativo | Peso igual venceu toda tentativa de seleção |

### O que esperar

Medido: **~2,2% ao mês**, rebaixamento máximo de 52,8%, **erro de ±60 pontos percentuais** —
não distinguível de um controle aleatório. Comprar e manter os mesmos ativos rendeu
**+215,5%** no mesmo período de 3,5 anos, contra +146,1% da sobreposição.

> **O saldo subir não é evidência de que o sistema funciona.** A carteira mantém 98% de
> exposição: ela acompanha o mercado por construção. A evidência pertinente é a taxa de
> acerto das decisões de regime, e sua avaliação exige um número de decisões que só o tempo
> produz.

---

##  Arquitetura da V9

Três componentes de responsabilidade isolada. A decisão de aprendizado de máquina sai do
nível da operação individual e vai para o nível do contexto macroeconômico.

* **Motor Bull** — compra correções em tendência de alta. Nunca vende. Não conhece o saldo,
  a corretora, nem o outro motor.
* **Motor Bear** — vende repiques em tendência de baixa. Nunca compra.
* **Oráculo de Regime** — decide qual motor está em serviço. Uma decisão por período; não
  escolhe ativo, momento nem preço.

### As três garantias estruturais

| Garantia | Como |
|---|---|
| **Não vê o futuro** | A estratégia recebe um objeto que contém apenas fatias terminadas no instante corrente. Não é disciplina do programador — é impossível por construção |
| **Um caminho de código** | Simular e operar diferem apenas em qual adaptador de corretora está acoplado (`papel.py` ou `carteira.py`) |
| **Nenhum número sozinho** | `avaliacao/robustez.py` percorre todas as fases da grade e reporta média com erro padrão |

Verificado por **111 testes automatizados**, entre os quais o de causalidade — que recalcula
cada indicador usando apenas dados anteriores a um instante e compara com o cálculo sobre a
série completa (divergência medida: `0,00e+00`).

---

##  Estrutura do Projeto

```text
Trader.AI/
│
│  ══ V9 — sistema atual, reconstruído do zero ═════════════════════════
├── trader-v9/
│   ├── nucleo/
│   │   ├── tipos.py               # Barra, Sinal, Posicao, Regime (dataclasses congeladas)
│   │   └── protocolos.py          # Protocol: Motor, Oraculo, Corretora, VisaoDeMercado
│   ├── dados/
│   │   ├── indicadores.py         # RSI, MACD, Bollinger, ATR — escritos à mão, testados
│   │   ├── visao.py               # Historico (tem tudo) e VisaoDeMercado (só o passado)
│   │   ├── fonte.py               # Carrega parquets e traduz para o vocabulário da V9
│   │   └── atualizar.py           # Completa o histórico local com o que já aconteceu
│   ├── motores/
│   │   ├── base.py                # Contrato comum + validação de lado
│   │   ├── bull.py                # Especialista de alta
│   │   └── bear.py                # Especialista de baixa
│   ├── oraculo/
│   │   ├── features.py            # 30 atributos macro (4h e diário, barras fechadas)
│   │   ├── modelo.py              # Walk-forward com embargo; acurácia balanceada
│   │   ├── teto.py                # Oráculo perfeito, moeda e controles fixos
│   │   ├── ruidoso.py             # Oráculo de acurácia controlada (curva de alvo)
│   │   └── classificador.py       # O modelo treinado, para uso ao vivo
│   ├── execucao/
│   │   ├── risco.py               # Sinal -> Ordem: tamanho e barreiras em ATR
│   │   ├── papel.py               # Corretora simulada
│   │   └── carteira.py            # Corretora real: comprado ou caixa, e nada mais
│   ├── avaliacao/
│   │   ├── replay.py              # O único laço — produção e simulação usam este
│   │   ├── metricas.py            # Retorno, rebaixamento, comprar-e-manter
│   │   └── robustez.py            # Varredura de fases + correção de Bonferroni
│   ├── app/                       # Programas executáveis (ver seção de uso)
│   ├── testes/                    # 111 testes, inclui o de não-vazamento
│   ├── modelos/                   # Modelo treinado (.joblib) — treinar na máquina que usa
│   ├── config/v9.toml             # Todo número ajustável do projeto
│   └── deploy/                    # systemd + LEIA-ME do servidor
│
│  ══ V8 — instrumento de medição sobre o código de produção ══════════
├── v8_simulador.py                # Replay minuto a minuto chamando as funções de produção
├── v8_rank.py                     # Ranking de configurações
├── v8_rank_honesto.py             # O mesmo, com margem de erro (o único defensável)
│
│  ══ V7 — rotulagem por barreiras triplas e torneio de modelos ═══════
├── v7_dataset.py                  # Base supervisionada: 6.619 eventos, 4 rotulagens
├── v7_modelos.py                  # Torneio: Dummy, LogReg, RF, LightGBM, XGBoost, CatBoost
├── v7_regras_por_regime.py        # Edge por BULL/LATERAL/BEAR
├── v7_tendencia_diaria.py         # Seguir tendência em prazo diário
├── v7_edge_funding.py             # Edge da taxa de financiamento
├── v7_coleta_alternativa.py       # Coleta diária: funding, open interest, long/short
│
│  ══ V5–V6 — rede neural e ferramentas de investigação ═══════════════
├── v5_model.py                    # BiLSTM + Attention (3 classes)
├── v5_data_prep.py                # Features (18) e rótulos direcionais
├── v5_train.py                    # Treino: Focal Loss, early stopping, --resume
├── v5_backtest.py                 # Backtest híbrido LONG/SHORT com TP/SL intrabar
├── v5_live.py                     # Paper trading
├── v6_executor.py                 # Ordens na Binance Futures (parado desde 05/09/2026)
├── v6_ciclo.py                    # Ponte estratégia -> execução
├── v6_auditoria.py                # Consulta a corretora: saldo, ordens, proteção, resultado
├── v6_ablacao.py                  # Contribuição da rede neural (com vs sem)
├── v6_*.py                        # Demais ferramentas de análise e experimentos
│
│  ══ Bot V4 — API REST + WebSocket (histórico) ═══════════════════════
├── main.py, execution.py, strategy.py, market_state.py, binance_stream.py
│
│  ══ ETL ═════════════════════════════════════════════════════════════
├── download_binance_data.py       # Extração do Binance Vision (11 pares)
├── processar_dados.py             # CSV -> Parquet
│
│  ══ Documentação ════════════════════════════════════════════════════
├── README.md                      # Este arquivo
├── COMANDOS.md                    # Catálogo de comandos do projeto
├── DOCUMENTACAO_TCC.md            # Documentação técnica completa
├── METODOLOGIA_EXPERIMENTAL.md    # Protocolo científico e experimentos
│
│  ══ Gerados em runtime (não versionados) ════════════════════════════
├── relatorios/                    # Toda saída: logs, relatórios, resultados
├── data/                          # Datasets históricos .parquet
├── *.pth                          # Modelos neurais treinados
└── .env                           # Chaves da API
```

> **Convenção:** toda saída gerada em execução vai para `relatorios/` — uma pasta, uma regra
> no `.gitignore`. O conteúdo é 100% regenerável, pois o código que o produz está versionado.

---

##  Instalação

### Pré-requisitos
* Python 3.12
* Git
* (Opcional, apenas para treinar a rede neural das V5/V6) GPU NVIDIA com CUDA

### Ambiente

```bash
git clone https://github.com/FilipeSJ1002/Trader.AI.git
cd Trader.AI
python -m venv venv
.\venv\Scripts\activate          # Windows
source venv/bin/activate         # Linux / macOS
pip install -r requirements.txt
pip install -r trader-v9/requirements-v9.txt
```

### Credenciais

A Binance mantém **testnets separadas** para spot e futuros, com contas e chaves distintas.
Usar chave de spot em endpoint de futuros retorna `APIError -2015`.

Crie um `.env` na raiz com:

```
BINANCE_FUTURES_API_KEY=sua_chave_de_futuros
BINANCE_FUTURES_SECRET_KEY=sua_secret_de_futuros
```

Obtenha-as em `testnet.binancefuture.com` (aba "API Key" no rodapé).

---

##  Uso — V9

Todos os comandos partem de `trader-v9/` com `PYTHONPATH=.`.

### Semear o histórico (uma vez)

Baixa 2.600 dias em barras de 4h. **Não reduza esse número:** treinar com 257 dias derruba
a acurácia para 50,07% — moeda.

```bash
cd trader-v9
PYTHONPATH=. python -m app.semear
```

> As features do oráculo saem exclusivamente das visões diária e de 4h. Reconstruindo o
> histórico a partir de barras de 4h, os 16 atributos resultam **idênticos** com 240 vezes
> menos dados — de 3,7 horas de download para cerca de um minuto.

### Treinar o classificador

Treine **na máquina que vai usá-lo**: modelo em *pickle* não atravessa versões do
scikit-learn.

```bash
PYTHONPATH=. python -m app.treinar_oraculo
```

Confira a linha de treino: deve indicar **~2.100 dias**. Se indicar 257, o histórico está
curto e o modelo não vale nada.

### Operar

```bash
PYTHONPATH=. python -m app.vivo                 # dry-run: decide e mostra, não envia
PYTHONPATH=. python -m app.vivo --armar         # envia ordens (testnet)
PYTHONPATH=. python -m app.vivo --armar --real  # produção (exige confirmação digitada)
```

**Leia a saída do dry-run antes de armar.** Ela já revelou um defeito de precisão numérica
impresso em texto claro que passou despercebido — a quantidade era `0.009600000000000001`.

O programa aborta se os dados não estiverem em dia: decidir sobre um mercado que já mudou
é pior que não decidir.

### Medir

```bash
PYTHONPATH=. python -m pytest testes -q         # 111 testes
PYTHONPATH=. python -m app.cli teto             # teto do projeto, com margem de erro
PYTHONPATH=. python app/curva_acuracia.py       # quanto rende cada nível de acurácia
PYTHONPATH=. python app/treinar.py              # o classificador, fora da amostra
PYTHONPATH=. python app/horizontes.py           # o regime é previsível em algum prazo?
PYTHONPATH=. python app/alavancagem.py          # a alavancagem ajuda?
PYTHONPATH=. python app/sobreposicao.py         # usar o sinal para proteger em vez de gerar
PYTHONPATH=. python app/confianca.py            # a confiança do modelo carrega informação?
PYTHONPATH=. python app/janela_de_treino.py     # quanto histórico o modelo precisa?
PYTHONPATH=. python app/figuras.py --saida .    # figuras do artigo
```

### Servidor

Passo a passo completo em [`trader-v9/deploy/LEIA-ME.md`](trader-v9/deploy/LEIA-ME.md).

```bash
sudo cp trader-v9/deploy/trader-v9.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now trader-v9.timer
sudo journalctl -u trader-v9 -n 60 --no-pager
```

---

##  Uso — ferramentas de pesquisa (V5 a V8)

Preservadas e funcionais. Produziram os resultados das etapas 1 a 7 do artigo.

### Auditoria da corretora

```bash
python v6_auditoria.py --dias 30
```

Mostra saldo, posições, ordens (comuns **e** condicionais), verificação de proteção e
resultado financeiro. Fatia as consultas em janelas de 7 dias — a API limita o intervalo, e
pedir 30 dias devolvia silenciosamente apenas os 7 primeiros.

### Simulação sobre o código de produção

```bash
python v8_simulador.py --verificar               # prova que não vê o futuro
python v8_simulador.py --de 2021-01-01 --ate 2026-07-25
python v8_rank_honesto.py                        # ranking com margem de erro
```

### Rede neural (V5/V6)

```bash
python v5_data_prep.py --dual
python v5_train.py --data data_v6 --model-out v6_model.pth --resume
python v5_backtest.py --model v5_model_b.pth --val
python v6_ablacao.py                             # contribuição da rede neural
```

### ETL

```bash
python download_binance_data.py    # Binance Vision, 11 pares desde 2019
python processar_dados.py          # CSV -> Parquet
```

---

##  Trajetória das versões

| Geração | Abordagem | O que estabeleceu |
|---|---|---|
| **V1** | Regras determinísticas (RSI, MACD, Bollinger) | +16% no 1º mês real — posteriormente reclassificado como exposição ao mercado |
| **V2–V3** | Gestão de risco; busca de equilíbrio | Segurança sim; retorno marginal ou negativo |
| **V4** | ML clássico de árvores | Infraestrutura de ML; lucros baixos |
| **V5** | BiLSTM + Attention; estratégia híbrida | Preservação de capital em regime adverso |
| **V6** | Investigação sistemática; execução real | Enriquecimento de features refutado; ordens com proteção na corretora |
| **V7** | Barreiras triplas; torneio de modelos | Alvo maior dilui a taxa — depois refutado pela curva de capital |
| **V8** | Simulação sobre o código de produção | **Nenhuma configuração distinguível de zero** (14 testadas) |
| **V9** | Dois motores + oráculo de regime | **Sinal real de 53,69% ± 0,90 (t = 3,40), abaixo do limiar de 58%** |

### Sobre os resultados das gerações V1 a V7

Os resultados dessas etapas foram obtidos por **execução única, sem estimativa de
dispersão**. Nenhum deles sobrevive ao critério estabelecido pela V8 — incluindo os +16% da
V1, o ganho de 2,2% que motivou a adoção da configuração então vigente, e as comparações
entre gerações.

Isso **não invalida** os resultados apoiados em amostras grandes de operações: a expectância
líquida medida sobre 8.702 operações e o teste do passeio aleatório permanecem válidos.
Invalida especificamente as comparações de curva de capital entre configurações.

O estudo de ablação da rede neural (V6, +8,0 p.p. no teste) **não foi reavaliado** sob o
critério de robustez. Sua magnitude está na fronteira do ruído medido, e ele deve ser lido
como não verificado, não como estabelecido.

---

##  Achados transversais

**A escala temporal da barreira de risco domina o resultado.** O ATR calculado sobre candles
de um minuto corresponde a 0,073% do preço no Bitcoin; um *stop* a 1,5 desses valores fica
dentro do ruído. Mantida toda a demais configuração:

| Escala do ATR | Retorno (6 meses) | Operações |
|---|---|---|
| 1 minuto | **−60,8%** | ~10.000 |
| Diário | **+3,7%** | ~50 |

**A alavancagem não compensa a ausência de vantagem — amplifica-a.**

| Alavancagem | Retorno (3,5 anos) | Rebaixamento | Contas zeradas |
|---|---|---|---|
| **1x** | **+13,9% ± 7,6** | −32,6% | 0 / 6 |
| 2x | +12,3% ± 14,2 | −56,6% | 0 / 6 |
| 3x | −4,1% ± 17,3 | −73,5% | 0 / 6 |
| 20x | −100,0% | −100% | **6 / 6** |

**A confiança do modelo não carrega informação.** Nos 10% de dias em que o classificador
atribui maior probabilidade, a acurácia é *inferior* à média em dois dos três modelos
avaliados (50,64% contra 51,18% na regressão logística).

**O limiar de viabilidade não é atributo do modelo, mas da arquitetura que o emprega.** A
mesma capacidade de 53,69% exige 63% quando o sinal deve gerar todo o retorno, e 58% quando
lhe cabe apenas modular a exposição a uma posição comprada.

---

##  Roadmap

- [x] **Etapas 1–4:** Arquitetura base, ETL, WebSockets, execução de ordens
- [x] **Etapa 5:** Rede neural direcional e estratégia híbrida bidirecional
- [x] **Etapa 6:** Validação *walk-forward* e *paper trading*
- [x] **Etapa 7:** Investigação sistemática dos limites (ablação, calibração, universo, stops)
- [x] **Etapa 8:** Execução real na Binance Futures com proteção na corretora
- [x] **Etapa 9:** Instrumento de medição com margem de erro; reconstrução V9; operação contínua
- [ ] **Novembro de 2026:** Dados alternativos — a única alavanca técnica restante

### A pergunta em aberto

O coletor acumula taxa de financiamento, *open interest*, razão long/short e fluxo de
*taker* desde 20/08/2026. Por volta de novembro haverá amostra utilizável, e a pergunta é
fechada: **essas features levam os 53,69% além de 58%?**

```bash
cd trader-v9 && PYTHONPATH=. python app/horizontes.py
```

São 4,3 pontos percentuais de distância. Dados de posicionamento de mercado são
qualitativamente diferentes de preço, que é o único insumo testado até aqui. Se passarem, há
sistema. Se ficarem em 54%, a conclusão está fechada — e a resposta sai numa tarde.

---

##  O que não repetir

Registrado porque foi medido, não porque foi suposto:

* **Aumentar a alavancagem.** De 1x para 2x o retorno cai; em 20x a conta zera.
* **Operar apenas nos dias de alta confiança.** A probabilidade emitida não discrimina.
* **Ajustar TP/SL/parâmetros procurando algo melhor.** A variação entre ajustes é menor que
  a variação entre relógios.
* **Rede neural decidindo operação minuto a minuto.** Refutada na V6 e no torneio da V7.
* **Aceitar número de *backtest* sem barra de erro.**
* **Semear pouco histórico.** 257 dias derrubam o modelo para 50,07%.

---

## Autor

Desenvolvido por **Filipe Spirlandeli Junqueira**.

---
> Este projeto é estritamente educacional e experimental. Os retornos citados são resultados
> de simulação sobre dados históricos e de operação em ambiente de testes com capital
> fictício — não constituem previsão de resultado futuro nem recomendação de investimento.
> O autor não se responsabiliza por perdas financeiras decorrentes do uso deste software em
> contas reais.
