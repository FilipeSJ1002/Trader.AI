# Evolução do Sistema: das Regras Determinísticas à Rede Neural Híbrida

> Seção redigida para o artigo do TCC (tom acadêmico, alinhado ao texto existente).
> Trechos entre colchetes `[...]` indicam dados que o autor deve confirmar/preencher.

---

O desenvolvimento do Trader.AI foi conduzido de forma **incremental e experimental**, organizado
em cinco gerações principais (V1 a V5). Cada versão representa uma hipótese distinta sobre como
equilibrar **agressividade** (maximização de lucro) e **segurança** (preservação de capital), e
cada uma forneceu aprendizados empíricos que fundamentaram a versão seguinte. Esta seção descreve
essa trajetória, destacando a proposta, os pontos fortes, as limitações e os resultados de cada
geração.

## Versão 1 — Estratégia de Confluência Determinística

A primeira versão fundamentou-se exclusivamente em **análise técnica quantitativa**, sem qualquer
componente de aprendizado de máquina. O sistema computava indicadores clássicos — Índice de Força
Relativa (RSI), convergência/divergência de médias móveis (MACD) e Bandas de Bollinger — e os
combinava em um **sistema de pontuação (score)** de compra e venda. Quando a confluência de sinais
ultrapassava um limiar, uma ordem era disparada.

Tratou-se da configuração **mais agressiva** já desenvolvida no projeto. Seu principal ponto forte
foi a **transparência** (cada decisão era rastreável a uma regra explícita) e a **alta
rentabilidade em mercados de alta**: operada em ambiente real entre abril e maio de 2026, a V1
obteve um retorno financeiro de aproximadamente **16% no primeiro mês**.

Entretanto, esse resultado revelou-se **dependente do regime de mercado**: o período coincidiu com
uma forte valorização do Bitcoin. A limitação estrutural da estratégia tornou-se evidente quando o
mercado inverteu para uma tendência de baixa — a lógica de "comprar correções" (*buy the dip*)
passou a acumular prejuízos, agravados por uma **gestão de risco ainda imatura**, na qual o
mecanismo de *stop loss* não atuava de forma eficaz. Conclui-se que a V1 era lucrativa, porém
**frágil e não generalizável** para diferentes condições de mercado.

## Versão 2 — Gestão de Risco e Estabilização

Em resposta às falhas de segurança da V1, a segunda versão concentrou-se na **gestão de risco**.
Foram introduzidos mecanismos como o dimensionamento de stops com base na volatilidade real do
ativo (*Average True Range* — ATR), *stop móvel* (*trailing stop*) e a persistência do estado das
operações em banco de dados (SQLite).

O ponto forte foi inequívoco: o sistema tornou-se **substancialmente mais seguro**, eliminando as
perdas catastróficas observadas na V1. Contudo, a calibração resultou **excessivamente
conservadora** — os filtros de proteção restringiam demasiadamente as entradas, reduzindo a
frequência de operações e, consequentemente, o retorno. A V2 foi a versão **mais segura** do
projeto, mas ao custo de uma rentabilidade reduzida [confirmar/quantificar resultado].

## Versão 3 — Busca pelo Equilíbrio

A terceira versão buscou um **equilíbrio entre a agressividade da V1 e a segurança da V2**.
Incorporou um filtro de tendência de longo prazo (média móvel exponencial de 200 períodos —
EMA 200), um gatilho de entrada sensível a picos de volatilidade (ATR) e uma relação
risco/retorno ambiciosa de **1:6**, viabilizada por um *stop loss* muito ajustado (*micro-stop*).

Embora conceitualmente promissora, a configuração **não atingiu o equilíbrio pretendido**. O
*stop* excessivamente apertado, necessário para sustentar a relação risco/retorno de 1:6, tornava
as posições vulneráveis ao ruído natural do mercado, disparando saídas prematuras de forma
recorrente. O resultado líquido foi **negativo** [confirmar/quantificar prejuízo], evidenciando que
a simples recalibração de parâmetros das regras determinísticas era insuficiente para resolver o
problema de adaptação de regime.

## Versão 4 — Primeira Incursão em Inteligência Artificial

Reconhecida a limitação das regras fixas, a quarta versão introduziu o **aprendizado de máquina**
como núcleo decisório. Foram empregados modelos baseados em **árvores de decisão** — inicialmente
*Random Forest* e, posteriormente, *Gradient Boosting* via biblioteca **LightGBM** — operando como
filtro preditivo. A arquitetura contemplava um modelo de entrada (probabilidade de sucesso da
operação) e um modelo de saída, além de um mecanismo pioneiro de **auto-retreino** e aprendizado a
partir das próprias operações reais.

O principal mérito da V4 foi **estabelecer toda a infraestrutura de aprendizado de máquina** do
projeto: engenharia de *features*, pipeline de treinamento e avaliação. No entanto, os resultados
ficaram **aquém do esperado**, com lucros baixos. Duas limitações explicam o desempenho: primeiro,
modelos de árvore de decisão **não capturam adequadamente a dependência temporal sequencial** de
uma série de preços, tratando cada instante de forma relativamente isolada; segundo, a IA operava
de **forma isolada**, sem o respaldo das regras técnicas, o que a tornava sensível a ruído. A V4
demonstrou que a mera substituição das regras por aprendizado de máquina, sem a arquitetura
adequada, não era suficiente.

## Versão 5 — Arquitetura Neural Híbrida (versão atual)

A versão atual nasce da síntese de todos os aprendizados anteriores. Duas decisões a definem.

A primeira foi a adoção de uma **rede neural profunda** — especificamente uma **BiLSTM
(Bidirectional Long Short-Term Memory) com mecanismo de Attention** — em substituição às árvores de
decisão da V4. Diferentemente destas, a arquitetura recorrente é **projetada para séries
temporais**, capturando padrões ao longo de toda a janela de observação (120 minutos), enquanto o
mecanismo de atenção pondera os instantes mais relevantes para a decisão.

A segunda, e mais importante, foi a transição de um modelo isolado para uma **estratégia híbrida**.
Em vez de delegar toda a decisão à IA (como na V4), a V5 reaproveita os **cálculos matemáticos da
V1** como **gatilho de oportunidade**: as regras técnicas identificam *quando* há uma condição
favorável, e a rede neural atua como **filtro de direção**, decidindo *se* e *em qual sentido*
(alta ou queda) a operação deve ocorrer. Apenas quando ambos concordam a posição é aberta. Soma-se
a isso a operação **bidirecional** (posições compradas e vendidas), um filtro de regime de mercado
e uma gestão de risco assimétrica (relação risco/retorno de 2:1 com verificação de *stop* a cada
minuto).

Durante o desenvolvimento da V5, conduziu-se um **experimento controlado** comparando dois modelos
de mesma arquitetura, diferenciados apenas pela rotulagem dos dados de treino: um **modelo
equilibrado**, concebido para lucrar tanto em alta quanto em baixa, e um **modelo especialista em
quedas**, mais sensível à detecção de movimentos de baixa. O **modelo especialista venceu de forma
expressiva**, apresentando resultados muito superiores ao modelo equilibrado no período de teste —
coerente com o objetivo central do projeto de operar com segurança em mercados adversos.

Avaliada pela metodologia **walk-forward** (sem vazamento de dados futuros), a V5 demonstrou sua
capacidade de **preservação de capital**: em um período de teste de mercado de baixa, obteve
retorno de **+2,0%** enquanto a estratégia passiva de *buy-and-hold* do Bitcoin recuou **−26,8%** —
uma vantagem de aproximadamente **29 pontos percentuais** —, sem registrar uma única liquidação. A
principal limitação remanescente é a **frequência ainda baixa de sinais de altíssima convicção**, o
que mantém os retornos modestos e motivou a investigação sistemática conduzida na versão seguinte.

## Versão 6 — Investigação dos Limites e Validação por Ablação

Se as versões anteriores foram guiadas pela busca de uma arquitetura que
funcionasse, a sexta volta-se para uma questão diferente e mais madura:
**quanto cada componente do sistema realmente contribui, e onde estão seus
limites?** Trata-se de uma etapa predominantemente investigativa, na qual cinco
hipóteses de aprimoramento foram formuladas e submetidas a teste sistemático.

Duas decisões metodológicas sustentam esta etapa. A primeira foi a adoção de um
**conjunto de validação virgem**: dados de junho e julho de 2026 obtidos
*posteriormente* ao congelamento de toda a arquitetura, parâmetros e limiares.
Nenhuma decisão de projeto pôde ser influenciada por eles, o que os torna a
aproximação mais fiel de operação real sem exposição de capital. A segunda foi a
definição de **critérios de promoção anteriores à observação dos resultados**,
impedindo que a régua fosse ajustada em favor da conclusão desejada.

O experimento central da etapa foi um **estudo de ablação**: executar a
estratégia de forma idêntica, porém com o filtro neural neutralizado, isolando
assim sua contribuição.

| Período | Com rede neural | Sem rede neural | Contribuição |
|---|---|---|---|
| Validação (jul–dez/2025) | −2,0% | −4,1% | +2,1 p.p. |
| Teste (jan–jul/2026) | **+2,0%** | **−6,0%** | **+8,0 p.p.** |
| Holdout virgem (jun–jul/2026) | +0,2% | −0,6% | +0,8 p.p. |

O resultado sustenta empiricamente a tese central do trabalho: **sem a rede
neural o sistema é deficitário em todos os períodos analisados**. Ela descarta
aproximadamente 75% dos sinais gerados pelo componente determinístico e eleva a
taxa de acerto de 36,6% para 41,2%.

Esse achado ganha relevância adicional quando confrontado com uma medição
anterior. Ao avaliar isoladamente a capacidade preditiva da rede — sua taxa de
acerto direcional em janelas arbitrárias do mercado — obteve-se valor próximo de
0,50, equivalente ao acaso, o que sugeriria que o componente era dispensável. A
ablação demonstrou o oposto. A explicação reside na natureza da tarefa: a rede
**não** prevê a direção do mercado a partir do zero; ela **discrimina entre
candidatos previamente filtrados pelos indicadores técnicos**. São problemas
distintos, e apenas o segundo corresponde ao seu uso efetivo. Daí a lição
metodológica registrada: métricas devem ser aferidas no contexto real de
aplicação, e a ablação constitui o instrumento adequado para atribuir valor a um
componente.

Das demais hipóteses, três foram refutadas. A **ampliação do universo de ativos**
(de seis para onze) degradou o desempenho em todos os períodos, apesar do maior
número de oportunidades. Os **stops adaptativos por volatilidade** — motivados
pela observação de que os ativos mais voláteis concentravam as perdas — não
superaram o stop percentual fixo em nenhuma calibração testada; demonstrou-se,
inclusive, que as mesmas operações eram encerradas independentemente da largura
do stop, descartando a hipótese de acionamento por ruído. O **enriquecimento do
conjunto de atributos** (de 18 para 26 variáveis, incluindo Bandas de Bollinger,
regime de mercado e sazonalidade) não superou o modelo anterior em erro de
validação.

A quinta hipótese produziu o achado transversal mais relevante: **a alavancagem
comporta-se como multiplicador de regime, não de competência preditiva**. Em
mercado lateral, operar sem alavancagem alguma superou a configuração de produção
em 1,2 ponto percentual — resultado com explicação econômica direta, uma vez que
os custos de transação escalam com o valor nocional. Com margem preditiva
modesta, a alavancagem não melhora o valor esperado; amplifica o custo de fricção
e a variância. Uma variante condicionada à força da tendência apresentou soma
superior e variância substancialmente menor, embora com desempenho inferior em
mercado de forte tendência — configurando um compromisso entre retorno e
consistência, e não uma melhoria absoluta.

A V6, portanto, não elevou o retorno do sistema. Sua contribuição é de outra
natureza: **delimitou com precisão o que funciona, o que não funciona e por quê**,
eliminando quatro caminhos improdutivos e estabelecendo que o gargalo reside na
capacidade discriminativa do modelo — não em parâmetros de execução, universo de
ativos ou cronograma de retreinamento.

---

## Quadro-Resumo da Evolução

| Versão | Abordagem | Ponto Forte | Limitação | Resultado |
|--------|-----------|-------------|-----------|-----------|
| **V1** | Regras determinísticas (RSI, MACD, Bollinger) | Transparência; alta rentabilidade no *bull* | *Stop loss* ineficaz; dependente de regime | **+16% real** (1º mês), seguido de perdas no *bear* |
| **V2** | Gestão de risco (ATR, *trailing stop*) | Segurança; fim das perdas catastróficas | Conservadora demais | Retorno marginal |
| **V3** | Equilíbrio (EMA 200, R:R 1:6, *micro-stop*) | Filtro de tendência; R:R ambicioso | *Stop* apertado → saídas prematuras | Resultado negativo (*drawdown*) |
| **V4** | ML de árvores (*Random Forest* → *LightGBM*) | Infraestrutura de ML; auto-retreino | Árvores não capturam tempo; IA isolada | Lucros baixos |
| **V5** | Rede neural híbrida (BiLSTM+Attention + regras) | IA temporal + regras; bidirecional; *walk-forward* | Poucos sinais de alta convicção | **+2,0%** vs BTC **−26,8%** (~+29 p.p.), zero liquidações |
| **V6** | Investigação de limites e validação por ablação | Prova empírica da contribuição da IA (**+8,0 p.p.**); *holdout* virgem; 4 hipóteses refutadas | Não elevou o retorno; gargalo permanece na capacidade discriminativa | Confirmação da arquitetura; delimitação precisa dos limites |

### Resultados consolidados da configuração vigente

| Período | Trader.AI | *Buy-and-hold* BTC | Vantagem |
|---|---|---|---|
| Teste jan–jul/2026 (*bear market*) | **+2,0%** | −26,8% | +28,8 p.p. |
| *Holdout* virgem jun–jul/2026 | **+0,2%** | −13,0% | +13,2 p.p. |
| Validação jul–dez/2025 (lateral) | −2,0% | −18,2% | +16,2 p.p. |

A taxa de acerto manteve-se **idêntica (41,2%)** no período de teste e no *holdout*
virgem, indicando estabilidade do comportamento e não calibração fortuita.
Nenhuma liquidação foi registrada em qualquer período.

---

*Observação metodológica: internamente, o histórico de desenvolvimento foi mais granular que a
divisão em seis versões aqui apresentada (incluindo iterações intermediárias de estratégias de
*trend-following* e múltiplas "Fases" de engenharia). A consolidação em V1–V6 foi adotada por
clareza didática, agrupando as iterações por sua hipótese central predominante. O detalhamento
completo dos experimentos, com metodologia e resultados negativos, encontra-se em
`METODOLOGIA_EXPERIMENTAL.md`.*
