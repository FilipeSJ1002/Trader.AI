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
retorno de **+1,8%** enquanto a estratégia passiva de *buy-and-hold* do Bitcoin recuou **−15,9%** —
uma vantagem de aproximadamente **18 pontos percentuais** —, sem registrar uma única liquidação. A
principal limitação remanescente é a **frequência ainda baixa de sinais de altíssima convicção**, o
que mantém os retornos modestos e constitui a próxima fronteira de aprimoramento do sistema.

---

## Quadro-Resumo da Evolução

| Versão | Abordagem | Ponto Forte | Limitação | Resultado |
|--------|-----------|-------------|-----------|-----------|
| **V1** | Regras determinísticas (RSI, MACD, Bollinger) | Transparência; alta rentabilidade no *bull* | *Stop loss* ineficaz; dependente de regime | **+16% real** (1º mês), seguido de perdas no *bear* |
| **V2** | Gestão de risco (ATR, *trailing stop*) | Segurança; fim das perdas catastróficas | Conservadora demais | Lucro baixo [confirmar] |
| **V3** | Equilíbrio (EMA 200, R:R 1:6, *micro-stop*) | Filtro de tendência; R:R ambicioso | *Stop* apertado → saídas prematuras | Prejuízo [confirmar] |
| **V4** | ML de árvores (*Random Forest* → *LightGBM*) | Infraestrutura de ML; auto-retreino | Árvores não capturam tempo; IA isolada | Lucros baixos |
| **V5** | Rede neural híbrida (BiLSTM+Attention + regras) | IA temporal + regras; bidirecional; *walk-forward* | Poucos sinais de alta convicção | **+1,8%** vs BTC **−15,9%** (~+18 p.p.), zero liquidações |

---

*Observação metodológica: internamente, o histórico de desenvolvimento foi mais granular que a
divisão em cinco versões aqui apresentada (incluindo iterações intermediárias de estratégias de
*trend-following* e múltiplas "Fases" de engenharia). A consolidação em V1–V5 foi adotada por
clareza didática, agrupando as iterações por sua hipótese central predominante.*
