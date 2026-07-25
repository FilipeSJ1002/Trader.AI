# Metodologia Experimental — Trader.AI

> Registro do protocolo científico adotado no projeto e de todos os experimentos
> executados, com seus resultados — inclusive (e especialmente) os negativos.
>
> Autor: **Filipe Spirlandeli Junqueira** — TCC de Ciência da Computação.
> Última atualização: julho/2026.

---

## 1. Por que este documento existe

Em finanças quantitativas é trivial produzir um backtest lucrativo: basta ajustar
parâmetros até que os números fiquem bons. O resultado costuma evaporar em
operação real — fenômeno conhecido como *overfitting de backtest*.

Este documento registra as **regras que o projeto se impôs** para evitar esse
autoengano, e o histórico completo de experimentos, incluindo as hipóteses que
foram refutadas. Resultados negativos não são omitidos: eles delimitam o que o
sistema **não** é, e isso tem valor científico igual ou maior que os positivos.

---

## 2. O protocolo

### 2.1 Separação temporal dos dados

| Split | Período | Papel |
|---|---|---|
| **Treino** | até 30/06/2025 | Ajuste dos pesos da rede |
| **Validação** | jul–dez/2025 | **Escolha de hiperparâmetros e decisões de projeto** |
| **Teste** | 2026 em diante | Confirmação — uso restrito |
| **Holdout virgem** | jun–jul/2026 | Dados obtidos *após* todas as decisões |

O **holdout virgem** merece destaque: esses candles foram baixados em 25/07/2026,
depois que toda a arquitetura, parâmetros e thresholds já estavam congelados.
Nenhuma decisão do projeto pôde ser influenciada por eles — é a aproximação mais
honesta possível de operação real sem arriscar capital.

### 2.2 Regras invioláveis

1. **Parâmetros são escolhidos olhando apenas a validação.** O teste e o holdout
   são usados uma única vez, ao final, para confirmar.
2. **O critério de promoção é definido antes de ver os números.** Mudar a régua
   depois do resultado é o autoengano que o protocolo existe para impedir.
3. **Uma variável por experimento.** Quando várias mudam juntas, não se sabe qual
   funcionou.
4. **Resultados negativos são documentados**, com a mesma ênfase dos positivos.
5. **Números só entram na documentação se verificados no código ou no artefato de
   execução** — nunca de memória.

### 2.3 Realismo do simulador

O backtest incorpora as fricções que a maioria das simulações ignora:

- **Taxas** de 0,04% por lado (futuros Binance), escalando com o notional
- **Verificação intrabar** de TP/SL minuto a minuto (não apenas no fechamento)
- **Prioridade ao stop loss** quando TP e SL ocorrem no mesmo candle (conservador)
- **Simulação de liquidação** com margem de manutenção
- **Uma posição por ativo**, com bloqueio de reentrada enquanto aberta
- Comparação obrigatória contra dois *benchmarks*: comprar-e-segurar BTC e manter
  o capital em dólar

---

## 3. Experimentos executados

### 3.1 Rotulagem: modelo equilibrado vs. especialista em quedas

**Pergunta:** a definição dos rótulos de treino altera materialmente o resultado?

Dois modelos, arquitetura idêntica, diferindo apenas nos limiares que definem
ALTA e QUEDA:

| Modelo | Limiares | Teste 2026 |
|---|---|---|
| A — equilibrado | ALTA/QUEDA simétricos em 0,4% | +0,3% |
| **B — especialista em quedas** | ALTA 0,8% / QUEDA 0,4% | **+11,1%** |

**Resultado:** o modelo B venceu de forma expressiva. A assimetria — exigir mais
evidência para comprar do que para vender — alinhou-se ao objetivo de operar com
segurança em mercados adversos.

**Ressalva registrada:** esse +11,1% foi obtido em configuração ainda sem gestão
de risco estruturada, e o mesmo modelo perdia 19,6% na validação. O número
**não** representa o sistema final; é citado aqui apenas como comparação entre
rotulagens.

### 3.2 Gestão de risco: calibração de stop loss e take profit

**Pergunta:** qual a relação risco/retorno ótima?

Varredura na validação:

| SL / TP | Validação |
|---|---|
| **0,5% / 1,0%** | **−2,0%** (melhor) |
| 1,0% / 2,0% | −6,5% |
| 1,5% / 3,0% | −7,4% |
| 1,0% / 3,0% | −7,8% |
| 1,5% / 6,0% | −7,3% |

**Resultado contraintuitivo:** stops largos pioram o desempenho. Em mercado
picotado, movimentos adversos prosseguem (perda maior por operação) enquanto os
favoráveis revertem antes de alcançar alvos distantes.

### 3.3 Filtro de regime: qual escala temporal?

Uma tentativa inicial usou a EMA de 200 períodos no gráfico de 1 minuto como
filtro de tendência. O resultado foi drástico: as operações vendidas caíram de
154 para 2.

**Causa:** a EMA200 em 1 minuto cobre apenas ~3,3 horas. Um repique local que
gera sinal de venda quase sempre coloca o preço acima dela — as duas condições
eram praticamente excludentes.

**Correção:** média móvel de 24 horas (1440 períodos). Regime deve ser medido na
escala em que o regime existe.

### 3.4 Walk-forward: retreinar compensa?

**Pergunta:** um modelo retreinado a cada trimestre supera um modelo congelado?

Três *folds*, cada um treinando apenas com dados anteriores ao trimestre operado:

| Trimestre | Modelo retreinado | Modelo congelado |
|---|---|---|
| Q4-2025 | +0,8% | +0,4% |
| Q1-2026 | +0,8% | +1,1% |
| Q2-2026 | +0,4% | +0,5% |
| **Total** | **+2,0%** | **+2,0%** |

**Resultado: empate.** O retreino trimestral não compensou seu custo (~13h de GPU
por trimestre). O modelo congelado generalizou bem por aproximadamente 11 meses.

**Consequência prática:** retreino por calendário foi abandonado. O gatilho
correto é a divergência entre desempenho observado e esperado.

### 3.5 Expansão do universo de ativos

**Pergunta:** operar mais ativos aumenta as oportunidades lucrativas?

O modelo opera sobre features adimensionais, o que em tese permite aplicá-lo a
ativos não vistos no treino. Cinco pares foram adicionados (DOGE, LINK, ADA, DOT,
LTC), totalizando 11.

| Configuração | Teste | Holdout virgem |
|---|---|---|
| **6 ativos** | **+2,0%** (68 ops) | **+0,2%** (17 ops) |
| 11 ativos | −0,7% (125 ops) | −1,1% (30 ops) |

**Resultado: refutada.** Mais operações, qualidade inferior — o *win rate* caiu de
41,2% para 38,4%.

**Análise por ativo** (padrão consistente em períodos independentes):

| Ativo | Volatilidade 2h | Resultado |
|---|---|---|
| ADA | 1,08% | Prejuízo em ambos os períodos |
| DOT | 1,11% | Prejuízo em ambos os períodos |
| LINK | 0,96% | Prejuízo em ambos os períodos |
| DOGE | 0,99% | Lucro em ambos |
| LTC | 0,81% | Lucro em ambos |

A correlação com volatilidade motivou o experimento seguinte.

### 3.6 Stops adaptativos por volatilidade

**Hipótese:** ativos mais voláteis sofrem porque o stop fixo de 0,5% é menor que
seu ruído natural — seriam estopados por flutuação aleatória.

Implementação: `SL = k × ATR do próprio ativo`, mantendo relação 2:1.

**Erro de escala detectado durante o experimento:** o multiplicador inicial (1,2)
foi herdado de uma versão anterior que usava ATR em outra escala temporal. O
`atr_pct` do projeto é ATR(14) em escala de 1 minuto (0,064% a 0,121%), de modo
que o stop calculado colidia com o piso configurado. A primeira rodada, na
prática, testou um stop fixo de 0,4% — não o adaptativo. Após medir a escala
real, a faixa correta de `k` mostrou-se entre 6 e 12.

| Configuração | Validação | Teste | Holdout |
|---|---|---|---|
| **Stop fixo 0,5%** | **−2,0%** | **+2,0%** | **+0,2%** |
| 6×ATR | −3,7% | −2,7% | −1,2% |
| 8×ATR | −3,8% | +0,9% | −1,2% |
| 10×ATR | −3,1% | — | — |
| 12×ATR | −3,0% | — | — |

**Resultado: refutada.** O stop fixo venceu em todos os valores de `k` e em todos
os splits.

**Evidência que descarta a hipótese do ruído:** no holdout, exatamente as mesmas
18 operações foram estopadas com stop de 0,5%, 0,72% e 0,85% — e o stop mais
largo apenas aumentou o prejuízo. Se o problema fosse ruído estopando cedo,
alargar o stop reduziria os acionamentos. Não reduziu: quando o preço se move
contra a posição nesses ativos, ele percorre qualquer stop razoável.

### 3.7 Calibração: o modelo sabe quando sabe?

**Pergunta:** a confiança declarada pelo modelo corresponde à sua taxa de acerto?

Métrica: `edge = FAVOR / (FAVOR + CONTRA)`, medida diretamente sobre os preços,
sem stops, alavancagem ou taxas. Valor 0,50 equivale a decisão aleatória.

| Faixa de confiança | Edge no teste | Edge na validação |
|---|---|---|
| 0,52–0,57 | 0,530 | 0,498 |
| 0,57–0,62 | 0,538 | 0,498 |
| 0,62–0,67 | 0,515 | 0,456 |
| acima de 0,72 | 0,594 | 0,402 |
| **Geral** | **0,529** | **0,493** |

**Resultado:** o *edge* não cresce com a confiança declarada — em alguns períodos
chega a decrescer. O modelo **não é bem calibrado** nas faixas superiores.

Esse achado tem consequência prática direta: a alavancagem, que era proporcional
à confiança, estava atribuindo maior exposição a faixas de menor confiabilidade.

### 3.8 Ablação: qual a contribuição real da rede neural?

O achado anterior levantou uma dúvida legítima: se o *edge* medido é próximo do
aleatório, a rede neural realmente contribui?

**Método:** executar a estratégia de forma idêntica, com o filtro neural
neutralizado (todo candidato do componente determinístico é aceito).

| Período | Com rede neural | Sem rede neural | Contribuição |
|---|---|---|---|
| Validação H2-2025 | −2,0% (55 ops) | −4,1% (232 ops) | **+2,1 p.p.** |
| Teste jan–jul/2026 | **+2,0%** (68 ops) | **−6,0%** (246 ops) | **+8,0 p.p.** |
| Holdout virgem | +0,2% (17 ops) | −0,6% (57 ops) | **+0,8 p.p.** |

**Resultado: confirmada.** Sem a rede neural o sistema perde dinheiro em todos os
períodos. Ela descarta cerca de 75% dos sinais e eleva o *win rate* de 36,6% para
41,2%.

**Reconciliação com o experimento anterior — lição metodológica central:**

A medição de *edge* avaliava a rede prevendo a direção em janelas aleatórias do
mercado. Não é assim que ela opera. Ela nunca decide isoladamente: **seleciona
entre candidatos previamente filtrados pelos indicadores técnicos**.

| Capacidade avaliada | Desempenho |
|---|---|
| Prever a direção do mercado a partir do zero | Próximo do aleatório |
| **Discriminar entre sinais técnicos já disparados** | **+8,0 p.p. de contribuição** |

São tarefas distintas. Uma métrica agregada, medida fora do contexto de uso,
quase levou ao descarte do componente mais valioso do sistema. **A ablação — medir
o sistema com e sem o componente, mantendo todo o resto constante — é a régua
correta.**

### 3.9 Curvas de alavancagem

**Pergunta:** como a exposição deve variar com a confiança?

Cinco curvas comparadas na validação:

| Curva | Comportamento | Validação |
|---|---|---|
| `pico` | Concentra na faixa de melhor *edge* | **+0,5%** |
| `flat1` | **Sem alavancagem** | −0,8% |
| `edge` | Realinhada ao *edge* medido | −1,1% |
| `flat2` | Exposição uniforme de 2x | −1,7% |
| `v59` | Histórica (proporcional à confiança) | −2,0% |

**Observação relevante:** operar **sem alavancagem alguma** superou a curva em
produção em 1,2 p.p. A explicação é econômica — as taxas escalam com o notional.
Com *edge* modesto, alavancar não melhora a expectativa; multiplica o custo de
fricção e a variância.

### 3.10 Alavancagem condicionada ao regime

**Hipótese derivada:** a alavancagem compensa em tendência forte e prejudica em
mercado lateral.

Medida de força: distância do preço à média de 24h, normalizada pela mediana
histórica do próprio ativo.

| Período | Alavancagem por regime | Curva histórica |
|---|---|---|
| Validação (lateral) | **−1,0%** | −2,0% |
| Teste (tendência forte) | +1,2% | **+2,0%** |
| Holdout virgem | **+0,8%** | +0,2% |
| **Soma** | **+1,0%** | +0,2% |

**Resultado: trade-off, não melhoria absoluta.** A hipótese se confirma — menor
alavancagem protege em mercado lateral e sacrifica ganho em tendência forte.

Pelo critério pré-definido (superar em ambos os períodos de confirmação), a curva
histórica foi mantida. Registra-se, porém, que a alternativa apresentou soma
superior, variância significativamente menor e melhor desempenho no holdout
virgem — o período mais confiável. A escolha entre elas é uma decisão de perfil
de risco, não de desempenho absoluto.

---

## 4. Síntese

| Hipótese | Veredito |
|---|---|
| Rotulagem assimétrica (especialista em quedas) | ✅ Confirmada |
| Gestão de risco 2:1 com stops estreitos | ✅ Confirmada |
| Filtro de regime em escala diária | ✅ Confirmada |
| **Rede neural agrega valor** | ✅ **Confirmada (+8,0 p.p.)** |
| Retreino trimestral | ❌ Refutada (empate) |
| Expansão do universo de ativos | ❌ Refutada |
| Stops adaptativos por volatilidade | ❌ Refutada |
| Realinhamento da curva de alavancagem | ⚠️ Trade-off dependente de regime |
| Enriquecimento de features (18 → 26) | ⚠️ Em avaliação |

### Lições metodológicas transferíveis

1. **Meça no contexto de uso.** Métricas agregadas fora do contexto real podem
   inverter completamente a conclusão.
2. **Ablação é a régua definitiva** para atribuir valor a um componente.
3. **Verifique a escala das variáveis** antes de escolher multiplicadores —
   parâmetros herdados de outra configuração podem estar ordens de grandeza fora.
4. **Resultados negativos direcionam o trabalho.** Quatro hipóteses refutadas
   eliminaram caminhos improdutivos e concentraram o esforço no que importa.
5. **O critério de decisão precede o resultado.** Sem isso, qualquer conjunto de
   experimentos produz a conclusão desejada.

---

## 5. Limitações reconhecidas

- **Amostras pequenas em alguns recortes.** O holdout virgem contém 17 operações;
  configurações muito seletivas chegam a 5–8. Conclusões nesses recortes têm
  incerteza estatística elevada.
- **Sobreposição parcial entre splits.** O período de teste (jan–jul/2026) contém
  o holdout (jun–jul/2026). Comparações estritamente independentes exigem separar
  jan–mai de jun–jul.
- **Ausência de validação em execução real.** Todos os resultados provêm de
  simulação. *Slippage*, latência, rejeição de ordens e custos de *funding* não
  estão modelados.
- **Duas features projetadas não são geradas** por incompatibilidade de nomes em
  versão beta de biblioteca — documentado, não corrigido no modelo em produção.
- **Período de avaliação predominantemente baixista.** O desempenho em mercado
  altista sustentado permanece não caracterizado.

---

*Documento vivo. Novos experimentos devem ser registrados aqui seguindo o mesmo
formato: pergunta, método, resultado — inclusive quando o resultado contraria a
hipótese.*
