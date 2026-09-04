# Subir a sobreposição V9 no servidor

O modelo (`modelos/oraculo_3d.joblib`, 236 KB) vai versionado — não precisa
treinar no servidor.

## 1. Dependências

    cd ~/trader-ai
    ./venv/bin/pip install -r trader-v9/requirements-v9.txt

## 2. Semear o histórico (uma vez, ~1 min)

O servidor não guarda os parquets de preço. Baixa 2.600 dias em barras de 4h —
que é o que as features do oráculo realmente usam:

    cd ~/trader-ai/trader-v9
    rm -f ../data/*_1m.parquet          # descarta a semeadura curta anterior
    PYTHONPATH=. ../venv/bin/python -m app.semear

**Não reduza os 2.600 dias.** Medido em 04/09/2026:

| Janela de treino | Acurácia |
|---|---|
| 257 dias | 50,07% (moeda) |
| 1.000 dias | 52,71% |
| completo | 53,69% |

## 3. Treinar o modelo NESTA máquina

Modelo em pickle não atravessa versões do scikit-learn. Treinar aqui, com os
dados que o passo 2 baixou, elimina o acoplamento de vez (leva ~1 min):

    PYTHONPATH=. ../venv/bin/python -m app.treinar_oraculo

## 4. Testar sem enviar ordem

    PYTHONPATH=. ../venv/bin/python -m app.vivo

Confira no log: a defasagem dos dados deve ficar em minutos, e a linha de
regime deve dizer BULL ou BEAR com a probabilidade.

## 5. Armar

    PYTHONPATH=. ../venv/bin/python -m app.vivo --armar

## 6. Agendar

    sudo cp deploy/trader-v9.service deploy/trader-v9.timer /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now trader-v9.timer
    systemctl list-timers trader-v9.timer

## Parar

    sudo systemctl disable --now trader-v9.timer

Para sair das posições e ficar em caixa, edite `config/v9.toml` ou rode uma vez
com o regime forçado — ou simplesmente deixe o próximo ciclo decidir.

## O que esperar

Medido: ~2,2% ao mês em 3,5 anos, com queda máxima de 52,8%. O erro é de ±60
pontos percentuais e o resultado NÃO se distingue de um controle aleatório.
Comprar e manter os mesmos ativos rendeu mais no mesmo período.

Isto está no ar para ser observado, não porque a medição prometa lucro.
