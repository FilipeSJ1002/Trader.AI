# Subir a sobreposição V9 no servidor

O modelo (`modelos/oraculo_3d.joblib`, 236 KB) vai versionado — não precisa
treinar no servidor.

## 1. Dependências

    cd ~/trader-ai
    ./venv/bin/pip install polars scikit-learn joblib

## 2. Semear o histórico (uma vez, ~10 min)

O servidor não guarda os parquets de preço. As features precisam de 200 barras
diárias, então é preciso baixar ~260 dias:

    cd ~/trader-ai/trader-v9
    PYTHONPATH=. ../venv/bin/python -m app.semear

## 3. Testar sem enviar ordem

    PYTHONPATH=. ../venv/bin/python -m app.vivo

Confira no log: a defasagem dos dados deve ficar em minutos, e a linha de
regime deve dizer BULL ou BEAR com a probabilidade.

## 4. Armar

    PYTHONPATH=. ../venv/bin/python -m app.vivo --armar

## 5. Agendar

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
