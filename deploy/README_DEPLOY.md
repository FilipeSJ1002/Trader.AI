# Deploy do executor no servidor (24/7, TESTNET)

Objetivo: rodar `v6_executor.py --armar` sem parar, para capturar as primeiras
ordens reais na **testnet de futuros**. Sem dinheiro de verdade envolvido — a
flag `--real` não é usada em lugar nenhum deste deploy.

## ⚠️ Antes de tudo: a região da AWS importa

A Binance **bloqueia requisições vindas de IPs dos EUA** (HTTP 451). Se a
instância estiver em `us-east-1` (o padrão que quase todo mundo usa), o bot
não conecta. Use uma região não-americana — **`sa-east-1` (São Paulo)** é a
escolha natural: mais perto de você e latência menor para a Binance.

Teste antes de instalar qualquer coisa:

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://api.binance.com/api/v3/ping
```

`200` = liberado. `451` ou `403` = região bloqueada, troque de região.

## 1. Instância

| Item | Valor | Porquê |
|---|---|---|
| Tipo | **t3.small** (2 GB RAM) | o `torch` sozinho ocupa ~800 MB; `t3.micro` (1 GB) mata o processo por OOM |
| Disco | 15 GB | torch + pandas ocupam ~3 GB; sobra folga para logs |
| SO | Ubuntu 24.04 LTS | Python 3.12 de fábrica, compatível |
| Região | `sa-east-1` | ver aviso acima |

O executor **não precisa dos parquets** de `data/` — ele busca os candles pela
API pública a cada ciclo. Não copie os datasets (são GBs à toa).

## 2. O que copiar para o servidor

```bash
# Do seu PC (PowerShell), ajuste o IP e o caminho da chave .pem
scp -i chave.pem v6_executor.py v6_ciclo.py v5_model.py v5_data_prep.py v5_backtest.py v5_live.py requirements-server.txt v5_model_b.pth ubuntu@SEU_IP:/home/ubuntu/trader-ai/
```

O `.env` vai **separado e nunca pelo git**:

```bash
scp -i chave.pem .env ubuntu@SEU_IP:/home/ubuntu/trader-ai/.env
```

> No servidor, deixe só as chaves de **futuros testnet** no `.env`
> (`BINANCE_FUTURES_*`). Não suba as chaves de produção da Binance para uma
> máquina que ainda está em fase de teste.

## 3. Instalação (no servidor)

```bash
sudo apt update && sudo apt install -y python3-venv
cd /home/ubuntu/trader-ai
python3 -m venv venv
./venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
./venv/bin/pip install -r requirements-server.txt
```

Teste de fumaça antes de ligar o serviço (não envia nada):

```bash
./venv/bin/python v6_executor.py --status
./venv/bin/python v6_executor.py --once
```

Se o `--status` mostrar o saldo de $5.000 fictícios, a conexão está certa.

## 4. Ligar 24/7

```bash
sudo cp deploy/trader-executor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now trader-executor
```

Acompanhar ao vivo:

```bash
sudo journalctl -u trader-executor -f
```

Parar / reiniciar:

```bash
sudo systemctl stop trader-executor
sudo systemctl restart trader-executor
```

## 5. O que observar na PRIMEIRA ordem

Quando aparecer `ABRIR LONG/SHORT` no log, confira as três linhas seguintes:

1. `[ENVIADO] entrada ... | id=...` — a corretora aceitou a entrada
2. `[ENVIADO] STOP LOSS ... | id=...` — **a mais importante**
3. `[ENVIADO] TAKE PROFIT ... | id=...`

Se alguma delas vier como `[ERRO]`, pare o serviço e feche tudo:

```bash
sudo systemctl stop trader-executor
./venv/bin/python v6_executor.py --fechar-tudo
```

## 6. O que este deploy NÃO é

- **Não é dinheiro real.** Nenhum comando aqui usa `--real`.
- **Não substitui o bot que já está em produção.** Rode em diretório e serviço
  próprios; decidir se o bot antigo continua ou não é decisão do Filipe.
- **Não é o modelo V6.** Ele usa o `v5_model_b.pth`; o modelo V6 foi refutado
  em 28/07/2026 (ver `Experimentos V6` no vault).
