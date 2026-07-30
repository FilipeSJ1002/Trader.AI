# Comandos do Trader.AI

Referência única de operação do projeto. Todo comando importante mora aqui.

- Rode tudo a partir de `M:\.Projects\Trader.AI` com o venv ativado.
- PowerShell 5.1 **não tem `&&`** — use `;` para encadear.
- O *porquê* de cada coisa e o troubleshooting completo estão no vault
  (`M:\.Projects\Brains\Trader.AI` → `Manual de Operacao`).

---

## 1. Ambiente

Criar o ambiente virtual (só na 1ª vez):
```powershell
python -m venv venv
```

Ativar (sempre que abrir o projeto):
```powershell
.\venv\Scripts\Activate.ps1     # PowerShell
.\venv\Scripts\activate         # CMD
source venv/bin/activate        # Linux/Mac
```

Instalar/atualizar bibliotecas:
```powershell
pip install -r requirements.txt
```

---

## 2. Treino da rede neural (GPU — compartilhada com o uso do PC)

Ligar ou retomar do último checkpoint (nunca perde progresso):
```powershell
python v5_train.py --data data_v6 --model-out v6_model.pth --log v6_training.log --label V6.0 --resume --batch 192
```

Ligar em segundo plano (treino + paper trading, janelas minimizadas):
```powershell
.\religar_trader.cmd
```

Acompanhar ao vivo (uma linha por época):
```powershell
Get-Content v6_training.log -Wait -Tail 30
```

Pausar (congela e **libera a GPU inteira**, sem perder progresso):
```powershell
.\treino_pausar.cmd
```

Retomar depois da pausa:
```powershell
.\treino_retomar.cmd
```

Parar de vez (o checkpoint já está salvo em disco):
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*v5_train.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

A GPU está ocupada? (0 MiB = nenhum treino rodando):
```powershell
nvidia-smi
```

Treinar do zero — apaga o checkpoint antes (**cuidado: ~12 h de GPU**):
```powershell
Remove-Item v6_model.pth, v6_model.pth.meta.json
```

> `--resume` restaura o `best_val_loss` do `.meta.json` mas **zera a paciência**:
> religar um treino já convergido faz ele rodar mais 15 épocas antes do early
> stopping. Não é "encerra em poucos minutos".

---

## 3. Paper trading (dinheiro simulado, CPU, ciclo de 15 min)

Ligar:
```powershell
python v5_live.py
```

Acompanhar ao vivo:
```powershell
Get-Content v5_live.log -Wait -Tail 20
```

Parar:
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*v5_live.py*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

> Os logs `v6_training.log` e `v5_live.log` ficam na **raiz** do projeto.
> As cópias dentro de `relatorios/` estão congeladas e enganam.

---

## 4. Execução real — Binance Futures Testnet (no PC)

Só inspecionar a conta (não envia nada):
```powershell
python v6_executor.py --status
```

Um ciclo em dry-run (mostra o que faria):
```powershell
python v6_executor.py --once
```

Um ciclo **enviando ordens** (testnet, dinheiro fictício):
```powershell
python v6_executor.py --once --armar
```

Loop contínuo (a cada 15 min):
```powershell
python v6_executor.py --armar
```

Botão de emergência:
```powershell
python v6_executor.py --fechar BTCUSDT
python v6_executor.py --fechar-tudo
```

Log do executor:
```powershell
Get-Content relatorios\v6_executor.log -Wait -Tail 30
```

Como obter as chaves da testnet de futuros:
```powershell
python v6_executor.py --ajuda-chaves
```

---

## 5. Servidor AWS — executor 24/7 na testnet (desde 30/07/2026)

EC2 `ap-northeast-1` (Tóquio) · Ubuntu 24.04 · 911 MB RAM + 2 GB swap · disco 15 GB.
Pasta `~/trader-ai` (o V4 antigo, parado, fica em `~/Trader.AI`).
Serviço systemd `trader-executor`, `Restart=always`, **sempre testnet**.

Conectar (do PC):
```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" ubuntu@54.249.163.86
```

Ver o bot ao vivo (no servidor):
```bash
sudo journalctl -u trader-executor -f
```

Só os eventos que importam nas últimas 24 h — **nada = nenhum sinal, tudo normal**:
```bash
sudo journalctl -u trader-executor --since "24 hours ago" | grep -E "ABRIR|ENVIADO|ERRO"
```

Estado do serviço:
```bash
sudo systemctl status trader-executor --no-pager
```

Parar / iniciar / reiniciar:
```bash
sudo systemctl stop trader-executor
sudo systemctl start trader-executor
sudo systemctl restart trader-executor
```

Consulta pontual da conta (sem esperar o ciclo):
```bash
cd ~/trader-ai && ./venv/bin/python v6_executor.py --status
```

**Emergência — fechar todas as posições:**
```bash
sudo systemctl stop trader-executor
cd ~/trader-ai && ./venv/bin/python v6_executor.py --fechar-tudo
```

Atualizar o código depois de mexer no PC (rodar **no PC**, na raiz do projeto):
```powershell
scp -i "$env:USERPROFILE\.ssh\id_ed25519" v6_executor.py v6_ciclo.py v5_model.py v5_data_prep.py v5_backtest.py ubuntu@54.249.163.86:~/trader-ai/
```
depois, no servidor:
```bash
sudo systemctl restart trader-executor
```

Religar o bot V4 antigo (está parado; o venv dele foi apagado em 30/07):
```bash
cd ~/Trader.AI && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
tmux new -s traderv4 -d '/home/ubuntu/Trader.AI/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000'
```

Instalação do zero num servidor novo: ver `deploy/README_DEPLOY.md`.

---

## 6. Backtest e validação

Teste (2026), configuração oficial:
```powershell
python v5_backtest.py --model v5_model_b.pth
```

Validação (H2-2025) — **é aqui que se ajustam parâmetros, nunca no teste**:
```powershell
python v5_backtest.py --model v5_model_b.pth --val
```

Modelo V6 (26 features) — precisa do featset casado:
```powershell
python v5_backtest.py --model v6_model.pth --featset v6 --val
```

Ablação (quanto a rede neural realmente agrega):
```powershell
python v5_backtest.py --model v5_model_b.pth --ablacao sem_nn
```

Calibração do `dir_conf`:
```powershell
python v6_calibracao.py --model v5_model_b.pth --data data_v5a --y-dir data_v5b --split val
```

Diário operação a operação e recortes de período:
```powershell
python v5_backtest.py --model v5_model_b.pth --realop --from 2026-04-01 --to 2026-05-05
```

---

## 7. Dados

Atualizar os parquets até hoje:
```powershell
python v6_refresh_dados.py
```

Gerar os datasets de treino (V6, 26 features):
```powershell
python v6_data_prep.py
```

---

## 8. Diagnóstico do projeto

Varredura completa (sintaxe, imports, arquivos, dependências, linter, processos):
```powershell
python v6_diagnostico.py --salvar
```

Erros de tipo iguais aos do VSCode (regras silenciadas em `pyrightconfig.json`):
```powershell
npx -y pyright@latest
```

---

## 9. Git

Somente o Filipe executa commits e pushes neste projeto.

```powershell
git status
git add . ; git commit -m "mensagem" ; git push origin trader-ai-v6
```
