@echo off
REM Religa o Trader.AI apos reiniciar o PC (atualizado para a era V6):
REM   1. Live paper trading (retoma capital e posicoes de v5_live_state.json)
REM   2. Treino V6 (retoma do ultimo checkpoint via --resume; se ja concluiu,
REM      o early stopping encerra sozinho em poucos minutos sem estragar nada)
cd /d "%~dp0"

REM Remove pausa antiga (se existia antes do desligamento)
if exist "v5_train.pause" (
    del "v5_train.pause"
    echo Pausa antiga removida.
)

set PYTHONIOENCODING=utf-8
start "" /min "%~dp0venv\Scripts\python.exe" v5_live.py
start "" /min "%~dp0venv\Scripts\python.exe" v5_train.py --data data_v6 --model-out v6_model.pth --log v6_training.log --label V6.0 --resume --batch 192

echo.
echo  ============================================
echo   TRADER.AI RELIGADO!
echo   - Live paper trading: retomando estado salvo
echo   - Treino V6: retomando do checkpoint
echo.
echo   Acompanhar treino:
echo     Get-Content v6_training.log -Wait
echo   Pausar treino: treino_pausar.cmd
echo  ============================================
echo.
pause
