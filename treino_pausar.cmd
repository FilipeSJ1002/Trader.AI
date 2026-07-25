@echo off
REM Pausa o treino da rede neural e LIBERA A GPU INTEIRA.
REM O treino fica congelado (sem perder progresso) ate voce rodar treino_retomar.cmd
echo pausa > "%~dp0v5_train.pause"
echo.
echo  ============================================
echo   TREINO PAUSADO!
echo   A GPU sera liberada em ate ~2 minutos.
echo   Pode usar o PC a vontade.
echo.
echo   Para retomar: treino_retomar.cmd
echo  ============================================
echo.
pause
