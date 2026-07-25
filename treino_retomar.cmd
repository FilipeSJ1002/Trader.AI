@echo off
REM Retoma o treino da rede neural exatamente de onde parou.
del "%~dp0v5_train.pause" 2>nul
echo.
echo  ============================================
echo   TREINO RETOMADO!
echo   A GPU volta a treinar em ate ~10 segundos.
echo.
echo   Para acompanhar:
echo   Get-Content v5_training_wf1.log -Wait
echo  ============================================
echo.
pause
