@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
py -3 -m venv .venv >nul 2>&1
if not exist ".venv\Scripts\python.exe" python -m venv .venv
if not exist ".venv\Scripts\python.exe" goto error
:install
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error
".venv\Scripts\python.exe" tests.py
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip freeze > requirements.lock.txt
echo.
echo Instalacion completada. Ejecuta INICIAR.bat.
pause
exit /b 0
:error
echo.
echo La instalacion no ha terminado. Copia el error que aparece arriba.
pause
exit /b 1
