@echo off
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)
echo 正在启动识别服务。请保持本窗口开启；关闭窗口即停止服务。
echo.
"%PYTHON_EXE%" runServer.py
echo.
echo 服务已经停止。
pause
