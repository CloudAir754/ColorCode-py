@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo [1/3] 正在检查 Python...
python --version || goto :no_python
echo [2/3] 正在创建独立运行环境 .venv...
if not exist ".venv\Scripts\python.exe" python -m venv .venv
echo [3/3] 正在安装程序依赖...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
echo.
echo 安装完成。以后直接双击“启动服务.bat”即可。
pause
exit /b 0

:no_python
echo 未找到 Python。请先安装 64 位 Python 3.11 或 3.12，并勾选 Add Python to PATH。
pause
exit /b 1
