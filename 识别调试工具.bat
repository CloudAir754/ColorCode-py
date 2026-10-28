@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)
echo 请把图片或视频拖到本窗口，然后按回车：
set /p "INPUT_FILE=> "
set "INPUT_FILE=%INPUT_FILE:"=%"
echo.
echo 如果要详细检查视频中的某一帧，请输入帧号；检查整个视频请直接按回车：
set /p "FRAME_NUMBER=> "
if defined FRAME_NUMBER (
  "%PYTHON_EXE%" debug_recognition.py "%INPUT_FILE%" --frame %FRAME_NUMBER% --open-folder
) else (
  "%PYTHON_EXE%" debug_recognition.py "%INPUT_FILE%" --open-folder
)
echo.
pause
