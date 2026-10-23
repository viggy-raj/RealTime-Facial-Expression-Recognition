@echo off
setlocal
cd /d "%~dp0"
echo Live scores API. The webcam window is still run.bat.
python -m facial_cues.api.server
endlocal
