@echo off
setlocal
cd /d "%~dp0"

if /I "%~1"=="train" goto train
if /I "%~1"=="video" goto video
if /I "%~1"=="help" goto help

echo Starting webcam. Press Q in the window to stop.
python -m facial_cues.main --deception
goto end

:video
if "%~2"=="" (
  echo Usage: run.bat video "D:\path\to\clip.mp4"
  exit /b 1
)
echo Running on "%~2"
python -m facial_cues.main --deception --source "%~2"
goto end

:train
echo Extracting cues, then training the deception model.
python -m facial_cues.training.extract_trial_clips --dataset "D:\datasets\avl\RealLifeDeceptionDetection.2016\Real-life_Deception_Detection_2016" --output "data\trial_cues"
if errorlevel 1 goto end
python -m facial_cues.training.train_trial_deception --manifest "data\trial_cues\manifest.csv" --output "models\deception_mlp.pth"
goto end

:help
echo run.bat              webcam
echo run.bat video FILE   a video file
echo run.bat train        rebuild the model from the courtroom dataset

:end
endlocal
