@echo off
chcp 65001 >nul
set VSLANG=1033
set PYTHONUTF8=1
set "PYTHONPATH=D:\01_Workspace\NeuralShading\.trellis\tasks\08-25-03-neural-baseline-and-candidate\scratch;%PYTHONPATH%"
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
call D:\00_Application\miniconda3\condabin\conda.bat run -n neural-shading python -m pip install slangpy-torch==0.7.0 --no-build-isolation
