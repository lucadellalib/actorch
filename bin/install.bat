@echo off

rem ==============================================================================
rem Copyright 2022 Luca Della Libera. All Rights Reserved.
rem ==============================================================================

rem Windows installation script
rem If `conda` is not already available on the system (e.g. through Anaconda), Miniconda will be automatically downloaded and installed

set PATH=%PATH%;%HOMEDRIVE%%HOMEPATH%\miniconda3\condabin\

set root_dir=%~dp0..
set current_dir=%CD%
for /f "tokens=2 delims=: " %%i in (%root_dir%\bin\config.yml) do set env_name=%%i
set platform=windows

where conda >nul || (
  echo Installing conda...
  bitsadmin /transfer Miniconda3 https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe %~dp0Miniconda3-latest-Windows-x86_64.exe
  %~dp0Miniconda3-latest-Windows-x86_64.exe /S /AddToPath=1
  del %~dp0Miniconda3-latest-Windows-x86_64.exe
)

set PIP_SRC=%root_dir%\packages
conda env list | findstr %env_name% >nul && (
  echo Updating virtual environment...
  call conda env update -n %env_name% -f %root_dir%\conda\environment-%platform%.yml
) || (
  echo Installing virtual environment...
  call conda env create -n %env_name% -f %root_dir%\conda\environment-%platform%.yml --force
)

echo Installing git commit hook...
call conda activate %env_name%
cd %root_dir%
call pre-commit install -f
cd %current_dir%
call conda deactivate

echo Done!
