@echo off

rem ==============================================================================
rem Copyright 2022 Luca Della Libera. All Rights Reserved.
rem ==============================================================================

rem Windows installation script
rem If `conda` is not already available on the system (e.g. through Anaconda), Miniconda will be automatically downloaded and installed

set PATH=%PATH%;%HOMEDRIVE%%HOMEPATH%\miniconda3\condabin\

set root_dirpath=%~dp0..
set curr_dirpath=%CD%
for /f "tokens=2 delims=: " %%i in (%root_dirpath%\bin\config.yml) do set env_name=%%i
set platform=windows

where conda >nul || (
  echo Installing conda...
  bitsadmin /transfer Miniconda3 https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe %~dp0Miniconda3-latest-Windows-x86_64.exe
  %~dp0Miniconda3-latest-Windows-x86_64.exe /S /AddToPath=1
  del %~dp0Miniconda3-latest-Windows-x86_64.exe
)

set PIP_SRC=%root_dirpath%
conda env list | findstr %env_name% >nul && (
  echo Updating virtual environment...
  call conda env update -n %env_name% -f %root_dirpath%\conda\environment-%platform%.yml
) || (
  echo Installing virtual environment...
  call conda env create -n %env_name% -f %root_dirpath%\conda\environment-%platform%.yml --force
)

echo Installing actorch...
cd %root_dirpath%
call conda activate %env_name%
call pip install -e .[all]
if exist .git\ (
  echo Installing git commit hook...
  call pre-commit install -f
)
call conda deactivate
cd %curr_dirpath%

echo Done!

pause
