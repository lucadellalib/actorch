@echo off

rem ==============================================================================
rem Copyright 2022 Luca Della Libera.
rem
rem Licensed under the Apache License, Version 2.0 (the "License");
rem you may not use this file except in compliance with the License.
rem You may obtain a copy of the License at
rem
rem     https://www.apache.org/licenses/LICENSE-2.0
rem
rem Unless required by applicable law or agreed to in writing, software
rem distributed under the License is distributed on an "AS IS" BASIS,
rem WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
rem See the License for the specific language governing permissions and
rem limitations under the License.
rem ==============================================================================

rem Windows installation script
rem If `conda` is not already available on the system (e.g. through Anaconda), Miniconda will be automatically downloaded and installed

set PATH=%PATH%;%HOMEDRIVE%%HOMEPATH%\miniconda3\condabin\

set root_dir=%~dp0..
set curr_dir=%CD%
for /f "tokens=2 delims=: " %%i in (%root_dir%\bin\config.yml) do set env_name=%%i
set platform=windows

where conda >nul || (
  echo Installing conda...
  bitsadmin /transfer Miniconda3 https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe %~dp0Miniconda3-latest-Windows-x86_64.exe
  %~dp0Miniconda3-latest-Windows-x86_64.exe /S /AddToPath=1
  del %~dp0Miniconda3-latest-Windows-x86_64.exe
)

set PIP_SRC=%root_dir%
conda env list | findstr %env_name% >nul && (
  echo Updating virtual environment...
  call conda env update -n %env_name% -f %root_dir%\conda\environment-%platform%.yml
) || (
  echo Installing virtual environment...
  call conda env create -n %env_name% -f %root_dir%\conda\environment-%platform%.yml --force
)

echo Installing actorch...
cd %root_dir%
call conda activate %env_name%
call pip install -e .[all]
if exist .git\ (
  echo Installing git commit hook...
  call pre-commit install -f
)
call conda deactivate
cd %curr_dir%

echo Done!

pause
