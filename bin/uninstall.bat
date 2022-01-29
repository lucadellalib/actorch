@echo off

rem ==============================================================================
rem Copyright 2022 Luca Della Libera. All Rights Reserved.
rem ==============================================================================

rem Windows uninstallation script
rem NOTE: `conda` will not be uninstalled

set PATH=%PATH%;%HOMEDRIVE%%HOMEPATH%\miniconda3\condabin\

set root_dir=%~dp0..
for /f "tokens=2 delims=: " %%i in (%root_dir%\bin\config.yml) do set env_name=%%i

echo Uninstalling virtual environment...
call conda env remove -n %env_name%

echo Done!
