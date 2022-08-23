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

rem Windows uninstallation script
rem NOTE: `conda` will not be uninstalled

set PATH=%PATH%;%HOMEDRIVE%%HOMEPATH%\miniconda3\condabin\

set root_dirpath=%~dp0..
for /f "tokens=2 delims=: " %%i in (%root_dirpath%\bin\config.yml) do set env_name=%%i

echo Uninstalling virtual environment...
call conda env remove -n %env_name%

echo Done!

pause
