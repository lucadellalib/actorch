#!/bin/bash

# ==============================================================================
# Copyright 2022 Luca Della Libera.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Linux/macOS uninstallation script
# NOTE: `conda` will not be uninstalled

# Exit on first error
set -e

# Killable by SIGINT
trap "exit" INT

export PATH=~$HOME/miniconda3/condabin:$PATH

root_dirpath=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)")
env_name=$(sed "s/env_name: //" $root_dirpath/bin/config.yml)

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

echo "Uninstalling virtual environment..."
conda env remove -n $env_name

echo "Done!"
