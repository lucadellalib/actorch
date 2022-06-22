#!/bin/bash

# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
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
