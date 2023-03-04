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

# Linux/macOS installation script
# If `conda` is not already available on the system (e.g. through Anaconda), Miniconda will be automatically downloaded and installed

# Exit on first error
set -e

# Killable by SIGINT
trap "exit" INT

export PATH=~/miniconda3/condabin:$PATH

root_dir=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)")
curr_dir=$PWD
env_name=$(sed "s/env_name: //" $root_dir/bin/config.yml)
platform=$([ $(uname) == "Linux" ] && echo "linux" || echo "macos")

if ! which conda >/dev/null; then
  echo "Installing conda..."
  if [ $platform == "linux" ]; then
    # Linux
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b
    rm Miniconda3-latest-Linux-x86_64.sh
    echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc
    source ~/.bashrc
  else
    # macOS
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    bash Miniconda3-latest-MacOSX-x86_64.sh -b
    rm Miniconda3-latest-MacOSX-x86_64.sh
    if [ ! -f ~/.bash_profile ]; then
      # Create .bash_profile if not found
      touch ~/.bash_profile
    fi
    echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bash_profile
    source ~/.bash_profile
  fi
fi

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

export PIP_SRC=$root_dir
if conda env list | grep $env_name >/dev/null; then
  echo "Updating virtual environment..."
  conda env update -n $env_name -f $root_dir/conda/environment-$platform.yml
else
  echo "Installing virtual environment..."
  conda env create -n $env_name -f $root_dir/conda/environment-$platform.yml --force
fi

echo "Installing actorch..."
cd $root_dir
conda activate $env_name
pip install -e .[all]
if [ -d ".git" ]; then
  echo "Installing git commit hook..."
  pre-commit install -f
fi
conda deactivate
cd $curr_dir

echo "Done!"
