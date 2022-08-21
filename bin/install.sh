#!/bin/bash

# ==============================================================================
# Copyright 2022 Luca Della Libera. All Rights Reserved.
# ==============================================================================

# Linux/macOS installation script
# If `conda` is not already available on the system (e.g. through Anaconda), Miniconda will be automatically downloaded and installed

# Exit on first error
set -e

# Killable by SIGINT
trap "exit" INT

export PATH=~/miniconda3/condabin:$PATH

root_dirpath=$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd -P)")
curr_dirpath=$PWD
env_name=$(sed "s/env_name: //" $root_dirpath/bin/config.yml)
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

export PIP_SRC=$root_dirpath
if conda env list | grep $env_name >/dev/null; then
  echo "Updating virtual environment..."
  conda env update -n $env_name -f $root_dirpath/conda/environment-$platform.yml
else
  echo "Installing virtual environment..."
  conda env create -n $env_name -f $root_dirpath/conda/environment-$platform.yml --force
fi

echo "Installing actorch..."
cd $root_dirpath
conda activate $env_name
pip install -e .[all]
if [ -d ".git" ]; then
  echo "Installing git commit hook..."
  pre-commit install -f
fi
conda deactivate
cd $curr_dirpath

echo "Done!"
