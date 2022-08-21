# syntax=docker/dockerfile:1

# Parent image must be a Linux distribution with `apt-get` available
ARG PARENT_IMAGE=nvidia/cuda:11.3.1-devel-ubuntu20.04
FROM $PARENT_IMAGE

# Install system dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ffmpeg \
  gedit \
  git \
  sudo \
  unzip \
  vim \
  wget \
  zip \
  && rm -rf /var/lib/apt/lists/*

# Define useful environment variables
ENV HOME /home
ENV PATH $HOME/miniconda3/bin:$PATH

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
  bash Miniconda3-latest-Linux-x86_64.sh -b && \
  rm Miniconda3-latest-Linux-x86_64.sh && \
  echo ". ~/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
  bash -c "source ~/.bashrc" && \
  bash -c "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

# Copy and install project
COPY . $HOME/actorch
RUN conda env update -n base -f $HOME/actorch/conda/environment-linux.yml
RUN pip install -e $HOME/actorch[all]

# If a git repository, install git commit hook
RUN cd $HOME/actorch && bash -c "if [ -d '.git' ]; then pre-commit install -f; fi"

# Set working directory
WORKDIR $HOME
