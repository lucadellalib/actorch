![logo](docs/_static/images/actorch-logo.png)

[![Python version: 3.6 | 3.7 | 3.8 | 3.9 | 3.10](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10-blue)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

Welcome to ACTorch, a deep reinforcement learning framework for fast prototyping based on
[PyTorch](https://pytorch.org).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

For Windows, make sure the latest [Visual C++ runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)
is installed.

### Using Pip

First of all, install [Python](https://www.python.org).

### Using Conda

Clone or download and extract the repository, navigate to `<path-to-repository>/bin` and run the
installation script (`install.sh` for Linux/macOS, `install.bat` for Windows).

### Using Docker (Linux/macOS only)

First of all, install [Docker](https://www.docker.com) and [NVIDIA Container Runtime](https://developer.nvidia.com/nvidia-container-runtime).
Clone or download and extract the repository, navigate to `<path-to-repository>`, open a terminal
and run:

```
docker build -t <desired-image-name> .                  # Build image
docker run -it --runtime=nvidia <desired-image-name>    # Run container from image
```

### From source

First of all, install [Python](https://www.python.org).
Clone or download and extract the repository, navigate to `<path-to-repository>`, open a terminal
and run `pip install .`.

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

**NOTE**: to record videos of the environment, [FFmpeg](https://ffmpeg.org) must be available
on the system.

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
