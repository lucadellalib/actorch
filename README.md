![logo](docs/_static/images/actorch-logo.png)

[![Python version: 3.6 | 3.7 | 3.8 | 3.9 | 3.10](https://img.shields.io/badge/python-3.6%20|%203.7%20|%203.8%20|%203.9%20|%203.10-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://github.com/lucadellalib/bayestorch/blob/main/LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://github.com/PyCQA/isort)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![PyPI version](https://img.shields.io/pypi/v/actorch)
[![](https://pepy.tech/badge/actorch)](https://pypi.org/project/actorch/)

Welcome to `actorch`, a deep reinforcement learning framework for fast prototyping based on
[PyTorch](https://pytorch.org). The following algorithms have been implemented so far:

- [REINFORCE](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf)
- [Advantage Actor-Critic (A2C)](https://arxiv.org/abs/1602.01783)
- [Actor-Critic Kronecker-Factored Trust Region (ACKTR)](https://arxiv.org/abs/1708.05144)
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347)
- [Advantage-Weighted Regression (AWR)](https://arxiv.org/abs/1910.00177)
- [Deep Deterministic Policy Gradient (DDPG)](https://arxiv.org/abs/1509.02971)
- [Twin Delayed Deep Deterministic Policy Gradient (TD3)](https://arxiv.org/abs/1802.09477)

---------------------------------------------------------------------------------------------------------

## 💡 Key features

- Support for [OpenAI Gymnasium](https://gymnasium.farama.org/) environments
- Support for custom observation/action spaces
- Support for custom multimodal input multimodal output models
- Support for recurrent models (e.g. RNNs, LSTMs, GRUs, etc.)
- Support for custom policy/value distributions
- Support for custom preprocessing/postprocessing pipelines
- Support for custom exploration strategies
- Support for [normalizing flows](https://arxiv.org/abs/1906.02771)
- Batched environments (both for training and evaluation)
- Batched trajectory replay
- Batched and distributional value estimation (e.g. batched and distributional [Retrace](https://arxiv.org/abs/1606.02647) and [V-trace](https://arxiv.org/abs/1802.01561))
- Data parallel and distributed data parallel multi-GPU training and evaluation
- Automatic mixed precision training
- Integration with [Ray Tune](https://docs.ray.io/en/releases-1.13.0/tune/index.html) for experiment execution and hyperparameter tuning at any scale
- Effortless experiment definition through Python-based configuration files
- Built-in visualization tool to plot performance metrics
- Modular object-oriented design
- Detailed API documentation

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

For Windows, make sure the latest [Visual C++ runtime](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads)
is installed.

### Using Pip

First of all, install [Python 3.6 or later](https://www.python.org). Open a terminal and run:

```bash
pip install actorch
```

### Using Conda virtual environment

Clone or download and extract the repository, navigate to `<path-to-repository>/bin` and run
the installation script (`install.sh` for Linux/macOS, `install.bat` for Windows).
`actorch` and its dependencies (pinned to a specific version) will be installed in
a [Conda](https://www.anaconda.com/) virtual environment named `actorch-env`.

**NOTE**: you can directly use `actorch-env` and the `actorch` package in the local project
directory for development (see [For development](#for-development)).

### Using Docker (Linux/macOS only)

First of all, install [Docker](https://www.docker.com) and [NVIDIA Container Runtime](https://developer.nvidia.com/nvidia-container-runtime).
Clone or download and extract the repository, navigate to `<path-to-repository>`, open a
terminal and run:

```bash
docker build -t <desired-image-name> .                  # Build image
docker run -it --runtime=nvidia <desired-image-name>    # Run container from image
```

`actorch` and its dependencies (pinned to a specific version) will be installed in the
specified Docker image.

**NOTE**: you can directly use the `actorch` package in the local project directory inside
a Docker container run from the specified Docker image for development (see [For development](#for-development)).

### From source

First of all, install [Python 3.6 or later](https://www.python.org).
Clone or download and extract the repository, navigate to `<path-to-repository>`, open a
terminal and run:

```bash
pip install .
```

### For development

First of all, install [Python 3.6 or later](https://www.python.org) and [Git](https://git-scm.com/).
Clone or download and extract the repository, navigate to `<path-to-repository>`, open a
terminal and run:

```bash
pip install -e .[all]
pre-commit install -f
```

This will install the package in editable mode (any change to the package in the local
project directory will automatically reflect on the environment-wide package installed
in the `site-packages` directory of your environment) along with its development, test
and optional dependencies.
Additionally, it installs a [git commit hook](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks).
Each time you commit, unit tests, static type checkers, code formatters and linters are
run automatically. Run `pre-commit run --all-files` to check that the hook was successfully
installed. For more details, see [`pre-commit`'s documentation](https://pre-commit.com).

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

In this example we will solve the [OpenAI Gymnasium](https://gymnasium.farama.org/) environment
`CartPole-v1` using [REINFORCE](https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf).
Copy the following configuration in a file named `REINFORCE_CartPole-v1.py` (**with the
same indentation**):

```python
import gymnasium as gym
from torch.optim import Adam

from actorch import *


experiment_params = ExperimentParams(
    run_or_experiment=REINFORCE,
    stop={"training_iteration": 50},
    resources_per_trial={"cpu": 1, "gpu": 0},
    checkpoint_freq=10,
    checkpoint_at_end=True,
    log_to_file=True,
    export_formats=["checkpoint", "model"],
    config=REINFORCE.Config(
        train_env_builder=lambda **config: ParallelBatchedEnv(
            lambda **kwargs: gym.make("CartPole-v1", **kwargs),
            config,
            num_workers=2,
        ),
        train_num_episodes_per_iter=5,
        eval_freq=10,
        eval_env_config={"render_mode": None},
        eval_num_episodes_per_iter=10,
        policy_network_model_builder=FCNet,
        policy_network_model_config={
            "torso_fc_configs": [{"out_features": 64, "bias": True}],
        },
        policy_network_optimizer_builder=Adam,
        policy_network_optimizer_config={"lr": 1e-1},
        discount=0.99,
        entropy_coeff=0.001,
        max_grad_l2_norm=0.5,
        seed=0,
        enable_amp=False,
        enable_reproducibility=True,
        log_sys_usage=True,
        suppress_warnings=False,
    ),
)
```

Open a terminal in the directory where you saved the configuration file and run
(if you installed `actorch` in a virtual environment, you first need to activate
it, e.g. `conda activate actorch-env` if you installed `actorch` using [Conda](https://www.anaconda.com/)):

```bash
pip install gymnasium[classic_control]  # Install dependencies for CartPole-v1
actorch run REINFORCE_CartPole-v1.py    # Run experiment
```

**NOTE**: training artifacts (e.g. checkpoints, metrics, etc.) are saved in nested subdirectories.
This might cause issues on Windows, since the maximum path length is 260 characters. In that case,
move the configuration file (or set `local_dir`) to an upper level directory (e.g. `Desktop`),
shorten the configuration file name, and/or shorten the algorithm name
(e.g. `DistributedDataParallelREINFORCE.rename("DDPR")`).

Wait for a few minutes until the training ends. The mean cumulative reward over
the last 100 episodes should exceed 475, which means that the environment was
successfully solved. You can now plot the performance metrics saved in the auto-generated
[TensorBoard](https://www.tensorflow.org/tensorboard) (or CSV) log files using [Plotly](https://plotly.com/)
(or [Matplotlib](https://matplotlib.org/)):

```bash
pip install actorch[vistool]  # Install dependencies for VisTool
cd experiments/REINFORCE_CartPole-v1/<auto-generated-experiment-name>
actorch vistool plotly tensorboard
```

You can find the generated plots in `plots`.

Congratulations, you ran your first experiment!

**HINT**: since a configuration file is a regular Python script, you can use all the
features of the language (e.g. inheritance).

---------------------------------------------------------------------------------------------------------

## 🔗 Useful links

- [Introduction to deep reinforcement learning](https://spinningup.openai.com/en/latest/)

- [Hyperparameter tuning with Ray Tune](https://docs.ray.io/en/releases-1.13.0/tune/tutorials/tune-lifecycle.html)
  and [Optuna integration](https://docs.ray.io/en/releases-1.13.0/tune/examples/optuna_example.html)

- [Logging with Ray Tune](https://docs.ray.io/en/releases-1.13.0/tune/api_docs/logging.html)

- [Monitoring jobs with Ray Dashboard](https://docs.ray.io/en/releases-1.13.0/ray-core/ray-dashboard.html)

- [Setting up a cluster with Ray Cluster](https://docs.ray.io/en/releases-1.13.0/cluster/index.html)

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------
