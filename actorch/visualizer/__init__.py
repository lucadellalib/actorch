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

"""Visualizer."""

try:
    import matplotlib
    import numpy
    import plotly
    import scipy
    import tensorboard
    import tqdm
    import yaml
except ImportError:
    raise ImportError(
        "Visualizer is not included in the default installation. "
        "Run `pip install actorch[visualizer]` to install it"
    )

from actorch.visualizer.loaders import *
from actorch.visualizer.plotters import *
from actorch.visualizer.visualize import *
