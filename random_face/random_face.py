"""
Copyright 2021 by Sergei Belousov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import json
import logging as log
import random_face.functional as F
from random_face.engine_openvino import EngineOpenvino


def get_engine():
    """ Get engine.
    Returns:
        engine (BaseEngine): engine.
    """
    cfg = json.load(open(F.res_path(f"configs/config.json")))
    return EngineOpenvino(cfg)
