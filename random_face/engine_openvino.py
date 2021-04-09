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
import numpy as np
from openvino.inference_engine import IECore
from random_face.engine_base import EngineBase
import random_face.functional as F


def check_openvino():
    """ Check that OpenVINO configured correctly.
    """
    ie = IECore()

class EngineOpenvino(EngineBase):
    """ Inference Engine based on OpenVINO framework.
    Recommended to use.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ie = IECore()
        # mapping network
        self.mnet = self.ie.read_network(model=F.res_path(cfg["mnet_xml"]), weights=F.res_path(cfg["mnet_bin"]))
        self.mnet_exec = self.ie.load_network(network=self.mnet, device_name="CPU")
        # synthesis network
        self.snet = self.ie.read_network(model=F.res_path(cfg["snet_xml"]), weights=F.res_path(cfg["snet_bin"]))
        self.snet_exec = self.ie.load_network(network=self.snet, device_name="CPU")
        # init style_mean
        self.style_mean = self.get_style_mean(cfg["n_samples"])

    def get_style_mean(self, n=4096):
        var = np.random.normal(size=(n, self.style_dim)).astype(np.float32)
        shape_default = self.mnet.input_info["var"].input_data.shape[0]
        if self.mnet.input_info["var"].input_data.shape[0] != var.shape[0]:
            self.mnet.reshape({"var": var.shape})
            self.mnet_exec = self.ie.load_network(network=self.mnet, device_name="CPU")
        style = self.mnet_exec.infer(inputs={"var": var})["style"].mean(axis=0, keepdims=True)
        return style

    def var_to_style(self, var):
        assert var.shape[1] == self.style_dim
        if self.mnet.input_info["var"].input_data.shape[0] != var.shape[0]:
            self.mnet.reshape({"var": var.shape})
            self.mnet_exec = self.ie.load_network(network=self.mnet, device_name="CPU")
        style = self.mnet_exec.infer(inputs={"var": var})["style"]
        return style

    def style_to_img(self, style):
        assert style.shape[1] == self.style_dim
        if self.snet.input_info["style"].input_data.shape[0] != style.shape[0]:
            self.snet.reshape({"style": style.shape})
            self.snet_exec = self.ie.load_network(network=self.snet, device_name="CPU")
        img = self.snet_exec.infer(inputs={"style": style})["img"]
        return img[0]
