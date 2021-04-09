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
import random_face.functional as F


class EngineBase:
    """ Base class for random_face inference engines.
    Arguments:
        cfg [dict]: config.
    """
    def __init__(self, cfg):
        # params
        self.style_dim = cfg["style_dim"]
        # img params
        self.img_normalize = cfg["img_normalize"]
        self.img_range = cfg["img_range"]
        self.img_rgb2bgr = cfg["img_rgb2bgr"]

    def get_random_face(self, postprocess=True, truncate=True, alpha=0.5):
        """Get random face.
        Arguments:
            postprocess (bool): convert output tensor to image.
        Returns:
            img (np.ndarray): output image.
        """
        var = np.random.normal(size=(1, self.style_dim)).astype(np.float32)
        style = self.var_to_style(var)
        if truncate:
            style = self.truncate(style, alpha)
        img = self.style_to_img(style)
        if postprocess:
            img = F.tensor_to_img(
                img,
                self.img_normalize,
                self.img_range,
                self.img_rgb2bgr
            )
        return img

    def truncate(self, style, alpha=0.5):
        """Truncate by apply mean style
        Arguments:
            style (np.ndarray): input style vector.
            alpha (float): alpha
        Returns:
            style (np.ndarray): output style vector.
        """
        return self.style_mean + alpha * (style - self.style_mean)

    def var_to_style(self, var):
        """Apply mapping network to input noise.
        Should be implemented in specific engine.
        """
        raise NotImplemented

    def style_to_img(self, style):
        """Apply synthesis network to input style.
        Should be implemented in specific engine.
        """
        raise NotImplemented
