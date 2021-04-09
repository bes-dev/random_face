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
import numpy as np
import gdown


def get_module_path():
    """ Get module path
    Returns:
        path (str): path to current module.
    """
    file_path = os.path.abspath(__file__)
    module_path = os.path.dirname(file_path)
    return module_path

def res_path(path):
    """ Resource path
    Arguments:
        path (str): related path from module dir to some resources.
    Returns:
        path (str): absolute path to module dir.
    """
    return os.path.join(get_module_path(), path)

def tensor_to_img(t, normalize=True, range=(-1, 1), rgb2bgr=True):
    """ Convert numpy tensor to image image
    Arguments:
        t (np.ndarray): input tensor (3xNxM).
        normalize (bool): normalize input tensor.
        range (tuple[float, float]): min and max value of the input image.
        rgb2bgr (bool): convert color space from RGB to BGR.
    Returns:
        img (np.ndarray): output image (NxMx3)
    """
    if normalize:
        t = np.clip(t, range[0], range[1])
        t = (t - range[0]) / (range[1] - range[0] + 1e-5)
    img = np.clip(t * 255 + 0.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)
    if rgb2bgr:
        img = img[:, :, ::-1]
    return img


def download_file_from_gdrive(url, name, md5=None, root_dir=None):
    """ Download file from Google Drive
    Arguments:
        url (str): link to file.
        name (str): name of file.
        md5 (str): md5 sum of remote file.
        root_dir (str): root dir to store file (by default files stored to the module's dir).
    Returns:
        file_path (str): absolute path to downloaded file.
    """
    print(f"load file from gdrive: {name}...")
    file_path = os.path.join(get_module_path() if root_dir is None else root_dir, name)
    gdown.cached_download(url, file_path, md5=md5)
    return file_path
