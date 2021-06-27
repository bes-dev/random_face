# Random Face

A simple python library for fast image generation of people who do not exist.

<p align="center">
  <img src="res/faces.jpeg"/>
</p>

[![Downloads](https://pepy.tech/badge/random-face)](https://pepy.tech/project/random-face)
[![Downloads](https://pepy.tech/badge/random-face/month)](https://pepy.tech/project/random-face)
[![Downloads](https://pepy.tech/badge/random-face/week)](https://pepy.tech/project/random-face)


For more details, please refer to the [paper](https://arxiv.org/abs/2104.04767).

## Requirements

* Linux, Windows, MacOS
* Python 3.8.+
* CPU compatible with OpenVINO.

## Install package

```bash
pip install random_face
```

## Install the latest version

```bash
git clone https://github.com/bes-dev/random_face.git
cd random_face
pip install -r requirements.txt
python download_model.py
pip install .
```

## Demo

```bash
python -m random_face.demo
```

## Example

```python
import cv2
import random_face

engine = random_face.get_engine()
face = engine.get_random_face()
cv2.imshow("face", face)
cv2.waitKey()
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZackPashkin/random_face/blob/master/Generate_faces.ipynb)
[![Open In Gradio](https://raw.githubusercontent.com/gradio-app/gradio/master/gradio/static/img/logo_inline.png)](https://gradio.app/hub/AK391/MobileStyleGAN.pytorch)

## Citation


```
@misc{belousov2021mobilestylegan,
      title={MobileStyleGAN: A Lightweight Convolutional Neural Network for High-Fidelity Image Synthesis},
      author={Sergei Belousov},
      year={2021},
      eprint={2104.04767},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
