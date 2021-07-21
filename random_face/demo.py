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
import argparse
import time
import cv2
import numpy as np
from random_face import get_engine
import random_face.functional as F
from blessed import Terminal

def main(args):
    engine = get_engine()
    style_a = engine.truncate(engine.var_to_style(np.random.normal(size=(1, engine.style_dim))), alpha=args.alpha)
    style_b = engine.truncate(engine.var_to_style(np.random.normal(size=(1, engine.style_dim))), alpha=args.alpha)

    if args.save_video is not None:
        img = engine.style_to_img(style_a)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_writer = cv2.VideoWriter(args.save_video, fourcc, 15.0, (img.shape[1], img.shape[2]))

    term = Terminal()
    while True:
        for i in range(args.n_steps):
            weight = (1.0 / args.n_steps) * i
            style = style_a + weight * (style_b - style_a)
            start = time.time()
            img = engine.style_to_img(style)
            stop = time.time()
            print(term.clear() + term.move(0, 0) + f"Processing time: {stop - start} s.")
            print(term.move(1, 0) + "Press 'q' for quit")
            img = F.tensor_to_img(img)
            if args.save_video is not None:
                video_writer.write(img)
            cv2.imshow("random_face_demo", img)
            key = chr(cv2.waitKey(20) & 255)
            if key == 'q':
                break
        if key == 'q':
            break
        style_a = style_b
        style_b = engine.truncate(engine.var_to_style(np.random.normal(size=(1, engine.style_dim))), alpha=args.alpha)

    if args.save_video is not None:
        video_writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5, help="Truncation mode with alpha.")
    parser.add_argument("--n-steps", type=int, default=30, help="Steps of interpolation.")
    parser.add_argument("--save-video", type=str, default=None, help="Path to store video (if 'None' then video will not write).")
    args = parser.parse_args()
    main(args)
