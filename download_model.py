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
import os
import json
import random_face.functional as F

def main(args):
    model_zoo = json.load(open("random_face/configs/model_zoo.json"))
    engines = args.engines.split(',')
    for engine in engines:
        print(f"download engine: {engine}...")
        for f in model_zoo[engine]:
            print(f"download {f['name']}...")
            F.download_file_from_gdrive(f["url"], f["name"], root_dir="random_face/data/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", type=str, default="openvino", help="Type of the inference engine [openvino].")
    args = parser.parse_args()
    main(args)
