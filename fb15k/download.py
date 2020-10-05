"""
Downloads the `FB15K-237` dataset. Alternatively, you can use the following commands:
```
wget https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip
unzip FB15K-237.2.zip
```
"""
import io
import os

import requests

from zipfile import ZipFile

FB15K_URL = "https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip"
DIRNAME = "./fb15k"
FB15K_DIRNAME = os.path.join(DIRNAME, "Release")
SPLITS = {"train", "test", "valid"}

def download(url=FB15K_URL, to=DIRNAME):
    r = requests.get(url)
    r.raise_for_status()

    raw = io.BytesIO(r.content)
    with ZipFile(raw) as zfile:
        zfile.extractall(to)

def load_one(split, dirname=DIRNAME):
    if split not in SPLITS:
        raise KeyError(f"Invalid split '{split}'. Valid splits are {', '.join(SPLITS)}.")
    with open(os.path.join(dirname, split + ".txt")) as f:
        triples = [line.split() for line in f]
    return triples

def load(*splits, dirname=FB15K_DIRNAME, download_if_absent=True):
    if download_if_absent and not (os.path.exists(dirname) and os.path.exists(os.path.join(dirname, "train.txt"))):
        download(to=os.path.join(dirname, os.path.pardir))
    datasets = [load_one(split, dirname) for split in splits]
    if len(datasets) == 1:
        return datasets[0]
    return datasets


