#!/bin/bash

python3 train.py --infer sample/01.jpg --output output/vae-01.jpg
python3 train.py --infer sample/02.jpg --output output/vae-02.jpg
python3 train.py --infer sample/03.jpg --output output/vae-03.jpg
python3 train.py --infer sample/04.jpg --output output/vae-04.jpg
