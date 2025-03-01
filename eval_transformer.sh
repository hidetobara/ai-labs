#!/bin/bash

python3 train_transformer.py --infer --load my_transformer.pth --input sample/01.jpg --output output/tf-01.jpg
python3 train_transformer.py --infer --load my_transformer.pth --input sample/02.jpg --output output/tf-02.jpg
python3 train_transformer.py --infer --load my_transformer.pth --input sample/03.jpg --output output/tf-03.jpg
python3 train_transformer.py --infer --load my_transformer.pth --input sample/04.jpg --output output/tf-04.jpg
python3 train_transformer.py --infer --load my_transformer.pth --input sample/05.jpg --output output/tf-05.jpg
