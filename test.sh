#!/bin/bash

LOAD="--load tfl.pth"
python3 train.py --infer --image images/sample/sample01.jpg $LOAD
python3 train.py --infer --image images/sample/sample02.jpg $LOAD
python3 train.py --infer --image images/spo/spo01.png $LOAD
python3 train.py --infer --image images/spo/spo05.jpg $LOAD
python3 train.py --infer --image images/spo/spo19.jpg $LOAD
python3 train.py --infer --image images/blue3/20191021-05973.jpg $LOAD
python3 train.py --infer --image images/doll/F3cBwOtbYAAMxSZ.jpg $LOAD
python3 train.py --infer --image images/doll/Gfc9AZubcAAPUBN.jpg $LOAD
