#!/bin/bash

read -e -p $'Super Sampler ( Enter int from 0 to 2 ) : \n' -i "0" PREP
python preprocessing.py --supersample $PREP
rm -r ../input
python createRecords.py --input ../preprocessed
python standardize.py
rm -r ../preprocessed
mkdir ../inference
mkdir ../convOut