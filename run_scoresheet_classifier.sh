#!/bin/bash
if [ "$#" -ne 1 ]; then
  echo "Please provide scoresheet file as argument!"
fi
echo $1
mkdir -p tmp
touch tmp/asdf
rm tmp/*
python scoresheet_preprocessor.py $1 > /dev/null
CUDA_VISIBLE_DEVICES="" python scoresheet_classifier.py tmp/*
