#!/bin/bash

filename=data/valid_preprocess.tsv
outfile=data/result.tsv

python predict.py --do_bulk --filename $filename --outfile $outfile