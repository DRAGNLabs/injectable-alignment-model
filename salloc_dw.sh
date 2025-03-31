#!/bin/bash
salloc --time 3:30:00 --nodes 1 --ntasks-per-node $1 --gpus $1 --mem 512g --qos dw87 --partition dw #--exclude
