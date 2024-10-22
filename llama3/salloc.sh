#!/bin/bash
salloc --time 4:00:00 --nodes 1 --ntasks 4 --gpus 2 --mem 100g --qos dw87 --partition dw
