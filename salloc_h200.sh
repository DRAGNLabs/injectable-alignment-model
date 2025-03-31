#!/bin/bash
salloc --time 0:30:00 --nodes 1 --ntasks-per-node 2 --gpus=h200:1 --mem 256g
