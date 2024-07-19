#!/bin/bash

deepspeed --num_nodes=1 --num_gpus=4 train.py --deepspeed_config=ds_config.json --pipeline-parallel-size 4 --steps=100
