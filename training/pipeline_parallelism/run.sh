#!/bin/bash

deepspeed train.py --num_nodes 1 --num_gpus 2 --deepspeed_config=ds_config.json -p 2 --steps=200
