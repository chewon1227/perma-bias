#!/bin/bash
source ~/.bashrc
micromamba activate vllm
cd ~/workspace/Career-Pathway

arg1=${1:-"English"}
arg2=${2:-"English"}
arg3=${3:-"us"}

cmd="python adhoc/${arg1}.py --model_name_or_path ${arg2} --country ${arg3}"
echo $cmd
eval $cmd