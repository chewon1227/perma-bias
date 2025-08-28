#!/bin/bash
source ~/.bashrc

# Function to determine which node to use based on GPU requirements
get_node() {
    if [[ $1 == *"gpt"* ]] || [[ $1 == *"o1"* ]]; then
        gpu_util="sbc"
    elif [[ $1 == *"9"* ]]; then
        gpu_util="sbgbig"
    elif [[ $1 == *"32"* ]]; then
        gpu_util="sbgbig4"
    elif [[ $1 == *"14"* ]]; then
        gpu_util="sbgbig2"
    else
        gpu_util="sbg"
    fi
}

# Define model list
models=(
    # "o1"
    # "gpt-4o"
    "Qwen/Qwen2.5-32B-Instruct"
    "Qwen/Qwen2.5-14B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-1.5B-Instruct"
    "Qwen/Qwen2.5-0.5B-Instruct"
)


# Submit jobs for each model and task type
for model in "${models[@]}"; do
    for country in "us" "ir" "pt"; do
        get_node "$model"
        echo $model
        cmd="$gpu_util scripts/poc.sh inference/do_counsel_nemotron $model $country"
        echo $cmd
        eval $cmd 
    done
done
