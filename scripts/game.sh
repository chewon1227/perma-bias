#!/bin/bash

# Hugging face 캐시 설정
export HF_HOME=/scratch2/chewon1227/.cache/huggingface/

# PyTorch 메모리 최적화
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1

# 모듈 초기화
ml purge

# Conda 환경 초기화 및 활성화
source /home/chewon1227/miniconda3/etc/profile.d/conda.sh
conda activate career

# 모듈명, 출력 파일명, reversed 모드를 인자로 받음
MODEL_NAME=${1:-"Qwen/Qwen2.5-14B-Instruct"}
OUTPUT_FILE=${2:-"/home/chewon1227/careerpathway/data/perma/balance_game_results.jsonl"}
REVERSED_MODE=${3:-""}

echo "Starting balance game with model: $MODEL_NAME"
echo "Output file: $OUTPUT_FILE"
echo "Using conda environment: career"
if [ "$REVERSED_MODE" = "reversed" ]; then
    echo "Reversed mode: ON"
else
    echo "Reversed mode: OFF"
fi

# 실행
cd /home/chewon1227/careerpathway

if [ "$REVERSED_MODE" = "reversed" ]; then
    python src/careerpathway/game.py \
      --model "$MODEL_NAME" \
      --output "$OUTPUT_FILE" \
      --reversed
else
    python src/careerpathway/game.py \
      --model "$MODEL_NAME" \
      --output "$OUTPUT_FILE"
fi

echo "Job completed!" 