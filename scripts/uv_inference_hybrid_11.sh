#!/bin/bash
export PYTHONPATH=.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
GPU_IDX=7 # Specify single GPU index to use

# Configuration variables
MODEL_DIR="huggingface_model_dir"
# ---> change this dir to the one you want to link to

METADATA_FILE="Seed-tts-eval/seedtts_testset/en/meta.lst"
OUTPUT_DIR="output"
APPLY_PRUNING=false
NUM_LAYERS=24
LOAD_JIT=true
LOAD_TRT=true
FP16=true
LOAD_VLLM=false


model="11"

OUTPUT_DIR=${OUTPUT_DIR}_${model}

export CUDA_VISIBLE_DEVICES=$GPU_IDX

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Remove existing output files to avoid conflicts
rm -f "$OUTPUT_DIR"/*

# Run the Python script with the specified arguments`   `
uv run analysis_code/hybrid/inference_lite.py \
    --model-dir "$MODEL_DIR" \
    --metadata-file "$METADATA_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --gpu-idx $GPU_IDX \
    --num-layers $NUM_LAYERS \
    --attn-layers 0 2 4 6 8 10 12 14 16 18 20 22 \
    $( [ "$APPLY_PRUNING" = true ] && echo "--apply-pruning" ) \
    $( [ "$LOAD_JIT" = true ] && echo "--load-jit" ) \
    $( [ "$LOAD_TRT" = true ] && echo "--load-trt" ) \
    $( [ "$LOAD_VLLM" = true ] && echo "--load-vllm" ) \
    $( [ "$FP16" = true ] && echo "--fp16" )
