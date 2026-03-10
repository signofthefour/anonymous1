export PYTHONPATH=.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH # Onlly needed if using CUDA toolkit from conda env
GPU_IDX=7 # Specify single GPU index to use

MODEL_DIR=huggingface_model_dir
# ---> change this dir to the one you want to link to

APPLY_PRUNING=false
NUM_LAYERS=24
LOAD_JIT=true
LOAD_TRT=true
FP16=true
LOAD_VLLM=false

export MODEL_DIR=$MODEL_DIR

# Basic usage
uv run single_inference.py \
  --model-dir $MODEL_DIR \
  --ref-text "We asked over twenty different people, and they all said it was his." \
  --ref-audio sample/common_voice_en_10119832.wav \
  --text "Hello, how are you doing today? I hope you have a very nice day with great results." \
  --output result.wav \
  --attn-layers 0 2 4 6 8 10 12 14 16 18 20 22 \
  $( [ "$APPLY_PRUNING" = true ] && echo "--apply-pruning" ) \
  $( [ "$NUM_LAYERS" != 0 ] && echo "--num-layers $NUM_LAYERS" ) \
  $( [ "$LOAD_VLLM" = true ] && echo "--load-vllm" ) \
  $( [ "$FP16" = true ] && echo "--fp16" )