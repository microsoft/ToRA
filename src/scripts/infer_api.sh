set -ex

MODEL_NAME_OR_PATH="gpt-4"
DATA_NAME="math"

SPLIT="train"
PROMPT_TYPE="tora"
NUM_TEST_SAMPLE=-1

python -um infer.inference_api \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir ../outputs/ \
    --data_name $DATA_NAME \
    --split $SPLIT \
    --prompt_type $PROMPT_TYPE \
    --num_test_sample $NUM_TEST_SAMPLE \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \

