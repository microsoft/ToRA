set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

NUM_GPUS=8


# ------------------- model -------------------
MODEL_SIZE=$2
if [ "$MODEL_SIZE" = "7b" ] || [ "$MODEL_SIZE" = "13b" ]; then
    LEARNING_RATE=2e-5
    DEEPSPEED=ds_configs/stage3_no_offload_accelerate.conf
    BATCH_SIZE_PER_GPU=16
elif [ "$MODEL_SIZE" = "34b" ]; then
    LEARNING_RATE=1e-5
    DEEPSPEED=ds_configs/stage3_offload_optim_accelerate.conf
    BATCH_SIZE_PER_GPU=8
elif [ "$MODEL_SIZE" = "70b" ]; then
    LEARNING_RATE=1e-5
    DEEPSPEED=ds_configs/stage3_offload_optim_accelerate.conf
    BATCH_SIZE_PER_GPU=4
else
    echo "MODEL_SIZE should be 7b, 13b, 34b, or 70b"
    exit 1
fi

BASE_MODEL=$1
if [ "$BASE_MODEL" = "llama2" ]; then
    MODEL_PATH=/path/to/llama2/${BASE_MODEL}/Llama-2-${MODEL_SIZE}-hf
elif [ "$BASE_MODEL" = "codellama" ]; then
    MODEL_PATH=/path/to/codellama/${BASE_MODEL}/CodeLlama-${MODEL_SIZE}-Python-hf
else
    echo "BASE_MODEL should be llama2 or codellama"
    exit 1
fi


# ------------------- data ------------------
DATA_NAME="tora"
NUM_TRAIN_EPOCHS=3
JOB_NAME=${DATA_NAME}_ep${NUM_TRAIN_EPOCHS}

TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

TRAIN_FILE=../data/${DATA_NAME}/examples.jsonl
OUTPUT_DIR=/path/to/output/${BASE_MODEL}_${MODEL_SIZE}/${JOB_NAME}
mkdir -p $OUTPUT_DIR


accelerate launch \
    --main_process_port 18200 \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file $DEEPSPEED \
    train/finetune.py \
    --model_name_or_path ${MODEL_PATH} \
    --use_slow_tokenizer \
    --gradient_checkpointing \
    --train_file $TRAIN_FILE \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 \
    --use_flash_attn \
    --mask_prompt \
    | tee $OUTPUT_DIR/logs.txt
