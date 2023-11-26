ONTONOTES_DATA_DIR=$1
PRECO_DATA_DIR=$2
LB_DATA_DIR=$3
DATA_DIRS="$1,$2,$3"
MODEL_NAME_OR_PATH=$4
MODEL_SAVE_DIR=$5
PREDICT_SAVE_DIR=$6
LOG_DIR=$7
SEQ_TYPE=$8
ACTION_TYPE=$9
lr=$10
# lr=5e-5 for T0_3B, lr=2e-5 for T0pp

epochs=100
gpus="0,1,2,3,4,5,6,7"
warmup=0.1
train_lens="2048,2048,2048"
eval_lens="4096,2048,4096"
eval_out_lens="4096,2560,6500"
eval_len_out=6500
joint_min_num_mentions="2,1,1"
joint_names="onto,preco,lb"
num_beams=2
train_bsz=1
eval_bsz=1
log_step=100
eval_step=12800
save_step=12800
eval_delay=10000
weight_decay=0.01
n_gpu=8
ds_config_dir="ds_configs"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "training generative coreference model with trainer ... "


deepspeed main_trainer.py \
    --output_dir $MODEL_SAVE_DIR  \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --do_train  \
    --save_strategy steps  \
    --load_best_model_at_end True \
    --metric_for_best_model average_f1 \
    --evaluation_strategy steps \
    --logging_steps $log_step \
    --eval_steps $eval_step \
    --data_dir $DATA_DIR \
    --save_dir $PREDICT_SAVE_DIR \
    --per_device_train_batch_size $train_bsz  \
    --per_device_eval_batch_size $eval_bsz \
    --learning_rate $lr \
    --num_train_epochs $epochs  \
    --logging_dir $LOG_DIR \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 0 \
    --predict_with_generate True \
    --warmup_ratio $warmup \
    --max_eval_len_out $eval_len_out \
    --generation_num_beams $num_beams \
    --generation_max_length $eval_len_out \
    --weight_decay $weight_decay \
    --save_predicts True \
    --do_predict True \
    --bf16 True \
    --save_total_limit 2 \
    --save_steps $save_step \
    --eval_delay $eval_delay \
    --deepspeed $ds_config_dir/ds_stage2.json \
    --gradient_checkpointing True \
    --seq2seq_type $SEQ_TYPE \
    --action_type $ACTION_TYPE \
    --joint_train True \
    --joint_min_num_mentions $joint_min_num_mentions \
    --joint_data_names $joint_names \
    --joint_data_dirs $DATA_DIRS \
    --joint_max_train_lens $train_lens \
    --joint_max_eval_lens $eval_lens \
    --joint_num_samples 2000 \
    --align_mode l \
    --mark_sentence False




