DATA_DIR="asp_data"
SAVE_DIR="t5_large_predicts"
MODEL_NAME="t5-large"
lr=5e-4
epochs=100
gpus="0,1,2,3,4,5,6,7"
warmup=0.1
model="t5_large_model"
logs="t5_large_logs"
train_len=2048
train_len_out=4096
eval_len=4096
eval_len_out=4096
num_beams=1
train_bsz=1
eval_bsz=4
log_step=100
eval_step=200
save_step=200
start_eval_epoch=80
weight_decay=0.01
n_gpu=8
ds_config_dir="config"
eval_delay=36000


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "training generative coreference model with trainer ... "

deepspeed main_trainer.py \
    --output_dir $model/ds_large  \
    --model_name_or_path $MODEL_NAME \
    --do_train  \
    --save_strategy steps  \
    --load_best_model_at_end False \
    --metric_for_best_model average_f1 \
    --evaluation_strategy steps \
    --logging_steps $log_step \
    --eval_steps $eval_step \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR/ds_large \
    --per_device_train_batch_size $train_bsz  \
    --per_device_eval_batch_size $eval_bsz \
    --learning_rate $lr \
    --num_train_epochs $epochs  \
    --logging_dir $logs/ds_large \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 0 \
    --predict_with_generate True \
    --warmup_ratio $warmup \
    --max_train_len $train_len \
    --max_train_len_out $train_len_out \
    --max_eval_len $eval_len \
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
    --gradient_checkpointing True \
    --deepspeed $ds_config_dir/ds_stage2.json



#    --save_total_limit 2 \
#    --optim adamw_apex_fused
# --sharded_ddp zero_dp_2



