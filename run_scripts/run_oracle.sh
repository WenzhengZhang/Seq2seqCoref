DATA_DIR="asp_data"
#SAVE_DIR="predicts"
#SAVE_DIR="long_t5_large_predicts"
#SAVE_DIR="t5_large_predicts"
SAVE_DIR="t5_base_predicts"
#MODEL_NAME="t5-large"
MODEL_NAME="t5-base"
#MODEL_NAME="google/long-t5-tglobal-large"
lr=5e-4
#lr=5e-5
#lr=3e-4
epochs=100
gpus="0,1,2,3,4,5,6,7"
#gpus="6,7"
warmup=0.1
#model="model"
#model="long_t5_large_model"
model="t5_base_model"
#model="t5_large_model"
#logs="logs"
logs="t5_base_logs"
#logs="t5_large_logs"
#logs="long_t5_large_logs"
#train_len=1152
#train_len=2048
train_len=2048
#train_len=1536
#train_len_out=1792
#train_len_out=3072
train_len_out=4096
#eval_len=2048
eval_len=4096
#eval_len=1536
#eval_len=4096
#eval_len_out=4096
eval_len_out=4096
num_beams=1
train_bsz=1
#train_bsz=3
eval_bsz=4
#eval_bsz=4
log_step=100
eval_step=6000
#eval_step=12000
#eval_step=500
start_eval_epoch=80
weight_decay=0.01
n_gpu=8
#adam_epsilon=1e-8
#--adam_epsilon $adam_epsilon \
ds_config_dir="config"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#n_gpu=4
#export CUDA_VISIBLE_DEVICES=6,7

echo "training generative coreference model with trainer ... "

#python main_trainer.py \
#deepspeed main_trainer.py \
#deepspeed --exclude localhost:0 main_trainer.py
#deepspeed main_trainer.py \
torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 main_oracle.py \
    --output_dir $model/backup_disjoint/disjoint  \
    --model_name_or_path $model/backup_disjoint/disjoint \
    --do_train  \
    --save_strategy no  \
    --save_total_limit 2 \
    --load_best_model_at_end False\
    --metric_for_best_model average_f1 \
    --evaluation_strategy steps \
    --logging_steps $log_step \
    --eval_steps $eval_step \
    --data_dir $DATA_DIR/disjoint \
    --save_dir $SAVE_DIR/oracle \
    --per_device_train_batch_size $train_bsz  \
    --per_device_eval_batch_size $eval_bsz \
    --learning_rate $lr \
    --num_train_epochs $epochs  \
    --logging_dir $logs/oracle \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --dataloader_num_workers 8 \
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
    --oracle_mentions_dir oracle_mentions
#    --optim adamw_apex_fused
# --sharded_ddp zero_dp_2

#    --deepspeed $ds_config_dir/ds_config_T5_large.json \

