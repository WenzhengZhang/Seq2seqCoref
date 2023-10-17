ROOT_DIR="/common/users/wz283/coref_dev/"
DATA_DIR=$ROOT_DIR"data/ontonotes/"
#SAVE_DIR="t5_3b_predicts"
#MODEL_NAME="t5-3b"
MODEL_NAME="bigscience/T0_3B"
#MODEL_NAME="bigscience/T0pp"
#MODEL_NAME="google/flan-t5-xl"
SAVE_DIR=$ROOT_DIR"predicts/action/t0_11b/ontonotes/"
#SAVE_DIR="flan_t5_xl_predicts"
lr=8e-4
epochs=100
gpus="0,1,2,3,4,5,6,7"
warmup=0.1
#model="t5_3b_model"
model=$ROOT_DIR"model/action/t0_11b/ontonotes/"
#model="flan_t5_xl_model"
#logs="t5_3b_logs"
logs=$ROOT_DIR"logs/action/t0_11b/ontonotes/"
#logs="flan_t5_xl_logs"
train_len=2048
train_len_out=4096
eval_len=4096
eval_len_out=4096
#num_beams=1
num_beams=1
train_bsz=1
#eval_bsz=2
eval_bsz=2
log_step=100
eval_step=800
save_step=800
eval_delay=32000
start_eval_epoch=80
weight_decay=0.01
n_gpu=8
ds_config_dir="ds_configs"
seq_type="action"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_KERNEL_CACHE_PATH="/common/users/wz283/hf_model_cache/"

echo "training generative coreference model with trainer ... "

#python main_trainer.py \
#deepspeed main_trainer.py \
#deepspeed --exclude localhost:0 main_trainer.py
#python main_trainer.py \

#deepspeed main_trainer.py \
torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 main_trainer.py \
    --output_dir $model  \
    --model_name_or_path $MODEL_NAME \
    --do_train  \
    --do_eval True \
    --save_strategy steps  \
    --load_best_model_at_end True \
    --metric_for_best_model average_f1 \
    --evaluation_strategy steps \
    --logging_steps $log_step \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --per_device_train_batch_size $train_bsz  \
    --per_device_eval_batch_size $eval_bsz \
    --learning_rate $lr \
    --num_train_epochs $epochs  \
    --logging_dir $logs \
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
    --save_total_limit 1 \
    --save_steps $save_step \
    --eval_delay $eval_delay \
    --gradient_checkpointing False \
    --seq2seq_type $seq_type \
    --mark_sentence False \
    --allow_singletons False \
    --use_peft True \
    --use_lora True \
    --lora_module_type all_linear \
    --eval_steps $eval_step \
    --use_low_bit True \
    --bits 8 \
    --ddp_find_unused_parameters True
#    --val_after_train True
# --deepspeed $ds_config_dir/ds_stage2.json \



# TODO: I changed use_cache=None from False in model.py, please check if this
#  is correct latter
#    --resume_from_checkpoint $model/checkpoint-9600
#    --bits 8 \
#    --use_low_bit
#    --align_mode l \



#    --save_total_limit 2 \
#    --optim adamw_apex_fused
# --sharded_ddp zero_dp_2



