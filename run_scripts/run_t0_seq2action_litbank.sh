ROOT_DIR="/common/users/wz283/coref_dev/"
DATA_DIR=$ROOT_DIR"data/litbank/"
#SAVE_DIR="t5_3b_predicts"
#MODEL_NAME="t5-3b"
MODEL_NAME="bigscience/T0_3B"
#MODEL_NAME="google/flan-t5-xl"
SAVE_DIR=$ROOT_DIR"predicts/action/t0_3b/litbank/"
#SAVE_DIR="flan_t5_xl_predicts"
lr=5e-5
epochs=100
gpus="0,1,2,3,4,5,6,7"
warmup=0.1
#model="t5_3b_model"
model=$ROOT_DIR"model/action/t0_3b/litbank/"
#model="flan_t5_xl_model"
#logs="t5_3b_logs"
logs=$ROOT_DIR"logs/action/t0_3b/litbank/"
#logs="flan_t5_xl_logs"
train_len=2048
train_len_out=4096
eval_len=4096
eval_len_out=6170
#num_beams=1
num_beams=4
train_bsz=1
#eval_bsz=2
eval_bsz=1
log_step=10
eval_step=100
save_step=100
eval_delay=1500
start_eval_epoch=80
weight_decay=0.01
n_gpu=8
ds_config_dir="ds_configs"
seq_type="action"

#lrs=(7e-5 8e-5 3e-5 5e-5)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "training generative coreference model with trainer ... "
num_splits=10
#python main_trainer.py \
#deepspeed main_trainer.py \
#deepspeed --exclude localhost:0 main_trainer.py
#python main_trainer.py \
for ((split=0; split<$num_splits; split++))
do
#    if [ $split == 1 ]; then
#      resume=$model/${split}/checkpoint-1800
#    elif [ $split == 3 ]; then
#      resume=$model/${split}/checkpoint-2000
#    else
#      resume=False
#    fi
    resume=False
    if [ $split == 0 ] || [ $split == 2 ]; then
      echo "train on split ${split}"
      deepspeed main_trainer.py \
          --output_dir $model/$split  \
          --model_name_or_path $MODEL_NAME \
          --do_train  \
          --save_strategy steps  \
          --load_best_model_at_end True \
          --metric_for_best_model average_f1 \
          --evaluation_strategy steps \
          --logging_steps $log_step \
          --eval_steps $eval_step \
          --data_dir $DATA_DIR/$split \
          --save_dir $SAVE_DIR/$split \
          --per_device_train_batch_size $train_bsz  \
          --per_device_eval_batch_size $eval_bsz \
          --learning_rate $lr \
          --num_train_epochs $epochs  \
          --logging_dir $logs/$split \
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
          --deepspeed $ds_config_dir/ds_stage2.json \
          --gradient_checkpointing True \
          --seq2seq_type $seq_type \
          --mark_sentence False \
          --allow_singletons True \
          --resume_from_checkpoint $resume
    fi


    #    --save_total_limit 2 \
    #    --optim adamw_apex_fused
    # --sharded_ddp zero_dp_2
  done



