DATA_DIR=$1
MODEL_NAME_OR_PATH=$2
MODEL_SAVE_DIR=$3
PREDICT_SAVE_DIR=$4
LOG_DIR=$5
SEQ_TYPE=$6
ACTION_TYPE=$7
lr=$8
epochs=$9
eval_len_out=${10}
min_num_mentions=${11}
eval_step=${12}
save_step=${13}
log_step=${14}
eval_delay=${15}
eval_bsz=${16}

# lr=5e-4 for T5-base/large, lr=5e-5 for T0_3B, lr=1e-5/3e-5 for T0pp
# min_num_mentions=2 for ontonotes, 1 for others
# ontonotes: eval_len_out=4096 PreCo: eval_len_out=2560 LB: eval_len_out=6170
# ontonotes: epochs=100 PreCO: epochs=10  LB: epochs=100
# For OntoNotes: eval_step=800, save_step=800, eval_delay=30000, log_step=100
# For PreCo: eval_step=3200, save_step=15200, eval_delay=30000, log_step=100
# For LB: eval_step=100, save_step=100, eval_delay=1500, log_step=10
# ontonotes and lb: eval_bsz=1  preco: eval_bsz=2

# SEQ_TYPE meaning:
# action: copy_action, short_seq: partial linearization, full_seq: token action
# tagging: copy action as decoder_input,
# input_feed: token action sequence+copy action sequence as decoder_input

# ACTION_TYPE meaning:
# integer: integer cluster identity representation
# non_integer: integer-free cluster identity representation


#MODEL_NAME="bigscience/T0_3B"
#MODEL_NAME="google/flan-t5-xl"
#epochs=100
gpus="0,1,2,3,4,5,6,7"
warmup=0.1
train_len=2048
train_len_out=4096
eval_len=4096
#eval_len_out=4096
num_beams=4
train_bsz=1
#eval_bsz=1
#log_step=100
#eval_step=800
#save_step=800
#eval_delay=30000

weight_decay=0.01
n_gpu=8
ds_config_dir="ds_configs"


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

echo "training generative coreference model with trainer ... "

#python main_trainer.py \
#deepspeed main_trainer.py \
#deepspeed --exclude localhost:0 main_trainer.py
#python main_trainer.py \

#deepspeed --exclude localhost:0 main_trainer.py \
deepspeed main_trainer.py \
    --output_dir $MODEL_SAVE_DIR  \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --do_train False \
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
    --deepspeed $ds_config_dir/ds_stage2.json \
    --gradient_checkpointing True \
    --seq2seq_type $SEQ_TYPE \
    --mark_sentence False \
    --action_type $ACTION_TYPE \
    --align_mode l \
    --min_num_mentions $min_num_mentions



