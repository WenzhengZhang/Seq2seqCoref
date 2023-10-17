DATA_DIR="data"
SAVE_DIR="predicts"
MODEL_NAME="t5-base"
lr=5e-4
epochs=100
gpus="0,1,2,3,4,5,6,7"
warmup=0.1
model="model"
train_len=1536
train_len_out=2048
eval_len=1536
eval_len_out=2048
num_beams=4
train_bsz=8
eval_bsz=8
log_step=100
start_eval_epoch=96


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main_no_trainer.py --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --model_name $MODEL_NAME \
    --model $model \
    --epochs $epochs \
    --gpus $gpus \
    --lr $lr \
    --warmup_proportion $warmup \
    --max_train_len $train_len \
    --max_train_len_out $train_len_out \
    --max_eval_len $eval_len \
    --max_eval_len_out $eval_len_out \
    --num_beams $num_beams \
    --train_bsz $train_bsz \
    --eval_bsz $eval_bsz \
    --logging_steps $log_step \
    --start_eval_epoch $start_eval_epoch 