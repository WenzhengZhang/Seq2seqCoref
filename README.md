# Seq2seqCoref
Official implementation for our EMNLP 2023 paper [Seq2seq is All You Need For
 Coreference Resolution](https://arxiv.org/pdf/2310.13774.pdf).
 
## Setup

```
conda create -n seq2seq_coref python==3.8
conda activate seq2seq_coref
pip install -r requirements.txt

```

## Download raw data

#### CoNLL-12 (OntoNotes)
- Follow http://conll.cemantix.org/2012/data.html and https://catalog.ldc.upenn.edu/LDC2013T19 to obtain the files {train,dev,test}.english.v4_gold_conll.




#### PreCo
- Download from [PreCo official website](https://preschool-lab.github.io/PreCo/)  or from [Community](https://drive.google.com/file/d/1q0oMt1Ynitsww9GkuhuwNZNq6SjByu-Y/view).

#### LitBank
- Follow https://github.com/dbamman/lrec2020-coref to obtain the 10-fold LitBank raw data.

## Preprocess data

```
python ./preprocess_scripts/preprocess_data.py \
    --dataset_name [ontonotes, preco, litbank] \
    --input_dir [your raw data directory for dataset_name] \
    --output_dir [your processed data directory for dataset_name] \
    --language english \
    --seg_lens 4096,2048 \
    --num_cross_val_splits [10 for litbank, 1 for others]

```

For partial linearization with sentence markers, preprocess by
 `preprocess_data_mark_sentence.py` with the same above script. 




## Training and evaluation

Train/evaluate the model on each dataset
```
bash run_scripts/train.sh \
    [input data directory] \
    [model_name_or_path] \
    [model save directory] \
    [predict save directory] \
    [logging directory] \
    [seq2seq type: (action, short_seq, full_seq, tagging, input_feed)] \
    [action type: (integer, non_integer)] \
    [learning rate: (5e-4, 5e-5, 3e-5, 2e-5)] \
    [num epochs: (100, 10)] \
    [maximum target length: (4096, 2560, 6170)] \
    [minimum num mentions per cluster: (2,1)] \
    [eval every n steps: (800,3200,100)] \
    [save every n steps: (800, 15200, 100)] \
    [log every n steps: (100, 10)] \
    [eval delay: eval after n steps (30000,1500)] \
    [eval batch size: (1,2)]

```
Train/evaluate the joint model on the union of the three datasets by

```
bash run_scripts/train.sh \
    [OntoNotes data directory] \
    [PreCo data directory] \
    [LitBank data directory] \
    [model_name_or_path] \
    [model save directory] \
    [predict save directory] \
    [logging directory] \
    [seq2seq type: (action, short_seq, full_seq, tagging, input_feed)] \
    [action type: (integer, non_integer)] \
    [learning rate: (5e-4, 5e-5, 3e-5, 2e-5)] 

```

- Check `train.sh` and `train_joint.sh` for recommended hyperparameters and meaning of argument flags.
- If you want to train the partial linearization model with sentence marker, set `--mark_sentence True` in `train.sh` and `train_joint.sh`.
- If you want to run evaluation without training, you can disable training by setting `--do_train False` in the  `train.sh` and `train_joint.sh` and provide the trained model checkpoint path for `[model_name_or_path]`. 

## Model checkpoints
We've uploaded model checkpoints to huggingface hub. You can download the following model checkpoints:

- T0-3b copy action full linearization model on OntoNotes [checkpoint](https://huggingface.co/VincentNLP/seq2seq-coref-t0-3b).
- T0-3b token action full linearization model on OntoNotes [checkpoint](https://huggingface.co/VincentNLP/seq2seq-coref-t0-3b-token-action).
- T0-3b partial linearization with sentence marker model on OntoNotes [checkpoint](https://huggingface.co/VincentNLP/seq2seq-coref-t0-3b-partial-linear).
- T0-3b integer-free copy action full linearization model on OntoNotes [checkpoint](https://huggingface.co/VincentNLP/seq2seq-coref-t0-3b-integer-free).
- T0-3b integer-free add mention end copy action full linearization model on OntoNotes [checkpoint](https://huggingface.co/VincentNLP/seq2seq-coref-t0-3b-integer-free-add-mention-end).
- T0-3b copy action full linearization model jointly trained on the union of OntoNotes, PreCo and Litbank datasets [checkpoint](https://huggingface.co/VincentNLP/seq2seq-coref-t0-3b-joint).
- T0pp copy action full linearization model on OntoNotes [checkpoint](https://huggingface.co/VincentNLP/seq2seq-coref-t0-pp).






