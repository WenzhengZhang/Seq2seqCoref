from dataclasses import dataclass, field
from typing import Optional, List, Union
from transformers import Seq2SeqTrainingArguments
from trainer import OptimizerNames


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="t5-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

    config_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    decay_rate: Optional[float] = field(
        default=0.6, metadata={"help": "Decay learning rate"}
    )
    low_cpu_mem_usage: Optional[bool] = field(
        default=False, metadata={"help": "low cpu mem usage when load model"}
    )


@dataclass
class DataArguments:
    data_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to data directory"}
    )

    max_train_len: Optional[int] = field(
        default=1536,
        metadata={
            "help": "maximum train source input length"
        },
    )
    max_train_len_out: Optional[int] = field(
        default=2048,
        metadata={
            "help": "maximum train target decoder length"
        },
    )
    max_eval_len: Optional[int] = field(
        default=1536,
        metadata={
            "help": "maximum dev/test source input length"
        },
    )
    max_eval_len_out: Optional[int] = field(
        default=2048,
        metadata={
            "help": "maximum dev/test target decode length"
        },
    )

    data_cache_dir: Optional[str] = field(
        default=None, metadata={
            "help": "Where do you want to store the data downloaded from huggingface"}
    )

    beam_sz: Optional[int] = field(
        default=4, metadata={
            "help": "num beams"
        }
    )

    oracle_mentions_dir: Optional[str] = field(
        default=None, metadata={
            "help": "oracle mentions directory"
        }
    )
    language: Optional[str] = field(
        default='english', metadata={
            "help": "coreference language"
        }
    )
    joint_data_dirs: Optional[str] = field(
        default=None, metadata={"help": "datasets dirs for joint training"}
    )
    joint_max_train_lens: Optional[str] = field(
        default=None, metadata={"help": "max train len for each dataset for "
                                        "joint training"}
    )
    joint_max_eval_lens: Optional[str] = field(
        default=None, metadata={"help": "max eval len for each dataset for "
                                        "joint training"}
    )
    joint_num_samples: Optional[int] = field(
        default=2000, metadata={"help": "num samples to subsample for joint "
                                        "training"}
    )


@dataclass
class CorefTrainingArguments(Seq2SeqTrainingArguments):
    do_train: bool = field(default=True,
                           metadata={"help": "Whether to run training."})
    save_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to save predicts directory"}
    )
    save_predicts: Optional[bool] = field(
        default=True, metadata={"help": "whether to save predictions"}
    )
    mark_sentence: Optional[bool] = field(
        default=False, metadata={"help": "mark sentence end for short target?"}
    )
    align_mode: Optional[str] = field(
        default='l', metadata={"help": "alignment mode: highroad (h) or "
                                       "lowroad (l) "}
    )
    optim: Union[OptimizerNames, str] = field(
        default="adamw_apex_fused",
        metadata={"help": "The optimizer to use."},
    )
    parallelize_model: Optional[bool] = field(
        default=False, metadata={"help": "whether to enable naive model "
                                         "parallel"}
    )
    manual_empty_cache: Optional[bool] = field(
        default=False, metadata={"help": "whether to empty cuda cache manually"}
    )
    is_stage3: Optional[bool] = field(
        default=False, metadata={"help": "use deepspeed stage3 for inference "
                                         "if is stage3"}
    )
    val_after_train: Optional[bool] = field(
        default=False, metadata={"help": "save the checkpoints then do "
                                         "validation after training"}
    )
    allow_singletons: Optional[bool] = field(
        default=False, metadata={
            "help": "whether to allow singletons"
        }
    )
    seq2seq_type: Optional[str] = field(
        default='action', metadata={
            "help": "seq2seq type: action, short_seq, full_seq, tagging, "
                    "input_feed, action_non_int"
        }
    )
    action_type: Optional[str] = field(
        default='integer', metadata={
            "help": "target action type: integer, non_integer"
        }
    )
    do_oracle: Optional[bool] = field(
        default=False, metadata={
            "help": "do oracle experiments or not. Provide (gold) mentions "
                    "and ask the model to predict coreference predictions"
        }
    )
    add_mention_end: Optional[bool] = field(
        default=False, metadata={
            "help": "add mention end token when using non-integer action format"
        }
    )
    joint_data_names: Optional[str] = field(
        default=None, metadata={"help": "datasets names for joint training"}
    )
    joint_min_num_mentions: Optional[str] = field(
        default=None, metadata={"help": "threshold for num mentions per epoch "
                                        "in joint training for each dataset"}
    )
    min_num_mentions: Optional[int] = field(
        default=2, metadata={"help": "minimum number of mentions per cluster,"
                                     "ontonotes is 2 other datasets is 1 "
                                     "(allow singletons)"}
    )
    joint_train: Optional[bool] = field(
        default=False, metadata={"help": "whether to use joint training"}
    )
