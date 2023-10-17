import logging
import os
import sys
from transformers import HfArgumentParser, set_seed
from transformers import AutoModelForSeq2SeqLM, \
    DataCollatorForSeq2Seq, AutoConfig, AutoTokenizer
from transformers.integrations import TensorBoardCallback
from arguments import DataArguments, ModelArguments, CorefTrainingArguments \
    as TrainingArguments
from data import CorefDataset, JointDataset, SPEAKER_START, SPEAKER_END, \
    MENTION_START, MENTION_END, COPY, CLUSTER_NEW, CLUSTERS
from trainer import CorefTrainer
from data import ConstrainedDataCollator, SPECIAL_IDS, NON_INT_SPECIAL_IDS
from model import ConstrainedT5
# try:
#     from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, \
#         LoraConfig, TaskType, PeftModel, prepare_model_for_int8_training
#     from peft.tuners.lora import LoraLayer
#     # from transformers import BitsAndBytesConfig
# except ImportError:
#     print("please install peft if you need it")
import torch
from utils import find_all_linear_names

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))


# TODO: support action/not-action, long/short, use alignment/constrained search

def get_last_checkpoint(checkpoint_dir):
    if os.path.isdir(checkpoint_dir):
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if os.path.isdir(os.path.join(checkpoint_dir, filename)) and \
                    filename.startswith('checkpoint'):
                max_step = max(max_step,
                               int(filename.replace('checkpoint-', '')))
        if max_step == 0:
            return None
        checkpoint_dir = os.path.join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir
    return None  # first training


def get_accelerate_model(train_args, model_args,
                         model_config,
                         checkpoint_dir):
    # TODO: check and fix this function
    # n_gpus = torch.cuda.device_count()
    # max_memory = f'{train_args.max_memory_MB}MB'
    # max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
    print(f'loading base model {model_args.model_name_or_path}...')
    # compute_dtype = (torch.float16 if train_args.fp16 else (
    #     torch.bfloat16 if train_args.bf16 else torch.float32))
    if train_args.seq2seq_type == 'action' or train_args.seq2seq_type \
            == 'tagging' or train_args.seq2seq_type == 'input_feed':
        special_ids = SPECIAL_IDS if train_args.action_type == "integer" \
            else NON_INT_SPECIAL_IDS
        model = ConstrainedT5.from_pretrained(
            model_args.model_name_or_path,
            config=model_config,
            special_ids=special_ids,
            seq2seq_type=train_args.seq2seq_type,
            action_type=train_args.action_type,
            add_mention_end=train_args.add_mention_end,
            # load_in_4bit=train_args.bits == 4,
            load_in_8bit=True,
            # load_in_8bit=train_args.bits == 8,
            device_map=device_map,
            # max_memory=max_memory,
            # quantization_config=BitsAndBytesConfig(
            #     load_in_4bit=train_args.bits == 4,
            #     load_in_8bit=train_args.bits == 8,
            #     llm_int8_threshold=6.0,
            #     llm_int8_has_fp16_weight=False,
            #     bnb_4bit_compute_dtype=compute_dtype,
            #     bnb_4bit_use_double_quant=train_args.double_quant,
            #     bnb_4bit_quant_type=train_args.quant_type  # {'fp4', 'nf4'}
            # ),
            torch_dtype=(torch.float32 if train_args.fp16 else (
                torch.bfloat16 if train_args.bf16 else torch.float32))
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path, config=model_config,
            # load_in_4bit=train_args.bits == 4,
            load_in_8bit=True,
            # load_in_8bit=train_args.bits == 8,
            device_map=device_map,
            # max_memory=max_memory,
            # quantization_config=BitsAndBytesConfig(
            #     load_in_4bit=train_args.bits == 4,
            #     load_in_8bit=train_args.bits == 8,
            #     llm_int8_threshold=6.0,
            #     llm_int8_has_fp16_weight=False,
            #     bnb_4bit_compute_dtype=compute_dtype,
            #     bnb_4bit_use_double_quant=train_args.double_quant,
            #     bnb_4bit_quant_type=train_args.quant_type  # {'fp4', 'nf4'}
            # ),
            torch_dtype=(torch.float32 if train_args.fp16 else (
                torch.bfloat16 if train_args.bf16 else torch.float32))
        )
    # if compute_dtype == torch.float16 and train_args.bits == 4:
    #     major, minor = torch.cuda.get_device_capability()
    #     if major >= 8:
    #         print('=' * 80)
    #         print(
    #             'Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
    #         print('=' * 80)
    #
    # setattr(model, 'model_parallel', True)
    # setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = (torch.float32 if train_args.fp16 else (
        torch.bfloat16 if train_args.bf16 else torch.float32))

    if train_args.use_low_bit:
        print("set int8 training")
        model = prepare_model_for_int8_training(
            model,
            use_gradient_checkpointing=train_args.gradient_checkpointing)
    if train_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if train_args.use_lora:
        if train_args.lora_module_type == "attention":
            target_modules = ["q", "v"]
        elif train_args.lora_module_type == "all_linear":
            target_modules = find_all_linear_names(train_args.bits,
                                                   model)
            print(f"target modules {target_modules}")
        else:
            raise ValueError("wrong lora module type")
        lora_config = LoraConfig(
            r=train_args.lora_r,
            lora_alpha=train_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(model, checkpoint_dir)
            for name, p in model.named_parameters():
                if 'lora' in name:
                    print(name, p.sum())
        else:
            print(f'adding LoRA modules...')
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

    if train_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(
                make_inputs_require_grad)
    # for name, module in model.named_modules():
    #     if isinstance(module, LoraLayer):
    #         if train_args.bf16:
    #             module = module.to(torch.bfloat16)
    #     if 'norm' in name:
    #         module = module.to(torch.float32)
    #     if 'lm_head' in name or 'embed_tokens' in name:
    #         if hasattr(module, 'weight'):
    #             if train_args.bf16 and module.weight.dtype == torch.float32:
    #                 module = module.to(torch.bfloat16)
    return model


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1,
                                                           0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, fp16 training: %s, bf16 training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
        training_args.bf16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("Data arguments %s", data_args)

    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if (training_args.seq2seq_type == "action" or training_args.seq2seq_type
        == "input_feed") and training_args.action_type == "non_integer":
        num_new_tokens = tokenizer.add_tokens([SPEAKER_START, SPEAKER_END,
                                               MENTION_START, MENTION_END, COPY,
                                               CLUSTER_NEW] +
                                              CLUSTERS)
    else:
        num_new_tokens = tokenizer.add_tokens([SPEAKER_START, SPEAKER_END,
                                               MENTION_START, MENTION_END,
                                               COPY])
    if training_args.seq2seq_type == 'short_seq' and \
            training_args.mark_sentence:
        from preprocess_mark_sentence import SENTENCE_START, SENTENCE_END
        num_new_tokens += tokenizer.add_tokens([SENTENCE_START, SENTENCE_END])
    # we  need to resize model token embeddings
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    if training_args.gradient_checkpointing:
        # use_cache is False for training, True for evaluation
        config.use_cache = False
    if training_args.use_peft:
        if training_args.resume_from_checkpoint is not None and training_args.resume_from_checkpoint != False and training_args.resume_from_checkpoint != "False":
            ckpt_dir = training_args.resume_from_checkpoint if \
                training_args.resume_from_checkpoint != True and \
                training_args.resume_from_checkpoint != "True" else \
                get_last_checkpoint(training_args.output_dir)
        else:
            ckpt_dir = None
        model = get_accelerate_model(training_args, model_args,
                                     config, ckpt_dir)
        training_args.skip_loading_checkpoint_weights = True
    else:
        if training_args.seq2seq_type == 'action' or training_args.seq2seq_type \
                == 'tagging' or training_args.seq2seq_type == 'input_feed':
            special_ids = SPECIAL_IDS if training_args.action_type == "integer" \
                else NON_INT_SPECIAL_IDS
            model = ConstrainedT5.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                special_ids=special_ids,
                seq2seq_type=training_args.seq2seq_type,
                action_type=training_args.action_type,
                add_mention_end=training_args.add_mention_end
            )
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path, config=config)

    if len(model.get_input_embeddings().weight) < len(tokenizer):
        logger.info('resize model input embeddings')
        model.resize_token_embeddings(len(tokenizer))

    if training_args.seq2seq_type == 'action' or training_args.seq2seq_type \
            == 'input_feed':
        collator = ConstrainedDataCollator(tokenizer, model=model)
    else:
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # model.resize_token_embeddings(len(tokenizer))
    data_cls = JointDataset if training_args.joint_train else CorefDataset
    train_set = data_cls(tokenizer, data_args, training_args, 'train')
    dev_set = data_cls(tokenizer, data_args, training_args, 'dev')
    test_set = data_cls(tokenizer, data_args, training_args, 'test')
    # pdb.set_trace()
    tb_callback = TensorBoardCallback()
    if training_args.parallelize_model:
        model.parallelize()
    trainer = CorefTrainer(
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=dev_set,
        data_collator=collator,
        callbacks=[tb_callback]
    )
    if training_args.resume_from_checkpoint is not None and training_args.resume_from_checkpoint != False and training_args.resume_from_checkpoint != "False":
        trainer.train(
            resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()
    if trainer.is_world_process_zero():
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

    if training_args.do_predict:
        test_results = trainer.evaluate(
            test_set,
            max_length=data_args.max_eval_len_out,
            num_beams=training_args.generation_num_beams)
        logger.info(f'test results: {test_results}')
        dev_results = trainer.evaluate(
            dev_set,
            max_length=data_args.max_eval_len_out,
            num_beams=training_args.generation_num_beams)
        logger.info(f'dev results: {dev_results}')


if __name__ == "__main__":
    main()
