import logging
import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import nltk  # Here to have a nice missing dependency error message early on
import datasets
from datasets import load_dataset

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    T5ForConditionalGeneration,
    T5ForSequenceClassification,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    AutoTokenizer,
    BartForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Pretrained tokenizer name or path if not the same as model_name. "
                "By default we use BART-large tokenizer for TAPEX-large."
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: str = field(
        default="squall", metadata={"help": "squall or selector"}
    )
    squall_plus: str = field(
        default="default", metadata={"help": "default or plus"}
    )
    task: str = field(
        default="tableqa", metadata={"help": "tableqa, text_to_sql or selector"}
    )
    add_from_train: bool = field(
        default=False, metadata={"help": "add samples from train set for training selector"}
    )
    predict_split: str = field(
        default="test", metadata={"help": "which split to predict"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    postproc_fuzzy_string: bool = field(
        default=True,
        metadata={
            "help": "Replace string value with table cell value by fuzzy match."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )


def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # disable wandb
    if data_args.max_train_samples or data_args.max_eval_samples or data_args.max_predict_samples:
        training_args.report_to = []

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    set_seed(training_args.seed)

    if data_args.task == 'selector':
        task = "./task/selector.py"
        raw_datasets = load_dataset(task, dataset=data_args.dataset_name, add_from_train=data_args.add_from_train, download_mode="force_redownload")
    elif data_args.dataset_name == 'squall':
        task = "./task/squall_plus.py"
        raw_datasets = load_dataset(task, data_args.squall_plus)
    else:
        raise NotImplementedError

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    config.max_length = 1024
    config.early_stopping = False
    padding = "max_length" if data_args.pad_to_max_length else False

    # load main tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )

    if training_args.resume_from_checkpoint != None:
        name = training_args.resume_from_checkpoint
    else:
        name = model_args.model_name_or_path

    if data_args.task.lower() == 'tableqa':
        model = BartForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=name,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.task.lower() == 'text_to_sql':
        model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=name,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    elif data_args.task.lower() == 'selector':
        model = BartForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=name,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        raise NotImplementedError

    if data_args.dataset_name=='squall' and data_args.task.lower()=='tableqa':
        from seq2seq.squall_tableqa import preprocess_function
    elif data_args.dataset_name=='squall' and data_args.task.lower()=='text_to_sql':
        from seq2seq.squall import preprocess_function
    elif data_args.task.lower()=='selector':
        from seq2seq.selector import preprocess_function
    else:
        raise NotImplementedError
    
    fn_kwargs={"tokenizer":tokenizer, 
                "max_source_length": data_args.max_source_length,
                "max_target_length": data_args.max_target_length,
                "ignore_pad_token_for_loss": data_args.ignore_pad_token_for_loss,
                "padding": padding}
        
    if training_args.do_train or data_args.predict_split=='train':
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                fn_kwargs=fn_kwargs,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on train dataset",
                )
    else:
        train_dataset = None
    
    if training_args.do_eval or data_args.predict_split=='dev':
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                fn_kwargs=fn_kwargs,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on validation dataset",
                )
    else:
        eval_dataset = None

    if training_args.do_predict:
        if data_args.predict_split=='train':
            predict_dataset = train_dataset
        elif data_args.predict_split=='dev':
            predict_dataset = eval_dataset
        else:
            predict_dataset = raw_datasets["test"]
            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))
            with training_args.main_process_first(desc="test dataset map pre-processing"):
                predict_dataset = predict_dataset.map(
                    preprocess_function,
                    fn_kwargs=fn_kwargs,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=False,
                    desc="Running tokenizer on predict dataset",
                    )

    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    
    if data_args.task == 'selector':
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )


    if data_args.dataset_name=='squall' and data_args.task.lower()=='tableqa':
        from metric.squall_tableqa import prepare_compute_metrics
    elif data_args.dataset_name=='squall' and data_args.task.lower()=='text_to_sql':
        from metric.squall import prepare_compute_metrics
    elif data_args.task.lower()=='selector':
        from metric.selector import prepare_compute_metrics

    else:
        raise NotImplementedError
    
    if training_args.do_train:            
        compute_metrics = prepare_compute_metrics(
            tokenizer=tokenizer, 
            eval_dataset=eval_dataset, 
            stage=None, 
            fuzzy=data_args.postproc_fuzzy_string)
    else:
        p = '_plus' if data_args.squall_plus == 'plus' else ''
        stage = f'{data_args.dataset_name}{p}_{data_args.task.lower()}_{data_args.predict_split}'
        compute_metrics = prepare_compute_metrics(
            tokenizer=tokenizer, 
            eval_dataset=predict_dataset, 
            stage=stage, 
            fuzzy=data_args.postproc_fuzzy_string)

    if data_args.task == 'selector':
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset= train_dataset,
            eval_dataset= eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )
    else:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset= train_dataset,
            eval_dataset= eval_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_ds_name, eval_ds in eval_dataset.items():
                dataset_metrics = trainer.evaluate(eval_dataset=eval_ds, metric_key_prefix=f"eval_{eval_ds_name}")
                metrics.update(dataset_metrics)
        else:
            metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")
        if data_args.task == 'selector':
            predict_results = trainer.predict(
                predict_dataset,
                metric_key_prefix="predict",
            )   
        else:
            predict_results = trainer.predict(
                predict_dataset,
                metric_key_prefix="predict",
                max_length=data_args.val_max_target_length,
                num_beams=data_args.num_beams,
            )
        metrics = predict_results.metrics
        max_predict_samples = data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)


if __name__ == "__main__":
    main()
