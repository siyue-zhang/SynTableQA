import pandas as pd

def preprocess_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding):

    # questions = [question.lower() for question in examples["question"]]
    questions = examples["question"]
    example_tables = [table for table in examples["table"]]
    tables = [
        pd.DataFrame.from_records(example_table["rows"], columns=[x.lower() for x in example_table["header"]])
        for example_table in example_tables
    ]

    answers = examples["answers"]

    model_inputs = tokenizer(
        table=tables, query=questions, max_length=max_source_length, padding=padding, truncation=True
    )

    labels = tokenizer(
        answer=["|".join(answer) for answer in answers],
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["truncated"] = [int(len(input_ids)==max_source_length) for input_ids in model_inputs["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


if __name__=='__main__':
    from datasets import load_dataset
    from transformers import TapexTokenizer
    import sys
    sys.path.append('./')
    # squall_tableqa can be plus or default
    datasets = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/wikisql_robut.py", 
                            split_id=1)
    train_dataset = datasets["train"].select(range(10))
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base")
    train_dataset = train_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer":tokenizer, 
                   "max_source_length": 1024,
                   "max_target_length": 512,
                   "ignore_pad_token_for_loss": True,
                   "padding": False}, 
        batched=True,)
    i = 0
    print(tokenizer.decode(train_dataset['input_ids'][i]))
    print(tokenizer.decode(train_dataset['labels'][i]))


