from utils.processor import get_default_processor

def preprocess_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding):
	
    TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024)
    input_sources = []
    output_targets = []
    input_truncated = []
    for i in range(len(examples['question'])): 
        table_content = examples['table'][i]
        answer = examples['answers'][i]
        question = examples['question'][i]

        if examples['split_key'][i] == "train":
            # in training, we employ answer to filter table rows to make LARGE tables fit into memory;
            # otherwise, we cannot utilize answer information
            input_source = TABLE_PROCESSOR.process_input(table_content, question, answer).lower()
        else:
            input_source = TABLE_PROCESSOR.process_input(table_content, question, []).lower()
        input_sources.append(input_source)
        
        n_row = len(examples['table'][i]['rows'])
        truncated = not all([f'row {r+1}' for r in range(n_row)])
        input_truncated.append(truncated)

        output_target = TABLE_PROCESSOR.process_output(answer).lower()
        output_targets.append(output_target)

    model_inputs = tokenizer(
        answer=input_sources, max_length=max_source_length, padding=padding, truncation=True
    )

    labels = tokenizer(
        answer=output_targets,
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
        
    model_inputs["labels"] = labels["input_ids"]
    model_inputs["truncated"] = [int(x) for x in input_truncated]

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


