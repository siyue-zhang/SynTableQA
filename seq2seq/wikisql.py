import pandas as pd

def preprocess_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding):

    questions = [question.lower() for question in examples["question"]]
    example_tables = [table for table in examples["table"]]
    table_ids = examples["table_id"]
    num_ex = len(questions)

    inputs, outputs = [], []
    serilized_tables = {}

    for i in range(num_ex):
        table_id = table_ids[i]
        table = example_tables[i]
        table['header'] = ['id', 'agg'] + table['header']
        for k in range(len(table['rows'])):
            table['rows'][k] = [str(k+1), str(0)] + table['rows'][k]

        question = questions[i]
        if table_id not in serilized_tables:
            tmp = [f"{i+1}_{h.replace(' ', '_').lower()}" for i, h in enumerate(table['header'])]
            serialized_schema = 'tab: w col: ' + ' | '.join(tmp)
            serialized_cell = ''
            for j, row in enumerate(table['rows']):
                if j>0:
                    serialized_cell += ' '
                if isinstance(row[0], list):
                    row = [', '.join([str(x) for x in cell]) for cell in row]
                else:
                    row = [str(cell) for cell in row]
                serialized_cell += f'row {j+1} : ' + ' | '.join([cell[:min(len(cell),64)] for cell in row])
            serilized_table = ' '.join([serialized_schema, serialized_cell])
            serilized_tables[table_id] = serilized_table

        input = ' '.join([question, serilized_tables[table_id]])
        inputs.append(input)
        outputs.append('unk')
        
    # use t5 tokenizer to convert text to ids        
    model_inputs = {
        "input_ids":[], 
        "attention_mask":[], 
        "labels":[], 
        "inputs": inputs
        }

    for n in range(len(inputs)):
        tokenized_inputs = tokenizer([inputs[n]], max_length=max_source_length, padding=padding, return_tensors="pt", truncation=True)
        model_inputs["input_ids"].append(tokenized_inputs["input_ids"].squeeze())
        model_inputs["attention_mask"].append(tokenized_inputs["attention_mask"].squeeze())
        
        if outputs[n] != '':
            tokenized_outputs = tokenizer([outputs[n]], max_length=max_target_length, padding=padding, return_tensors="pt", truncation=True)
            model_inputs["labels"].append(tokenized_outputs["input_ids"].squeeze())
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        # Pad ids (pad=0) are set to -100, which means ignore for loss calculation
        model_inputs["labels"][model_inputs["labels"][: ,:] == 0 ] = -100
    
    model_inputs = {i:model_inputs[i] for i in model_inputs if model_inputs[i] != []}
    return model_inputs



if __name__=='__main__':
    from datasets import load_dataset
    from transformers import T5Tokenizer
    import sys
    sys.path.append('./')
    datasets = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/wikisql_robut.py", 
                            split_id=1)
    train_dataset = datasets["validation"]
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
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
