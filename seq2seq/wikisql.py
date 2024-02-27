import sys
sys.path.append('./')
from copy import deepcopy
from utils.processor import get_default_processor
import pandas as pd

def preprocess_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding):

    TABLE_PROCESSOR = get_default_processor(max_cell_length=50, max_input_length=1024, target_delimiter=', ')

    questions = examples["question"]
    input_truncated = []
    output_targets = []
    inputs, outputs = [], []

    for i in range(len(questions)):
        question = questions[i]
        table_content = examples["table"][i]

        w = pd.DataFrame.from_records(table_content['rows'],columns=table_content['header'])
        w_expand = {}
        for col in w.columns:
            data = w[col].tolist()
            w_expand[col] = data
            if any([d.replace('$','').replace('%','').replace(',','').replace('.','',1).isdigit() for d in data]):
                data_expand = []
                for d in data:
                    if d.replace('$','').replace('%','').replace(',','').replace('.','',1).isdigit():
                        if '.' in d:
                            data_expand.append(str(float(d.replace('$','').replace('%','').replace(',',''))))
                        else:
                            data_expand.append(str(int(d.replace('$','').replace('%','').replace(',',''))))
                    else:
                        data_expand.append("")
                w_expand[f'{col}_number'] = data_expand
        w_expand = pd.DataFrame.from_dict(w_expand)

        table_content = {
            'header': w_expand.columns.tolist(),
            'rows': w_expand.values.tolist()
        }
        examples["table"][i] = table_content

        table_content['header'] = [x.replace('\n', ' ').replace(' ','_').strip().lower() for x in table_content['header']]
        table_content['header'] = ['id', 'agg'] + table_content['header']
        table_content['header'] = [f'{k+1}_{x}' for k, x in enumerate(table_content['header'])]
        new_rows = []
        for j, row in enumerate(table_content['rows']):
            row = [f'{j+1}', '0'] + row
            new_rows.append(row)
        table_content['rows'] = new_rows
        table_content_copy = deepcopy(table_content)

        answer = examples["answers"][i]
        if examples['split_key'][i] == "train":
            # in training, we employ answer to filter table rows to make LARGE tables fit into memory;
            # otherwise, we cannot utilize answer information
            input_source = TABLE_PROCESSOR.process_input(table_content_copy, question, answer).lower()
        else:
            input_source = TABLE_PROCESSOR.process_input(table_content_copy, question, []).lower()
        inputs.append(input_source)
        print(input_source, '\n')

        n_row = len(table_content['rows'])
        truncated = f'row {n_row}' not in input_source
        input_truncated.append(truncated)

        output_target = TABLE_PROCESSOR.process_output(answer).lower()
        output_targets.append(output_target)

        outputs.append('unk')
        
    # use t5 tokenizer to convert text to ids        
    model_inputs = {
        "input_ids":[], 
        "attention_mask":[], 
        "labels":[], 
        "inputs": inputs,
        "truncated": [int(x) for x in input_truncated],
        "output_targets": output_targets
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
    datasets = load_dataset("/home/siyue/Projects/SynTableQA/task/wikisql_robut.py", 
                            split_id=0, ignore_verifications=True,
                            perturbation_type='row',
                            # download_mode='force_redownload'
                            )
    train_dataset = datasets["validation"].select(range(1000))
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
