import re
import pandas as pd
import json
import sys
sys.path.append('./')
from utils.processor import get_default_processor

def preprocess_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding):
    # preprocess the squall datasets for the model input
    TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024, target_delimiter='|')
    # only keep cell truncation, remove row deletion
    TABLE_PROCESSOR.table_truncate_funcs = TABLE_PROCESSOR.table_truncate_funcs[0:1]

    nts = examples["nt"]
    tbls = examples["tbl"]
    nls = examples["question"]
    sqls = examples["query"]
    json_paths = examples["json_path"]

    num_ex = len(tbls)
    table_contents = {}
    inputs, outputs = [], []
    all_ori_headers, all_nl_headers = [], []

    input_truncated = []

    for i in range(num_ex):
        nl = nls[i]
        tbl = tbls[i]
        sql = sqls[i]

        if tbl=='204_56' and 'c1_year' in sql:
            sql = sql.replace('c1_year', 'c1_number')

        if tbl not in table_contents:
            table_dir = json_paths[i]
            f = open(table_dir)
            data = json.load(f)
            # get nl header
            headers = data["headers"][2:] # skip 'id', 'agg'
            tmp = []
            for header in headers:
                if header == '':
                    tmp.append('unk')
                else:
                    tmp.append(header.replace('\n', ' ').replace(' ','_').strip().lower())
            headers = tmp
            # load table
            columns = {}
            max_rows = 0
            for c in data['contents']:
                for cc in c:
                    columns[cc['col']] = cc['data']
                    if len(cc['data'])>max_rows:
                        max_rows=len(cc['data'])
            # ensure each column has same length
            for col in columns:
                if len(columns[col])<max_rows:
                    columns[col] += ['nan']*(max_rows-len(columns[col]))
                if isinstance(columns[col][0], list):
                    columns[col] = [', '.join([str(x) for x in cell]) for cell in columns[col]]
            df = pd.DataFrame(columns).astype(str)
            for col in columns:
                if len(columns[col])<max_rows:
                    columns[col] += [None]*(max_rows-len(columns[col]))
            df = pd.DataFrame(columns)
            # prepare original and nl headers
            ori_headers = list(df.columns)
            nl_headers = []
            for header in ori_headers:
                match = re.search(r'^c(\d+)', header)
                if match:
                    number_after_c = int(match.group(1))
                    nl_header = re.sub(r'c(\d+)', '{}'.format(headers[number_after_c-1]), header)
                else:
                    nl_header = header
                # nl header may have < which can't be tokenized
                nl_header = nl_header.replace('<','!>')
                nl_headers.append(nl_header)
            # make each col name unique
            df.columns = [f'{j+1}_{h}' for j, h in enumerate(nl_headers)]
            # save the table
            table_contents[tbl] = {'nl_header': list(df.columns), 'ori_header': ori_headers, 'rows': df.values.tolist()}

        table_content = table_contents[tbl]
        all_ori_headers.append(table_content['ori_header'])
        all_nl_headers.append(table_content['nl_header'])

        rows = []
        for row in table_content['rows']:
            new_row = [str(item) for item in row]
            rows.append(new_row)
        table_content_x = {'header': table_content['nl_header'], 'rows': rows}
        answer = examples["answer_text"][i].split('|')
        question = nl

        if examples['split_key'][i] == "train":
            # in training, we employ answer to filter table rows to make LARGE tables fit into memory;
            # otherwise, we cannot utilize answer information
            input_source = TABLE_PROCESSOR.process_input(table_content_x, question, answer).lower()
        else:
            input_source = TABLE_PROCESSOR.process_input(table_content_x, question, []).lower()
        
        n_row = len(table_content['rows'])
        truncated = f'row {n_row}' not in input_source
        input_truncated.append(truncated)

        for j in range(len(table_content['ori_header'])):
            sql = sql.replace(table_content['ori_header'][j], table_content['nl_header'][j])
        output = sql

        input = input_source.replace('<', '!>')
        output = output.replace('<', '!>')

        inputs.append(input)
        outputs.append(output)


    # use t5 tokenizer to convert text to ids        
    model_inputs = {
        "input_ids":[], 
        "attention_mask":[], 
        "labels":[], 
        "input_nt":nts, 
        "tbl":tbls, 
        "nl_headers": all_nl_headers,
        "ori_headers": all_ori_headers, 
        "inputs": inputs,
        "truncated": [int(x) for x in input_truncated]
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
    # ensure squall <-> default
    # squall_tableqa can be plus or default
    datasets = load_dataset("/home/siyue/Projects/SynTableQA/task/squall_plus.py", 
                            plus=True, split_id=1)
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
