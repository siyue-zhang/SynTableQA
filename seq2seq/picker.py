import pandas as pd
import json
import re
import torch


def preprocess_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding):
    # preprocess the squall datasets for the model input
    tbls = examples["tbl"]
    nls = examples["question"]
    labels = examples["label"]
    json_paths = examples["json_path"]
    choices = examples["choices"]

    num_ex = len(tbls)
    table_contents = {}
    tables = []
    concat = []

    for i in range(num_ex):
        tbl = tbls[i]
        concat.append(nls[i]+choices[i])

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
                    tmp.append(header.replace('\n', ' ').strip().replace(' ', '_').lower())
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
                nl_headers.append(nl_header)
            df.columns = nl_headers            
            # save the table
            table_contents[tbl] = df

        tables.append(table_contents[tbl])


    # use tapex tokenizer to convert text to ids        
    model_inputs = tokenizer(
        table=tables, 
        query=concat,
        max_length=max_source_length, 
        padding=padding, truncation=True)
     
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]
    
    model_inputs["labels"] = [int(x) for x in labels]
    return model_inputs


if __name__=='__main__':
    from datasets import load_dataset
    from transformers import TapexTokenizer
    # ensure squall <-> default
    # squall_tableqa can be plus or default
    datasets = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/picker.py", dataset='squall')
    train_dataset = datasets["validation"]
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-tabfact")
    train_dataset = train_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer":tokenizer, 
                   "max_source_length": 1024,
                   "max_target_length": 512,
                   "ignore_pad_token_for_loss": True,
                   "padding": False}, 
        batched=True,)
    print(train_dataset[2])
    print(tokenizer.decode(train_dataset[2]['input_ids']))

    # outputs = model(**encoding)
    # print(outputs.logits[0], outputs.logits[0].size)
    # output_id = int(outputs.logits[0].argmax(dim=0))
    # print(model.config.id2label[output_id])