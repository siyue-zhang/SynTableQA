import os
import torch
import random
import re
from copy import deepcopy
from typing import List, Dict

from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from tqdm import tqdm
import pandas as pd
import json

# fn_kwargs={"tokenizer": tokenizer}

def preprocess_squall_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding):
    # preprocess the squall datasets for the model input

    nts = examples["nt"]
    tbls = examples["tbl"]
    nls = examples["question"]
    sqls = examples["query"]
    answer_texts = examples["answer_text"]
    db_paths = examples["db_path"]
    json_paths = examples["json_path"]

    num_ex = len(tbls)
    table_contents = {}
    inputs, outputs = [], []
    all_ori_headers, all_nl_headers = [], []

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
                nl_headers.append(nl_header)
            # make each col name unique
            df.columns = [f'{j+1}_{h}' for j, h in enumerate(nl_headers)]
            # save the table
            table_contents[tbl] = {'nl_header': list(df.columns), 'ori_header': ori_headers, 'rows': df.values.tolist()}

        table_content = table_contents[tbl]
        all_ori_headers.append(table_content['ori_header'])
        all_nl_headers.append(table_content['nl_header'])
        serialized_schema = 'tab: w col: ' + ' | '.join(table_content['nl_header'])
        serialized_cell = ''
        for j, row in enumerate(table_content['rows']):
            if j>0:
                serialized_cell += ' '
            if isinstance(row[0], list):
                row = [', '.join([str(x) for x in cell]) for cell in row]
            else:
                row = [str(cell) for cell in row]
            serialized_cell += f'row {j+1} : ' + ' | '.join([cell[:min(len(cell),64)] for cell in row])
        
        input = ' '.join([nl, serialized_schema, serialized_cell])

        for j in range(len(table_content['ori_header'])):
            sql = sql.replace(table_content['ori_header'][j], table_content['nl_header'][j])
        output = sql

        input = input.replace('<', '!>')
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
    datasets = load_dataset("/home/siyue/Projects/SynTableQA/task/squall_plus.py", 'plus')
    train_dataset = datasets["train"]
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataset = train_dataset.map(
        preprocess_squall_function,
        fn_kwargs={"tokenizer":tokenizer, 
                   "max_source_length": 1024,
                   "max_target_length": 512,
                   "ignore_pad_token_for_loss": True,
                   "padding": False}, 
        batched=True,)
