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

def preprocess_squall_function(examples):
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
            df.columns = nl_headers
            # save the table
            table_contents[tbl] = {'nl_header': list(df.columns), 'ori_header': ori_headers, 'rows': df.values.tolist()}
            print(table_contents)
            assert 1==2



    #         columns = {"header": headers, "column": [], "column_type": []}
    #         contents = {}
    #         for c in data["contents"]:
    #             for cc in c:
    #                 # in table 204_475 5th column called "agg"
    #                 match = re.search(r'c(\d+)', cc["col"])
    #                 if match:
    #                     number_after_c = int(match.group(1))
    #                     col_name = re.sub(r'c(\d+)', '{}'.format(headers[number_after_c-1]), cc["col"])
    #                 else:
    #                     col_name = cc["col"]
    #                 contents[col_name] =[]
    #                 columns["column"].append(col_name)
    #                 columns["column_type"].append(cc['type'])
    #                 if model_args.ADD_TABLE_CONTENT_EXAMPLE>0:
    #                     for idx, v in enumerate(cc["data"]):
    #                         if idx == model_args.ADD_TABLE_CONTENT_EXAMPLE:
    #                             break
    #                         if isinstance(v, list):
    #                             cell = ', '.join([str(x) for x in v])
    #                         else:
    #                             cell = str(v)
    #                         truncated_cell = cell[:min(len(cell),128)]
    #                         contents[col_name].append(truncated_cell)

    #         # in table 204_776 8th column only has 1 item but others have 59 items
    #         if model_args.ADD_TABLE_CONTENT_EXAMPLE>0:
    #             content_len = [len(contents[x]) for x in contents]
    #             if len(set(content_len))!=1:
    #                 counter = Counter(content_len)
    #                 most_common = counter.most_common(1)[0][0]
    #                 contents = {x:contents[x] for x in contents if len(contents[x]) == most_common}
    #             table_contents[tbl] = contents
    #         table_columns[tbl] = columns
    #     columns = table_columns[tbl]
    #     # option to add col type
    #     if data_args.add_col_type:
    #         tmp = [ f"{col} ({col_type})" for col, col_type in zip(columns["column"], columns["column_type"])]
    #         nl += ' tab: w col: ' + ' | '.join(tmp)
    #     else:
    #         nl += ' tab: w col: ' + ' | '.join(columns["column"])

    #     if model_args.ADD_TABLE_CONTENT_EXAMPLE>0:
    #         num_row = len(table_contents[tbl][columns["column"][0]])
    #         select_row = min(num_row, model_args.ADD_TABLE_CONTENT_EXAMPLE)
    #         if model_args.RANDOM_TABLE_CONTENT_EXAMPLE:
    #             indices = random.sample(range(num_row), select_row)
    #         else:
    #             indices = range(select_row)
    #         for t in indices:
    #             row_content = [table_contents[tbl][col][t] for col in table_contents[tbl]]
    #             nl += f' row {t+1} : ' + ' | '.join(row_content)

    #     nl = nl.replace('<', '!>')
    #     inputs.append(nl)

    #     output = ''
    #     sql_dict = sqls[i]
    #     span_indices = sql_dict["span_indices"]
    #     converted_values = []
        
    #     for k in range(len(sql_dict["sql_type"])):
    #         v = sql_dict["value"][k]
    #         sql_type = sql_dict["sql_type"][k]

    #         if data_args.add_sql_type:
    #             insert = sql_type.replace('Literal.', '')
    #             converted_values.append(f'<{insert}>')
    #             if sql_type == 'Keyword' and v not in all_sql_keywords:
    #                 all_sql_keywords.append(v)

    #         if sql_type=="Column":
    #             # fix error column name in sql query
    #             if tbl=="204_56" and v=="c1_year":
    #                 v="c1_number"
    #                 print("column c1 in table 204_56 corrected.")
    #             match = re.search(r'c(\d+)', v)
    #             number_after_c = int(match.group(1))
    #             col_name = re.sub(r'c(\d+)', '{}'.format(columns["header"][number_after_c-1]), v)
    #             converted_values.append(col_name)
    #         elif data_args.use_literial_index and 'Literal' in sql_dict["sql_type"][k]:
    #             l = list(set(span_indices[k]))
    #             l.sort()
    #             assert len(l)<=2
    #             if sql_dict["sql_type"][k] == 'Literal.Number':
    #                 if len(l) == 1:
    #                     v = '[ '+ str(l[0]) + ' ]'
    #                 else:
    #                     v = '[ '+ str(l[0]) + ' , ' + str(l[1]) + ' ]'
    #             if sql_dict["sql_type"][k] == 'Literal.String':
    #                 if len(l) == 1:
    #                     v = '\'[ '+ str(l[0]) + ' ]\''
    #                 else:
    #                     v = '\'[ '+ str(l[0]) + ' , ' + str(l[1]) + ' ]\''
    #             converted_values.append(v)
    #         else:
    #             converted_values.append(v)
        
    #     output = ' '.join(converted_values)
    #     output = output.replace('<', '!>')
    #     outputs.append(output)


    # # use t5 tokenizer to convert text to ids        
    # model_inputs = {
    #     "input_ids":[], 
    #     "attention_mask":[], 
    #     "labels":[], 
    #     "input_nt":examples["nt"], 
    #     "tbl":tbls, 
    #     "header":[table_columns[t]["header"] for t in tbls],
    #     "inputs": inputs
    #     }
    
    # for n in range(len(inputs)):
    #     tokenized_inputs = tokenizer([inputs[n]], max_length=data_args.max_source_length, padding=padding, return_tensors="pt", truncation=True)
    #     model_inputs["input_ids"].append(tokenized_inputs["input_ids"].squeeze())
    #     model_inputs["attention_mask"].append(tokenized_inputs["attention_mask"].squeeze())
        
    #     if outputs[n] != '':
    #         tokenized_outputs = tokenizer([outputs[n]], max_length=data_args.max_target_length, padding=padding, return_tensors="pt", truncation=True)
    #         model_inputs["labels"].append(tokenized_outputs["input_ids"].squeeze())
    # # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # # padding in the loss.
    # if padding == "max_length" and data_args.ignore_pad_token_for_loss:
    #     # Pad ids (pad=0) are set to -100, which means ignore for loss calculation
    #     model_inputs["labels"][model_inputs["labels"][: ,:] == 0 ] = -100
    
    # model_inputs = {i:model_inputs[i] for i in model_inputs if model_inputs[i] != []}
    # return model_inputs
    return None


if __name__=='__main__':
    from datasets import load_dataset
    datasets = load_dataset("/home/siyue/Projects/SynTableQA/task/squall_plus.py", 'plus')
    train_dataset = datasets["train"]
    train_dataset = train_dataset.map(preprocess_squall_function, batched=True,)
