import pandas as pd
import json
import re
import random
from copy import deepcopy
import sys
sys.path.append('./')
from utils.processor import get_default_processor

def preprocess_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding, input_noise=None):
    # preprocess the squall datasets for the model input
    TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024, target_delimiter='|')

    tbls = examples["tbl"]
    nls = examples["question"]
    sqls = examples["query"]
    answer_texts = examples["answer_text"]
    json_paths = examples["json_path"]

    num_ex = len(tbls)
    table_contents = {}

    input_sources = []
    output_targets = []
    input_truncated = []

    for i in range(num_ex):
        tbl = tbls[i]
        sql = sqls[i]

        if tbl=='204_56' and 'c1_year' in sql:
            sql = sql.replace('c1_year', 'c1_number')

        if tbl not in table_contents:
            # load squall json
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
                    tmp.append(header.replace('\n', ' ').strip())
                    # tmp.append(header.replace('\n', ' ').strip().replace(' ', '_').lower())
            headers = tmp
            # load table
            columns = {}
            max_rows = 0
            for c in data['contents']:
                for cc in c:
                    if cc['col'] in ['id', 'agg']:
                        break
                    columns[cc['col']] = cc['data']
                    if len(cc['data'])>max_rows:
                        max_rows=len(cc['data'])
                    # only keep the original column for tableqa task
                    break
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
        
        df = table_contents[tbl]
        table_content = {'header': df.columns.tolist(), 'rows': df.values.tolist()}
        table_content_copy = deepcopy(table_content)
        answer = answer_texts[i].split('|')
        question = nls[i]

        if input_noise is not None:
            random.seed(input_noise)
            # print('\n---before---\n', table_content_copy)
            random.shuffle(table_content_copy['rows'])
            # print('\n---row shuffle---\n', table_content_copy)

            kk=3
            if len(table_content_copy['header'])>kk:
                fixed_header = table_content_copy['header'][:kk]
                shuffled_header = table_content_copy['header'][kk:]
                random.shuffle(shuffled_header)
                table_content_copy['header'] = fixed_header + shuffled_header
                for row in table_content_copy['rows']:
                    fixed_part = row[:kk]
                    shuffle_part = row[kk:]
                    shuffled_row = fixed_part + [shuffle_part[shuffled_header.index(col)] for col in table_content_copy['header'][kk:]]
                    row[:] = shuffled_row
            # print('\n---col shuffle---\n', table_content_copy)

        if examples['split_key'][i] == "train":
            # in training, we employ answer to filter table rows to make LARGE tables fit into memory;
            # otherwise, we cannot utilize answer information
            input_source = TABLE_PROCESSOR.process_input(table_content_copy, question, answer).lower()
        else:
            input_source = TABLE_PROCESSOR.process_input(table_content_copy, question, []).lower()
        input_sources.append(input_source)
        
        last_cell = str(table_content['rows'][-1][-1]).lower().strip()[:15]
        n_row = len(table_content['rows'])
        truncated = (f'row {n_row}' not in input_source) or (last_cell not in input_source.split('|')[-1].strip())
        input_truncated.append(truncated)
        # print('-------')
        # print(last_cell)
        # print(table_content['rows'])
        # print(truncated)
        # print(input_source)

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
    # squall_tableqa can be plus or default
    datasets = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/squall_plus.py", 
                            split_id=1, plus=False)
    train_dataset = datasets["test"]
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base")
    train_dataset = train_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer":tokenizer, 
                   "max_source_length": 1024,
                   "max_target_length": 512,
                   "ignore_pad_token_for_loss": True,
                   "padding": False}, 
        batched=True,)
    print(tokenizer.decode(train_dataset['input_ids'][0]))
    print(tokenizer.decode(train_dataset['labels'][0]))
