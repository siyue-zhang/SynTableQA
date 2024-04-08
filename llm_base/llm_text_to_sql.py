from datasets import load_dataset
import json
import pandas as pd
import re
from copy import deepcopy
from openai import OpenAI

import sys
sys.path.append('./')

from metric.squall_evaluator import Evaluator
from llm_base.llm_tableqa import get_default_processor, call_gpt
from metric.squall import postprocess_text

# node data/squall/eval/evaluator.js 

client = OpenAI(
    api_key='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'
)
model = 'gpt-3.5-turbo'
# model="gpt-4-0125-preview"


def preprocess_squall(examples):
    # preprocess the squall datasets for the model input
    TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024, target_delimiter='|')
    # only keep cell truncation, remove row deletion
    # TABLE_PROCESSOR.table_truncate_funcs = TABLE_PROCESSOR.table_truncate_funcs[0:1]

    nts = examples["nt"]
    tbls = examples["tbl"]
    nls = examples["question"]
    sqls = examples["query"]
    json_paths = examples["json_path"]

    num_ex = len(tbls)
    table_contents = {}

    input_sources = []
    output_targets = []

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
                # nl_header = nl_header.replace('<','!>')
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

        question = nl
        input_source = TABLE_PROCESSOR.process_input(table_content_x, question, []).lower()
        input_source = input_source.split('row 2 :')[0].strip()

        for j in range(len(table_content['ori_header'])):
            sql = sql.replace(table_content['ori_header'][j], table_content['nl_header'][j])
        output = sql

        # input_source = input_source.replace('<', '!>')
        # output = output.replace('<', '!>')

        input_sources.append(input_source)
        output_targets.append(output)

    examples = examples.add_column("ori_headers", all_ori_headers)
    examples = examples.add_column("nl_headers", all_nl_headers)
    examples = examples.add_column("input_sources", input_sources)
    examples = examples.add_column('output_targets', output_targets)

    return examples

if __name__=='__main__':

    dataset_name = 'squall'
    squall_evaluator = Evaluator()

    # Load dataset
    if dataset_name == 'squall':
        task = "./task/squall_plus.py"
        raw_datasets = load_dataset(task, 
                                    plus=True, 
                                    split_id=1)
        file_path = "llm_base/squall_plus_llm_text_to_sql_test1.csv"
    elif dataset_name == 'wikisql':
        task = "./task/wikisql_robut.py"
        raw_datasets = load_dataset(task, 
                                    split_id=0)
        # file_path = "llm_base/squall_plus_llm_text_to_sql_test0.csv"

    # for prompt exampler
    train_dataset = raw_datasets['validation']
    train_dataset = preprocess_squall(train_dataset) if dataset_name == 'squall' else None

    exampler='Based on the table and schema, write the SQL query to answer the question. Only response the SQL query.\n\n'
    ids = [1, 69, 141, 163, 259, 422, 1719, 1749]
    for i in ids:
        _exampler = train_dataset[i]['input_sources']
        _sql = train_dataset[i]['output_targets']
        exampler += "Question: " + _exampler.replace('col : ','\n\nTable: col : ') + f"\n\nSQL: {_sql}"
        if i!=ids[-1]:
            exampler += '\n\n'

    # for test samples
    test_dataset = raw_datasets['test']
    test_dataset = preprocess_squall(test_dataset) if dataset_name == 'squall' else None

    df = pd.read_csv(file_path)
    assert len(test_dataset)==df.shape[0]
    
    for i, row in df.iterrows():

        # if i < 10:
        #     continue

        if i > 200:
            break

        print(f'\n-----{i}--------\n')
        input_source = test_dataset[i]['input_sources']
        cur_prompt = exampler + "\n\nQuestion: " + input_source.replace('col : ','\n\nTable: col : ') + "\n\nSQL: "
        print(cur_prompt)
        returned, log_prob = call_gpt(cur_prompt)
        if '```sql' in returned:
            returned = returned.replace('```sql\n','').replace('\n```','')
        df.loc[i, 'query_pred_llm'] = returned
        df.loc[i, 'log_prob_llm_text_to_sql'] = log_prob
        print(returned,'\n\n')

        if dataset_name=='squall':
            predictions = postprocess_text([returned], test_dataset.select([i]), True)
            fuzzy_query = predictions[0]["result"][0]["sql"]
            correct_flag, predicted = squall_evaluator.evaluate_text_to_sql(predictions)
            
            while isinstance(predicted, list):
                predicted=predicted[0]
            while isinstance(correct_flag, list):
                correct_flag=correct_flag[0]                

        df.loc[i, 'query_fuzzy_llm'] = fuzzy_query
        df.loc[i, 'queried_ans_llm'] = predicted
        df.loc[i, 'acc_llm_text_to_sql'] = int(correct_flag)

    df.to_csv(file_path, index=False)   