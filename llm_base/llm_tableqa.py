from datasets import load_dataset
import json
import pandas as pd
import re
from copy import deepcopy

import sys
sys.path.append('./')
from utils.processor.table_linearize import IndexedRowTableLinearize
from utils.processor.table_truncate import CellLimitTruncate, RowDeleteTruncate
from utils.processor.table_processor import TableProcessor
from transformers import AutoTokenizer

from openai import OpenAI
from metric.squall_evaluator import Evaluator
from metric.wikisql import evaluate_example


def get_default_processor(max_cell_length, max_input_length, target_delimiter=', '):
    table_linearize_func = IndexedRowTableLinearize()
    table_truncate_funcs = [
        CellLimitTruncate(max_cell_length=max_cell_length,
                          tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large"),
                          max_input_length=max_input_length),
    ]
    processor = TableProcessor(table_linearize_func=table_linearize_func,
                               table_truncate_funcs=table_truncate_funcs,
                               target_delimiter=target_delimiter)
    return processor


def preprocess_squall(examples):
    # preprocess the squall datasets for the model input

    TABLE_PROCESSOR = get_default_processor(max_cell_length=50, max_input_length=1024, target_delimiter='|')

    tbls = examples["tbl"]
    nls = examples["question"]
    sqls = examples["query"]
    answer_texts = examples["answer_text"]
    json_paths = examples["json_path"]

    num_ex = len(tbls)
    table_contents = {}

    input_sources = []
    output_targets = []

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
        if len(table_content_copy['rows'])>50:
            table_content_copy['rows'] = table_content_copy['rows'][:50]
        input_source = TABLE_PROCESSOR.process_input(table_content_copy, question, []).lower()
        input_sources.append(input_source)

        output_target = TABLE_PROCESSOR.process_output(answer).lower()
        output_targets.append(output_target)

    examples = examples.add_column("input_sources", input_sources)
    examples = examples.add_column('output_targets', output_targets)

    return examples


def preprocess_wikisql(examples):

    TABLE_PROCESSOR = get_default_processor(max_cell_length=50, max_input_length=1024, target_delimiter=', ')
    input_sources = []
    output_targets = []
    tables = examples['table']
    answers = examples['answers']
    questions = examples['question']
    for i in range(len(examples['question'])): 
        table_content = tables[i]
        table_content_copy = deepcopy(table_content)
        answer = answers[i]
        question = questions[i]
        if len(table_content_copy['rows'])>50:
            table_content_copy['rows'] = table_content_copy['rows'][:50]
        input_source = TABLE_PROCESSOR.process_input(table_content_copy, question, []).lower()
        input_sources.append(input_source)
   
        output_target = TABLE_PROCESSOR.process_output(answer).lower()
        output_targets.append(output_target)

    examples = examples.add_column("input_sources", input_sources)
    examples = examples.add_column('output_targets', output_targets)

    return examples



def call_gpt(cur_prompt, temperature=0, version=3.5):

    client = OpenAI(
        api_key='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'
    )
    model = "gpt-4-0125-preview" if version == 4 else "gpt-3.5-turbo"
    ans = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": cur_prompt}
        ],
        temperature=temperature,
        logprobs=True,
        top_logprobs=1
    )
    returned = ans.choices[0].message.content
    log_probs = ans.choices[0].logprobs.content[0].logprob
    
    return returned, log_probs


if __name__=='__main__':

    version = 4
    # dataset_name = 'squall'
    dataset_name = 'wikisql'

    if dataset_name == 'squall':
        squall_evaluator = Evaluator()

    # Load dataset
    if dataset_name == 'squall':
        task = "./task/squall_plus.py"
        raw_datasets = load_dataset(task, 
                                    plus=True, 
                                    split_id=1)
        file_path = f"llm_base/gpt{version}/squall_plus_llm_tableqa_test1.csv"
    elif dataset_name == 'wikisql':
        task = "./task/wikisql_robut.py"
        raw_datasets = load_dataset(task, 
                                    split_id=0)
        file_path = f"llm_base/gpt{version}/wikisql_llm_tableqa_test0.csv"

    # for prompt exampler
    train_dataset = raw_datasets['validation']
    if len(train_dataset)>2000:
        train_dataset=train_dataset.select(range(2000))
    train_dataset = preprocess_squall(train_dataset) if dataset_name == 'squall' else preprocess_wikisql(train_dataset)

    exampler='Based on the table, answer the question. Only response the answer.\n\n'
    ids = [1, 128, 163, 275]
    for i in ids:
        _exampler = train_dataset[i]['input_sources']
        _answer = train_dataset[i]['output_targets']
        exampler += "Question: " + _exampler.replace('col : ','\n\nTable: col : ') + f"\n\nAnswer: {_answer}"
        if i!=ids[-1]:
            exampler += '\n\n'

    # for test samples
    test_dataset = raw_datasets['test']
    if dataset_name=='wikisql':
        test_dataset=test_dataset.select(range(2000))
    test_dataset = preprocess_squall(test_dataset) if dataset_name == 'squall' else preprocess_wikisql(test_dataset)

    df = pd.read_csv(file_path)
    # assert len(test_dataset)==df.shape[0]
    
    for i, row in df.iterrows():

        # if i < 1535:
        #     continue

        if i > 500:
            break

        print(f'\n-----{i}--------\n')
        input_source = test_dataset[i]['input_sources']
        cur_prompt = exampler + "\n\nQuestion: " + input_source.replace('col : ','\n\nTable: col : ') + "\n\nAnswer: "
        returned, log_prob = call_gpt(cur_prompt, version=version)
        df.loc[i, 'ans_llm_tableqa'] = returned
        df.loc[i, 'log_prob_llm_tableqa'] = log_prob
        # print(cur_prompt)
        # print(returned)
        # print(test_dataset['answers'][i])
        # assert 1==2
        if dataset_name=='squall':            
            correct_flag = squall_evaluator.evaluate_tableqa([{'pred': returned, 'nt': test_dataset['nt'][i]}], '|')
            if isinstance(correct_flag, list):
                correct_flag=correct_flag[0]
        else:
            answers = test_dataset['answers'][i]
            answers = ', '.join([a.strip().lower() for a in answers])
            correct_flag = evaluate_example(returned.lower(), answers.lower(), ', ')

        df.loc[i, 'acc_llm_tableqa'] = int(correct_flag)

    df.to_csv(file_path, index=False)   


