from datasets import load_dataset
import json
import pandas as pd
import re
import os
from copy import deepcopy
from openai import OpenAI
from pandasql import sqldf
import time

import sys
sys.path.append('./')

from metric.squall_evaluator import Evaluator
from llm_base.llm_tableqa import get_default_processor, call_gpt
from metric.squall import postprocess_text as squall_postproc
from metric.wikisql import postprocess_text as wiki_postproc
from metric.wikisql import evaluate_example
# node data/squall/eval/evaluator.js 

from din.din_squall import classification_prompt, easy_prompt, hard_prompt, debugger_prompt, GPT4_generation, GPT4_debug

# client = OpenAI(
#     api_key='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'
# )

def preprocess_squall(examples):
    # preprocess the squall datasets for the model input
    TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024, target_delimiter='|')

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
        if len(table_content_x['rows'])>5:
            table_content_x['rows'] = table_content_x['rows'][:5]
        input_source = TABLE_PROCESSOR.process_input(table_content_x, question, []).lower()

        for j in range(len(table_content['ori_header'])):
            sql = sql.replace(table_content['ori_header'][j], table_content['nl_header'][j])
        output = sql

        input_sources.append(input_source)
        output_targets.append(output)

    examples = examples.add_column("ori_headers", all_ori_headers)
    examples = examples.add_column("nl_headers", all_nl_headers)
    examples = examples.add_column("input_sources", input_sources)
    examples = examples.add_column('output_targets', output_targets)

    return examples


def preprocess_wikisql(examples):

    TABLE_PROCESSOR = get_default_processor(max_cell_length=15, max_input_length=1024, target_delimiter=', ')

    questions = examples["question"]
    tables = examples["table"]
    sqls = examples["sql"]

    inputs, outputs = [], []
    table_contents = {}
    
    for i in range(len(questions)):

        question = questions[i]
        table_content = tables[i]
        id = examples["id"][i]
        if id not in table_contents:
            table_content['header'] = [x.replace('\n', ' ').replace(' ','_').strip().lower() for x in table_content['header']]
            table_content['header'] = [f'{k+1}_{x}' for k, x in enumerate(table_content['header'])]
        else:
            table_content = table_contents[id]

        table_content_copy = deepcopy(table_content)

        # table_content_copy['rows'] = [table_content_copy['rows'][0]]
        table_content_copy['rows'] = []
        input_source = TABLE_PROCESSOR.process_input(table_content_copy, question, []).strip().lower()
        inputs.append(input_source)

        output = sqls[i]
        for k in range(len(table_content['header'])-1, -1, -1):
            output = output.replace(f"col{k}", table_content['header'][k])
        outputs.append(output)

    examples = examples.add_column("input_sources", inputs)
    examples = examples.add_column('output_targets', outputs)

    return examples


if __name__=='__main__':

    version = 3.5
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
        file_path = f"./llm_base/gpt{version}/squall_plus_llm_text_to_sql_test1.csv"
    elif dataset_name == 'wikisql':
        task = "./task/wikisql_robut.py"
        raw_datasets = load_dataset(task, 
                                    split_id=0)
        file_path = f"./llm_base/gpt{version}/wikisql_llm_text_to_sql_test0.csv"

    # for prompt exampler
    train_dataset = raw_datasets['validation']
    if len(train_dataset)>2000:
        train_dataset=train_dataset.select(range(2000))
    train_dataset = preprocess_squall(train_dataset) if dataset_name == 'squall' else preprocess_wikisql(train_dataset)

    # exampler='Based on the table schema, write the SQL query to answer the question. Only response the SQL query.\n\n'
    # ids = [1, 69, 141, 163, 259, 422, 1719, 1749]
    # for i in ids:
    #     _exampler = train_dataset[i]['input_sources']
    #     _sql = train_dataset[i]['output_targets']
    #     exampler += "Question: " + _exampler.replace('col : ','\n\nTable: col : ') + f"\n\nSQL: {_sql}"
    #     if i!=ids[-1]:
    #         exampler += '\n\n'

    # for test samples
    test_dataset = raw_datasets['test']
    if dataset_name=='wikisql':
        test_dataset=test_dataset.select(range(2000))
    test_dataset = preprocess_squall(test_dataset) if dataset_name == 'squall' else preprocess_wikisql(test_dataset)

    df = pd.read_csv(file_path)
    # assert len(test_dataset)==df.shape[0]
    
    for i, row in df.iterrows():

        # if i < 100:
        #     continue

        if i > 10:
            break

        print(f'\n-----{i}--------\n')
        
        prompt = classification_prompt
        prompt += f'Table: w, columns = {row["nl_headers"]}'
        prompt += f'\nQ: "{row["question"]}"'
        prompt += f'\nA: Let’s think step by step.'

        classification = None
        while classification is None:
            try:
                classification = GPT4_generation(prompt)
            except:
                time.sleep(3)
                pass
        try:
            predicted_class = classification.split("Label: ")[1]
        except:
            print("Slicing error for the classification module")
            predicted_class = '"NESTED"'


        if '"EASY"' in predicted_class:
            print("EASY")
            SQL = None
            while SQL is None:
                try:
                    prompt = easy_prompt
                    prompt += f'Table: w, columns = {row["nl_headers"]}'

                    prompt += f'\nQ: "{row["question"]}"'
                    prompt += f'\nSQL: '
                    # print(prompt)
                    SQL = GPT4_generation(prompt)
                except:
                    time.sleep(3)
                    pass
        else:
            print('xxxx ', classification)
            sub_questions = classification.split('questions = ["')[1].split('"]')[0]
            print("NESTED")
            print('sub_questions: ', sub_questions)
            SQL = None
            while SQL is None:
                try:
                    prompt = hard_prompt
                    prompt += f'Table: w, columns = {row["nl_headers"]}'
                    prompt += f'\nQ: "{row["question"]}"'
                    prompt += f'\nA: Let\'s think step by step. "{row["question"]}" can be solved by knowing the answer to the following sub-questions:["{sub_questions}"].'
                    # print(prompt)
                    SQL = GPT4_generation(prompt)
                    print('\n\n',SQL,'\n\n')
                except:
                    time.sleep(3)
                    pass
            try:
                SQL = SQL.split("Final SQL:")[1].strip()
                SQL = SQL.split('\n')[0]
            except:
                print("SQL slicing error")
                SQL = "SELECT"

        # print('before debug : ', SQL.lower())
        # debugged_SQL = None
        # while debugged_SQL is None:
        #     try:
        #         prompt = debugger_prompt
        #         prompt += f'#### Table: w, columns = {row["nl_headers"]}'
        #         prompt += f'\n#### Question: {row["question"]}'
        #         prompt += f'\n#### SQL QUERY: {SQL}'
        #         prompt += '\n#### FIXED SQL QUERY\nselect'
        #         debugged_SQL = GPT4_debug(prompt).replace("\n", " ").strip().replace("   ", " ").replace("  ", " ")
        #     except:
        #         time.sleep(3)
        #         pass
        # SQL = "select " + debugged_SQL

        # returned, log_prob = call_gpt(cur_prompt, version=version)
        # # print(returned)
        # if '```sql' in returned:
        #     returned = returned.replace('```sql\n','').replace('\n```','')
        returned  = SQL.lower()
        print('\nPrediction:')
        print(returned)
        # print('after debug  : ', returned)

        df.loc[i, 'query_pred_din'] = returned
        # df.loc[i, 'log_prob_llm_text_to_sql'] = log_prob
        # # print(returned,'\n\n')
        # # assert 1==2
        if dataset_name=='squall':
            predictions = squall_postproc([returned], test_dataset.select([i]), True)
            fuzzy_query = predictions[0]["result"][0]["sql"]
            correct_flag, predicted = squall_evaluator.evaluate_text_to_sql(predictions)
            
            while isinstance(predicted, list):
                predicted=predicted[0]
            while isinstance(correct_flag, list):
                correct_flag=correct_flag[0]
            
        else:
            eval_dataset = test_dataset.select([i])
            predictions = wiki_postproc([returned], eval_dataset, True)
            for k, pred in enumerate(predictions):

                answers = eval_dataset['answers'][k]
                answers = ', '.join([a.strip().lower() for a in answers])

                table = eval_dataset['table'][k]
                n_col = len(table['header'])
                w = pd.DataFrame.from_records(table['rows'],columns=[f"col{j}" for j in range(n_col)])
                try:
                    predicted_values = sqldf(pred).values.tolist()
                except Exception as e:
                    predicted_values = []
                predicted_values = [str(item).strip().lower() for sublist in predicted_values for item in sublist] if predicted_values else []
                
                predicted = ', '.join(predicted_values)
                correct_flag = evaluate_example(predicted, answers, target_delimiter=', ')
                fuzzy_query = pred

        print(predicted)
        print(correct_flag)
        assert 1==2
        df.loc[i, 'query_fuzzy_din'] = fuzzy_query
        df.loc[i, 'queried_ans_din'] = predicted
        df.loc[i, 'acc_din_text_to_sql'] = int(correct_flag)

    df.to_csv('DIN_squall.csv', index=False)   