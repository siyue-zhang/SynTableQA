import json
import re
import numpy as np
import pandas as pd
from pandasql import sqldf
import sqlite3
from fuzzywuzzy import process
from copy import deepcopy
from metric.squall_evaluator import to_value_list, check_denotation
from utils.misc import fetch_table_data

def postprocess_text(decoded_preds):

    predictions=[]
    for pred in decoded_preds:
        pred=pred.replace('!>', '<').replace('< =', '<=').replace('> =', '>=')
        predictions.append(pred)
    
    return predictions

def find_best_match(contents, col, ori):
    final_strings = []
    for row in contents['rows']:
        c = int(col[3:])
        final_strings.append(row[c])
    assert len(final_strings)>0, f'strings empty {final_strings}'
    final_strings = list(set(final_strings))
    best_match, _ = process.extractOne(ori, final_strings)
    return best_match

def find_fuzzy_col(col, mapping):
    assert col not in mapping
    # col->ori
    mapping_b = {value: key for key, value in mapping.items()}
    match = re.match(r'^(col\d+)', col)
    if match:
        c_num = match.group(1)
        # assert c_num in mapping, f'{c_num} not in {mapping}'
        if c_num not in mapping:
            print(f'predicted {c_num} is not valid ({mapping})')
            return list(mapping.keys())[0]
        else:
            best_match, _ = process.extractOne(col.replace(c_num, mapping[c_num]), [value for _, value in mapping.items()])
    else:
        best_match, _ = process.extractOne(col, [value for _, value in mapping.items()])
    return mapping_b[best_match]

def fuzzy_replace(table_content, pred, mapping):

    # verbose = True
    contents = table_content
    ori_pred = str(pred)

    # select col1 from table_1_10797463_1 where col4 = 65.9%
    pattern = r'((col\d+)\s*=\s*(\d+?\.\d+?%))'
    pairs = re.findall(pattern, pred)
    if len(pairs)>0:
        for old, col, val in pairs:
            new = old.replace(val, f"'{val}'")
            pred = pred.replace(old, new)

    # select c3 from w where c1 = 'american mcgee's crooked house' 
    indices = []
    for i, char in enumerate(pred):
        if char == "'":
            indices.append(i)
    if len(indices) == 3 and '"' not in pred:
        pred = pred[:indices[0]] + "\"" + pred[indices[0]+1:]
        pred = pred[:indices[2]] + "\"" + pred[indices[2]+1:]
    cols = list(mapping.keys())
    
    pairs = re.findall(r'where (col[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?\'([^"]*?"[^"]*?\'[^"]*".*?)\'', pred)
    # select c5 from w where c2 = '"i'll be your fool tonight"'
    buf = []
    n = 0
    if len(pairs)>0:
        for col, ori in pairs:
            if col not in cols:
                if 'and' in col:
                    col = col.split('and')[-1].strip()
                if 'or ' in col:
                    col = col.split('or')[-1].strip()
            if col not in cols:
                print(f'A: {col} not in {cols}, query ({pred})')
                col_replace = find_fuzzy_col(col, mapping)
                pred = pred.replace(col, col_replace)
                print(f' {col}-->{col_replace}')
                col = col_replace
            best_match = find_best_match(contents, col, ori)
            best_match = best_match.replace('\'','\'\'')
            pred = pred.replace(f'\'{ori}\'', f'[X{n}]')
            n += 1
            buf.append(best_match)

    pairs = re.finditer(r'where (col[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?\'(.{1,}?)\'', pred)
    tokens = []
    replacement = []
    for idx, match in enumerate(pairs):
        start = match.start(0)
        end = match.end(0)
        col = pred[match.start(1):match.end(1)]
        ori = pred[match.start(2):match.end(2)]
        to_replace = pred[start:end]

        token = str(idx) + '_'*(end-start-len(str(idx)))
        tokens.append(token)
        pred = pred[:start] + token + pred[end:]

        if col not in cols:
            if 'and' in col:
                col = col.split('and')[-1].strip()
            if 'or ' in col:
                col = col.split('or')[-1].strip()
        if verbose:
            print(f'B: part to be replaced: {to_replace}, col: {col}, string: {ori}')

        if col not in cols:
            print(f'B: {col} not in {cols}, query ({pred})')
            col_replace = find_fuzzy_col(col, mapping)
            to_replace = to_replace.replace(col, col_replace)
            print(f' {col}-->{col_replace}')
            col = col_replace
        best_match = find_best_match(contents, col, ori)
        to_replace = to_replace.replace(ori, best_match)
        replacement.append(to_replace)

    for i in range(len(tokens)):
        pred = pred.replace(tokens[i], replacement[i])


    pairs = re.finditer(r'where (col[0-9]{1,}.{,20}?) in \(\s*?\'(.{1,}?)\'\s*?,\s*?\'(.{1,}?)\'\s*?\)', pred)
    tokens = []
    replacement = []
    for idx, match in enumerate(pairs):
        start = match.start(0)
        end = match.end(0)
        col = pred[match.start(1):match.end(1)]
        ori1 = pred[match.start(2):match.end(2)]
        ori2 = pred[match.start(3):match.end(3)]
        if re.search(r"'\s*,\s*'", ori1+ori2):
            # select sum ( c5_number ) from w where c1 in ( 'argus', 'james carruthers', 'hydrus' )
            continue
        to_replace = pred[start:end]

        token = str(idx) + '_'*(end-start-len(str(idx)))
        tokens.append(token)
        pred = pred[:start] + token + pred[end:]

        if col not in cols:
            if 'and' in col:
                col = col.split('and')[-1].strip()
            if 'or ' in col:
                col = col.split('or')[-1].strip()
        if verbose:
            print(f'C: part to be replaced: {to_replace}, col: {col}, string: {ori1}, {ori2}')
 
        if col not in cols:
            print(f'C: {col} not in {cols}, query ({pred})')
            col_replace = find_fuzzy_col(col, mapping)
            to_replace = to_replace.replace(col, col_replace)
            print(f' {col}-->{col_replace}')
            col = col_replace

        if verbose:
            print(f'C')
 
        for ori in [ori1, ori2]:
            best_match = find_best_match(contents, col, ori)
            to_replace = to_replace.replace(ori, best_match)
        replacement.append(to_replace)

    for i in range(len(tokens)):
        pred = pred.replace(tokens[i], replacement[i])


    pairs = re.finditer(r'where (col[0-9]{1,}.{,20}?) in \(\s*?\'(.{1,}?)\'\s*?,\s*?\'(.{1,}?)\'\s*?, \'(.{1,}?)\'\s*?\)', pred)
    tokens = []
    replacement = []
    for idx, match in enumerate(pairs):
        start = match.start(0)
        end = match.end(0)
        col = pred[match.start(1):match.end(1)]
        ori1 = pred[match.start(2):match.end(2)]
        ori2 = pred[match.start(3):match.end(3)]
        ori3 = pred[match.start(4):match.end(4)]
        to_replace = pred[start:end]

        token = str(idx) + '_'*(end-start-len(str(idx)))
        tokens.append(token)
        pred = pred[:start] + token + pred[end:]

        if verbose:
            print(f'D: part to be replaced: {to_replace}, col: {col}, string: {ori1}, {ori2}, {ori3}')
        if col not in cols:
            print(f'D: {col} not in {cols}, query ({pred})')
            col_replace = find_fuzzy_col(col, mapping)
            to_replace = to_replace.replace(col, col_replace)
            print(f' {col}-->{col_replace}')
            col = col_replace
        for ori in [ori1, ori2, ori3]:
            best_match = find_best_match(contents, col, ori)
            to_replace = to_replace.replace(ori, best_match)
        replacement.append(to_replace)

    for i in range(len(tokens)):
        pred = pred.replace(tokens[i], replacement[i])

        if verbose:
            print(f'D')

    for j in range(len(buf)):
        pred = pred.replace(f'[X{j}]', f'\'{buf[j]}\'')
    
    if pred != ori_pred:
        print('String is replaced by fuzzy match!')
        print(f'From: {ori_pred}')
        print(f'To  : {pred}\n')
    
    return pred


def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=None):    
    def compute_metrics(eval_preds):
        # nonlocal tokenizer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # prepare the prediction format for the evaluator
        predictions = postprocess_text(decoded_preds)
        predicted = []
        fuzzy_query = []
        correct_flag = []
        table_contents = {}

        db_path = f'./data/wikisql/train.db'
        train_connection = sqlite3.connect(db_path)
        train_connection.row_factory = sqlite3.Row  # Use Row factory to get rows as dictionaries
        train_connection.text_factory = lambda b: b.decode(errors = 'ignore')

        for i, pred in enumerate(predictions):
            id = eval_dataset['id'][i]
            answers = eval_dataset['answers'][i]
            table = eval_dataset['table'][i]
            table_id = 'table_' + eval_dataset['table_id'][i].replace('-','_')

            nl_header = [x.replace(' ','_').lower() for x in table['header']]
            n_col = len(table["header"])
            nm_header = [f"col{j}" for j in range(n_col)]
            question = eval_dataset['question'][i]

            for j in range(n_col):
                pred = pred.replace(f'{j+1}_{nl_header[j]}', nm_header[j])

            if id[0]=='w':
                connection = train_connection
                if fuzzy:
                    if table_id not in table_contents:
                        table_content = fetch_table_data(connection, table_id)
                    else:
                        table_content = table_contents[table_id]

                    mapping = {ori: col for ori, col in zip(nm_header, nl_header)}
                    pred = fuzzy_replace(table_content, pred, mapping)
                    pred = pred.replace('from w', f'from {table_id}')
                    fuzzy_query.append(pred)

                cursor = connection.cursor()
                predicted_values = list()
                try:
                    cursor.execute(pred)
                    result = cursor.fetchall()
                    predicted_values = [str(cell) for row in result for cell in row] if result else []
                except Exception as e:
                    predicted_values = []

            else:
                # for robut data we can not use DB, need to query on dataframe
                if fuzzy:
                    mapping = {ori: col for ori, col in zip(nm_header, nl_header)}
                    pred = fuzzy_replace(table, pred, mapping)
                    fuzzy_query.append(pred)

                w = deepcopy(table)
                w['header'] = nm_header
                w = pd.DataFrame.from_records(w['rows'],columns=w['header'])

                try:
                    predicted_values = sqldf(pred).values.tolist()
                except Exception as e:
                    predicted_values = []

                # Flatten the list and convert elements to strings
                predicted_values = [str(item) for sublist in predicted_values for item in sublist] if predicted_values else []
                    
            predicted.append(predicted_values)
            predicted_values = to_value_list(predicted_values)
            target_values = to_value_list(answers)
            correct = check_denotation(target_values, predicted_values)
            correct_flag.append(correct)
            # print(correct, '\n-------')

        if stage:
            to_save = {'id': eval_dataset['id'],
                       'table_id': eval_dataset['table_id'],
                       'question': eval_dataset['question'],
                       'answer': eval_dataset['answers'],
                       'acc': [int(b) for b in correct_flag],
                       'query_pred': predictions,
                       'query_fuzzy':fuzzy_query,
                       'queried_ans': predicted,
                       'perturbation_type': eval_dataset['perturbation_type'],
                       'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids'])}
            df = pd.DataFrame(to_save)
            df.to_csv(f'./predict/wikisql/{stage}.csv', na_rep='')
            print('predictions saved! ', stage)

        return {"acc": np.round(np.mean(correct_flag),4)}
    return compute_metrics