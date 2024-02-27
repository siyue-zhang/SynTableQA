import re
import numpy as np
import pandas as pd
from pandasql import sqldf
from fuzzywuzzy import process
from copy import deepcopy
# from metric.squall_evaluator import to_value_list, check_denotation

from collections import defaultdict

def evaluate_example(_predict_str: str, _ground_str: str, target_delimiter=', '):
    _predict_spans = _predict_str.split(target_delimiter)
    _ground_spans = _ground_str.split(target_delimiter)
    _predict_values = defaultdict(lambda: 0)
    _ground_values = defaultdict(lambda: 0)
    for span in _predict_spans:
        # 2,000
        if span.replace(',','').replace('.','').isdigit():
            span = span.replace(',','')
        try:
            _predict_values[float(span)] += 1
        except ValueError:
            _predict_values[span.strip()] += 1
    for span in _ground_spans:
        if span.replace(',','').replace('.','').isdigit():
            span = span.replace(',','')
        try:
            _ground_values[float(span)] += 1
        except ValueError:
            _ground_values[span.strip()] += 1

    _is_correct = _predict_values == _ground_values
    return _is_correct


def postprocess_text(decoded_preds):

    predictions=[]
    for pred in decoded_preds:
        pred=pred.replace('!>', '<').replace('< =', '<=').replace('> =', '>=')
        predictions.append(pred)
    
    return predictions

def find_best_match(contents, col, ori):
    final_strings = []
    tmp = {'id':0, 'agg':1}
    c = tmp[col] if col in tmp else int(col[3:])
    for row in contents['rows']:
        final_strings.append(row[c+2])
    assert len(final_strings)>0, f'strings empty {final_strings}'
    final_strings = list(set(final_strings))
    best_match, _ = process.extractOne(ori, final_strings)
    return best_match

def find_fuzzy_col(col, mapping):
    assert col not in mapping
    # col->ori
    mapping_b = {value: key for key, value in mapping.items()}
    match = re.match(r'^(col\d+)', col)
    # if match:
    #     c_num = match.group(1)
    #     # assert c_num in mapping, f'{c_num} not in {mapping}'
    #     if c_num not in mapping:
    #         print(f'predicted {c_num} is not valid ({mapping})')
    #         return list(mapping.keys())[0]
    #     else:
    #         best_match, _ = process.extractOne(col.replace(c_num, mapping[c_num]), [value for _, value in mapping.items()])
    # else:
    #     best_match, _ = process.extractOne(col, [value for _, value in mapping.items()])
    if match and match.group(1) in mapping:
        return match.group(1)
    else:
        best_match, _ = process.extractOne(col, [value for _, value in mapping.items()])
        return mapping_b[best_match]

def fuzzy_replace(table_content, pred, mapping):

    verbose = False
    contents = table_content
    ori_pred = str(pred)

    # select col0 from w where col1< 150817.6878461314 and col3 = '1.928 (1.89%)'
    # print(pred)
    # assert 1==2

    # select 10_location from w where 4_date = 'november 20'
    # dev-160 select col1 from w where col3 ='shark' and col2 = 'week 14'
    # dev-164 select 3_country from w where 4_interview_score = 9.40 and 7_average = 9.44
    # select col0 from w where col1_score = 9.40 and col4 = 9.44
    # 

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

    # select col1 from w where col0 = "patrick O'Bryant"
    pattern = r'where (col[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?"([^"]*?\'[^"]*?)"'
    pairs = re.findall(pattern, pred)
    if len(pairs)>0:
        for col, ori in pairs:
            if col not in cols:
                print(f'A: {col} not in {cols}, query ({pred})')
                col_replace = find_fuzzy_col(col, mapping)
                pred = pred.replace(col, col_replace)
                print(f' {col}-->{col_replace}')
                col = col_replace
            best_match = find_best_match(contents, col, ori)
            pred = pred.replace(ori, best_match)
        if verbose:
            print(f'A: in col: {col}, string: {ori} is replaced by {best_match}')

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

    for j in range(len(buf)):
        pred = pred.replace(f'[X{j}]', f'\'{buf[j]}\'')
    
    if verbose and (pred != ori_pred):
        print('String is replaced by fuzzy match!')
        print(f'From: {ori_pred}')
        print(f'To  : {pred}\n')
    
    return pred


def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=None):    
    def compute_metrics(eval_preds, meta=None):
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
        tapex_flag = []
        sep = ', '
        for i, pred in enumerate(predictions):
            # print('bf: ', pred)
            answers = eval_dataset['answers'][i]
            answers = sep.join([a.strip().lower() for a in answers])
            table = eval_dataset['table'][i]
            nl_header = [x.replace('\n', ' ').replace(' ','_').strip().lower() for x in table['header']]
            is_expand = ['_number' in h for h in nl_header]
            n_col = len(nl_header)
            nm_header = ['id', 'agg'] + [f"col{j}" for j in range(n_col-2)]
            print(nl_header)
            print(nm_header)
            print(pred)
            for j in range(n_col):
                pred = pred.replace(nl_header[j], nm_header[j])
            # convert all cell values into lower case
            table['rows'] = [[item.lower() for item in row] for row in table['rows']]

            if fuzzy:
                mapping = {ori: col for ori, col in zip(nm_header, nl_header)}
                pred = fuzzy_replace(table, pred, mapping)
                fuzzy_query.append(pred)

            w = deepcopy(table)
            w['header'] = nm_header
            for row in w['rows']:
                for j, value in enumerate(row):
                    if is_expand[j]:
                        row[j] = float(value) if value != '' else ''
                    else:
                        row[j] = str(value)
            w = pd.DataFrame.from_records(w['rows'],columns=w['header'])

            try:
                predicted_values = sqldf(pred).values.tolist()
            except Exception as e:
                predicted_values = []
                
            print(eval_dataset['question'][i])
            print(nl_header)
            print(w)
            print(w.dtypes)
            print(pred)
            print(predicted_values)
            assert 1==2

            # Flatten the list and convert elements to strings
            predicted_values = [str(item).strip().lower() for sublist in predicted_values for item in sublist] if predicted_values else []
            pred = sep.join(predicted_values)
            tapex_acc = evaluate_example(pred, answers, sep)
            tapex_flag.append(tapex_acc)
            predicted.append(sep.join(predicted_values))

        if stage:
            to_save = {'id': eval_dataset['id'],
                       'table_id': eval_dataset['table_id'],
                       'question': eval_dataset['question'],
                       'answers': eval_dataset['answers'],
                       'acc': [int(b) for b in tapex_flag],
                       'query_pred': predictions,
                       'query_fuzzy': fuzzy_query,
                       'queried_ans': predicted,
                       'truncated': eval_dataset['truncated'],
                       'perturbation_type': eval_dataset['perturbation_type'],
                       'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids'])}
            if meta:
                to_save['log_probs_sum'] = meta['log_probs_sum']
                to_save['log_probs_avg'] = meta['log_probs_mean'] 

            df = pd.DataFrame(to_save)
            df.to_csv(f'./predict/wikisql/{stage}.csv', na_rep='',index=False)
            print('predictions saved! ', stage)

        return {"acc": np.round(np.mean(tapex_flag),4)}
    return compute_metrics