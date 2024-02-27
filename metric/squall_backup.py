import json
import re
import numpy as np
import pandas as pd
from fuzzywuzzy import process
from metric.squall_evaluator import Evaluator

def find_best_match(contents, col, ori):
    final_strings = []
    done = False
    for c in contents:
        for cc in c:
            if col == cc['col']:
                strings = cc['data']
                for item in strings:
                    if isinstance(item, list):
                        for ii in item:
                            final_strings.append(str(ii))
                    else:
                        final_strings.append(str(item))
                done = True
            if done:
                break
        if done:
            break
    assert len(final_strings)>0, f'strings empty {final_strings}'
    final_strings = list(set(final_strings))
    best_match, _ = process.extractOne(ori, final_strings)
    return best_match

def find_fuzzy_col(col, mapping):
    assert col not in mapping
    # col->ori
    mapping_b = {value: key for key, value in mapping.items()}
    match = re.match(r'^(c\d+)', col)
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

def fuzzy_replace(pred, table_id, mapping):
    verbose = False
    table_path = f'./data/squall/tables/json/{table_id}.json'
    with open(table_path, 'r') as file:
        contents = json.load(file)
    contents = contents["contents"]
    ori_pred = str(pred)

    # select c3 from w where c1 = 'american mcgee's crooked house' 
    indices = []
    for i, char in enumerate(pred):
        if char == "'":
            indices.append(i)
    if len(indices) == 3:
        pred = pred[:indices[0]] + "\"" + pred[indices[0]+1:]
        pred = pred[:indices[2]] + "\"" + pred[indices[2]+1:]

    cols = []
    for c in contents:
        for cc in c:
            cols.append(cc['col'].strip().replace(' ','_').lower())

    pairs = re.findall(r'where (c[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?\'([^"]*?"[^"]*?\'[^"]*".*?)\'', pred)
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

    pairs = re.finditer(r'where (c[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?\'(.{1,}?)\'', pred)
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


    pairs = re.finditer(r'where (c[0-9]{1,}.{,20}?) in \(\s*?\'(.{1,}?)\'\s*?,\s*?\'(.{1,}?)\'\s*?\)', pred)
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


    pairs = re.finditer(r'where (c[0-9]{1,}.{,20}?) in \(\s*?\'(.{1,}?)\'\s*?,\s*?\'(.{1,}?)\'\s*?, \'(.{1,}?)\'\s*?\)', pred)
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
        print(table_path)
        print(f'From: {ori_pred}')
        print(f'To  : {pred}\n')

    return pred

def postprocess_text(decoded_preds, eval_dataset, fuzzy):
    predictions = []
    for i, pred in enumerate(decoded_preds):
        pred=pred.replace('!>', '<').replace('< =', '<=').replace('> =', '>=')
        table_id = eval_dataset['tbl'][i]
        nt_id = eval_dataset['nt'][i]
        nl_headers = eval_dataset['nl_headers'][i]
        ori_headers = eval_dataset['ori_headers'][i]
        nl = eval_dataset['question'][i]
        label = eval_dataset['query'][i]
        # repalce the natural language header with c1, c2, ... headers
        for j, h in enumerate(nl_headers):
            pred=pred.replace(h, ori_headers[j])
        # print('pred before fuzzy: ', pred)
        if fuzzy:
            mapping = {ori: col for ori, col in zip(ori_headers, nl_headers)}
            pred = fuzzy_replace(pred, table_id, mapping)
        # print('nt: ', nt_id, 'fuzzy query: ', pred)
        result_dict = {"sql": pred, "id": nt_id, "tgt": label}
        res = {"table_id": table_id,
               "json_path": eval_dataset['json_path'][i],
               "db_path": eval_dataset['db_path'][i], 
               "result": [result_dict], 'nl': nl}
        predictions.append(res)
    
    return predictions

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
        predictions = postprocess_text(decoded_preds, eval_dataset, fuzzy)
        fuzzy_query = [ex["result"][0]["sql"] for ex in predictions]
        evaluator = Evaluator()
        correct_flag, predicted = evaluator.evaluate_text_to_sql(predictions)
        nl_headers = ['|'.join(nl) for nl in eval_dataset['nl_headers']]

        if stage:
            to_save = {'id': eval_dataset['nt'],
                       'tbl': eval_dataset['tbl'],
                       'question': eval_dataset['question'],
                       'answer': eval_dataset['answer_text'],
                       'acc': [int(b) for b in correct_flag],
                       'query': eval_dataset['query'], 
                       'query_pred': decoded_preds,
                       'query_fuzzy':fuzzy_query,
                       'queried_ans': predicted,
                       'src': eval_dataset['src'],
                       'nl_headers': nl_headers,
                        'truncated': eval_dataset['truncated'],
                       'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids'])}
            if meta:
                to_save['log_probs_sum'] = meta['log_probs_sum']
                to_save['log_probs_avg'] = meta['log_probs_mean']

            df = pd.DataFrame(to_save)
            df.to_csv(f'./predict/squall/{stage}.csv', na_rep='',index=False)
            print('predictions saved! ', stage)

        return {"acc": np.round(np.mean(correct_flag),4)}
    return compute_metrics