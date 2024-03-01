import json
import re
import numpy as np
import pandas as pd
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
from metric.squall_evaluator import Evaluator
from collections import defaultdict
from copy import deepcopy
import re

def header_check(pred, mapping_b):
    words = [x.strip() for x in pred.split()]
    correct_map = {}
    for w in words:
        if re.match(r'^\d*_', w) and w not in mapping_b:
            tmp = '_'.join(w.split('_')[1:])
            correct_map[w] = process.extractOne(tmp,sorted(mapping_b.keys(),key=len))[0]
    pred = ' '.join(words)
    for x in sorted(correct_map.keys(), key=len):
        pred = pred.replace(x, correct_map[x])
    return pred

def correctify(col, ori, to_replace, cell_dict, mapping, mapping_b, ori2=None, ori3=None, add_quote=False):

    if ori not in cell_dict:
        candidates = list(cell_dict.keys())
        candidates = sorted([x for x in candidates if isinstance(x, str)], key=len)
        # can = [x for x in candidates if len(x)>(len(ori)//2)]
        # if len(can)>0:
        #     candidates = can
        new_ori, _ = process.extractOne(ori, candidates)
        if add_quote:
            to_replace = to_replace.replace(f'"{ori}"', f"'{new_ori}'")
        else:
            to_replace = to_replace.replace(ori, new_ori)
    else:
        new_ori = ori

    if col not in cell_dict[new_ori]:
        if len(cell_dict[new_ori])>1:
            col_choices = list(cell_dict[new_ori])
            col_choices = sorted([mapping[x] for x in col_choices],key=len)
            tmp = deepcopy(col)
            prefix = tmp.split('_')[0]
            if prefix[0] == 'c' and prefix in mapping:
                tmp = tmp.replace(prefix, mapping[prefix])
            new_col, _ = process.extractOne(tmp, col_choices)
            new_col = mapping_b[new_col]
        else:
            new_col = list(cell_dict[new_ori])[0]
        to_replace = to_replace.replace(col, new_col)
    else:
        new_col = col
    
    if ori2 or ori3:
        candidates = [value for value in cell_dict if new_col in cell_dict[value]]
        candidates = sorted([x for x in candidates if isinstance(x, str)], key=len)

    if ori2 and ori2 not in cell_dict:
        can = [x for x in candidates if len(x)>(len(ori)//2)]
        if len(can)>0:
            new_ori2, _ = process.extractOne(ori2, can)
        else:
            new_ori2, _ = process.extractOne(ori2, candidates)
        to_replace = to_replace.replace(ori2, new_ori2)

    if ori3 and ori3 not in cell_dict:
        can = [x for x in candidates if len(x)>(len(ori)//2)]
        if len(can)>0:
            new_ori3, _ = process.extractOne(ori3, can)
        else:
            new_ori3, _ = process.extractOne(ori3, candidates)
        to_replace = to_replace.replace(ori3, new_ori3)

    return to_replace
        


def string_check(pred, mapping, contents):

    verbose = False
    contents = contents["contents"]
    ori_pred = str(pred)
    mapping_b = {mapping[k]:k for k in mapping}

    # select c3 from w where c1 = 'american mcgee's crooked house' 
    indices = []
    in_double_quote = False
    for i, char in enumerate(pred):
        if char == '"':
            in_double_quote = True if in_double_quote == False else False
        if char == "'" and in_double_quote == False:
            indices.append(i)
    if len(indices) == 3:
        pred = pred[:indices[0]] + "\"" + pred[indices[0]+1:]
        pred = pred[:indices[2]] + "\"" + pred[indices[2]+1:]

    cols = []
    cell_dict = defaultdict(lambda: set())
    for c in contents:
        for cc in c:
            col_name = cc['col'].strip().replace(' ','_').lower()
            cols.append(col_name)
            if cc['type'] == 'TEXT':
                for item in cc['data']:
                    cell_dict[item].add(col_name)
            elif cc['type'] == 'LIST TEXT':
                for item in cc['data']:
                    for list_item in item:
                        cell_dict[list_item].add(col_name)

    # select 1_id from w where 7_title = "alfie's birthday party"
    pairs = re.finditer(r'where (c[0-9]{1,}.{,20}?)\s*?[!=><]{1,}\s*?"([^"]*?\'[^"]*?)"', pred)
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

        to_replace = correctify(col, ori, to_replace, cell_dict, mapping, mapping_b, add_quote=True)
        replacement.append(to_replace)

    for i in range(len(tokens)):
        pred = pred.replace(tokens[i], replacement[i])
    
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

            if ori not in cell_dict:
                candidates = list(cell_dict.keys())
                candidates = sorted([x for x in candidates if isinstance(x, str)], key=len)
                can = [x for x in candidates if len(x)>(len(ori)//2)]
                if len(can)>0:
                    candidates = can
                new_ori, _ = process.extractOne(ori, candidates)
            else:
                new_ori = ori

            if col not in cell_dict[new_ori]:
                if len(cell_dict[new_ori])>1:
                    col_choices = list(cell_dict[new_ori])
                    col_choices = sorted([mapping[x] for x in col_choices], key=len)
                    tmp = deepcopy(col)
                    prefix = tmp.split('_')[0]
                    if prefix[0] == 'c' and prefix in mapping:
                        tmp = tmp.replace(prefix, mapping[prefix])
                    new_col, _ = process.extractOne(tmp, col_choices)
                    new_col = mapping_b[new_col]
                else:
                    new_col = list(cell_dict[new_ori])[0]
                pred = pred.replace(col, new_col)

            best_match = new_ori.replace('\'','\'\'')
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
        print(col, ori, to_replace)
        to_replace = correctify(col, ori, to_replace, cell_dict, mapping, mapping_b)
        replacement.append(to_replace)
        print(to_replace)
    
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
 
        to_replace = correctify(col, ori1, to_replace, cell_dict, mapping, mapping_b, ori2)

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

        to_replace = correctify(col, ori1, to_replace, cell_dict, mapping, mapping_b, ori2, ori3)
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

def postprocess_text(decoded_preds, eval_dataset, fuzzy):
    predictions = []
    for i, pred in enumerate(decoded_preds):
        pred=pred.replace('!', ' !').replace('!>', '<').replace('< =', '<=').replace('> =', '>=')
        table_id = eval_dataset['tbl'][i]
        nt_id = eval_dataset['nt'][i]
        nl_headers = eval_dataset['nl_headers'][i]
        ori_headers = eval_dataset['ori_headers'][i]
        nl = eval_dataset['question'][i]
        label = eval_dataset['query'][i]
        # print('\n', nt_id, ' : ', nl, ' ', table_id)
        # repalce the natural language header with c1, c2, ... headers
        mapping = {ori: col for ori, col in zip(ori_headers, nl_headers)}
        mapping_b = {mapping[k]:k for k in mapping}

        table_path = f'./data/squall/tables/json/{table_id}.json'
        with open(table_path, 'r') as file:
            contents = json.load(file)

        if fuzzy:
            pred = header_check(pred, mapping_b)
            used = []
            for h in sorted(nl_headers, key=len, reverse=True):
                if h in pred and all([h not in u for u in used]):
                    pred=pred.replace(h, mapping_b[h])
                    used.append(mapping_b[h])
            pred = string_check(pred, mapping, contents)

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