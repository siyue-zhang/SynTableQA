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


def correctify(col, ori, to_replace, cell_dict, mapping, mapping_b):

	if ori not in cell_dict:
		candidates = list(cell_dict.keys())
		candidates = sorted([x for x in candidates if isinstance(x, str)], key=len)
		new_ori, _ = process.extractOne(ori, candidates)
		to_replace = to_replace.replace(ori, new_ori)
	else:
		new_ori = ori

	if col not in cell_dict[new_ori]:
		if len(cell_dict[new_ori])>1:
			col_choices = list(cell_dict[new_ori])
			col_choices = sorted([mapping[x] for x in col_choices],key=len)
			col_pred = mapping[col]
			tmp = '_'.join(col_pred.split('_')[1:])
			new_col, _ = process.extractOne(tmp, col_choices)
			new_col = mapping_b[new_col]
		else:
			new_col = list(cell_dict[new_ori])[0]
		to_replace = to_replace.replace(col, new_col)
	else:
		new_col = col

	return to_replace


def string_check(pred, mapping, table):
	print('A: ', pred)
	mapping_b = {mapping[k]:k for k in mapping}
	cell_dict = defaultdict(lambda: set())
	nm_headers = [f'col{i}' for i in range(len(table['header']))]
	types = table['types']
	for row in table['rows']:
		for k, cell in enumerate(row):
			if types[k].lower() == 'text':
				cell_dict[cell].add(nm_headers[k])

	# select col1 from w where col4 = 'prime minister of italy'
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

		to_replace = correctify(col, ori, to_replace, cell_dict, mapping, mapping_b)
		replacement.append(to_replace)
		print(to_replace)

	for i in range(len(tokens)):
		pred = pred.replace(tokens[i], replacement[i])

	print(cell_dict)
	print(pred)
	print(mapping)

	assert 1==2
	# select col0 from w where col1 = '9.40' and col4 = '9.44'

	return pred

def postprocess_text(decoded_preds, eval_dataset, fuzzy):
	predictions=[]
	for i, pred in enumerate(decoded_preds):
		# print(eval_dataset['question'][i])
		# print(eval_dataset['sql'][i])
		# print(pred)
		# assert 1==2[']
		pred=pred.replace('!', ' !').replace('!>', '<').replace('< =', '<=').replace('> =', '>=')
		table = eval_dataset['table'][i]
		
		nl_header = table['header']
		n_col = len(nl_header)
		nm_header = [f"col{j}" for j in range(n_col)]

		for j in range(n_col):
			pred = pred.replace(nl_header[j], nm_header[j])
 
		mapping = {ori: col for ori, col in zip(nm_header, nl_header)}

		if fuzzy:
			pred = string_check(pred, mapping, table)

		predictions.append(pred)
	
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
		predicted = []
		tapex_flag = []
		sep = ', '

		for i, pred in enumerate(predictions):

			answers = eval_dataset['answers'][i]
			answers = sep.join([a.strip().lower() for a in answers])

			table = eval_dataset['table'][i]
			n_col = len(table['header'])
			w = pd.DataFrame.from_records(table['rows'],columns=[f"col{j}" for j in range(n_col)])
			try:
				predicted_values = sqldf(pred).values.tolist()
			except Exception as e:
				predicted_values = []
			predicted_values = [str(item).strip().lower() for sublist in predicted_values for item in sublist] if predicted_values else []
			
			predicted_string = ', '.join(predicted_values)
			acc = evaluate_example(predicted_string, answers, target_delimiter=', ')
			tapex_flag.append(acc)
			predicted.append(sep.join(predicted_values))

		if stage:
			to_save = {'id': eval_dataset['id'],
					   'table_id': eval_dataset['table_id'],
					   'question': eval_dataset['question'],
					   'answers': eval_dataset['answers'],
					   'query': eval_dataset['sql'],
					   'acc': [int(b) for b in tapex_flag],
					   'query_pred': decoded_preds,
					   'query_fuzzy': predictions,
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