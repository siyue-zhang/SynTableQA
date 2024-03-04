import re
import numpy as np
import pandas as pd
from pandasql import sqldf
from fuzzywuzzy import process
from copy import deepcopy
from collections import defaultdict
import pickle
import logging

# select  col1 from w where col3 = 'st. john''s'
# select col10 from w where col2 = '10' and col6 = '283'
# select count ( col0 ) from w where col2 = 'michael schumacher' and col3 = 'david coulthard' and col5 = 'mclaren - mercedes'
# select col6 from w where col2 = '"buzzkill"'
# select col10 from w where col4 = 27
# select col0 from w where col1 = '9.40' and col4 = '9.44'
# select col2 from w where col0 = 1

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


def string_check(pred, mapping, table):

	# BUGS
	# -> dev_117: select count ( col2 ) from w where col1 = 'east west' and col3 = 'none' and col0 ='sh 202'
	# 

	pred = pred.lower()
	pred_copy = deepcopy(pred)
	wwhere = ' where '
	aand = ' and '

	# 'select col3 from w where col2 = \'"home is where the hospital is"\''
	pairs = re.finditer(r'(\'.*? where .*?\')', pred)
	for x in pairs:
		ori = pred[x.start(0):x.end(0)]
		new = ori.replace(wwhere, ' !!! ')
		pred = pred.replace(ori, new)

	seperated = pred.split(wwhere)


	if len(seperated) == 1:
		return pred_copy
	elif len(seperated) > 2:
		print('ALARM ', pred)
		print(seperated)
		raise ValueError
	else:
		mapping_b = {mapping[k]:k for k in mapping}
		cell_dict = defaultdict(lambda: set())
		nm_headers = [f'col{i}' for i in range(len(table['header']))]
		types = table['types']
		for row in table['rows']:
			for k, cell in enumerate(row):
				cell_dict[str(cell)].add(nm_headers[k])

		if not cell_dict:
			return pred_copy

		before_where, after_where = seperated

		# "select  col5 from w where col4 = 'karen felix and don woodard'"
		pairs = re.finditer(r'(\'.*? and .*?\')', after_where)
		for x in pairs:
			# "select col3 from w where col9 = '1-0' and col6 = '1-0'"
			ori = after_where[x.start(0):x.end(0)]
			if "'" in ori[1:-1]:
				continue
			new = ori.replace(aand, ' ??? ')
			after_where = after_where.replace(ori, new)

		conds = after_where.split(aand)
		print('conds: ', conds)
		new_conds = []
		for cond in conds:

			col=None
			val=None
			for op in ['=','<','>']:
				if op in cond:
					loc = cond.index(op)
					col = cond[:loc].strip()
					val = cond[loc+1:].strip().replace(' ??? ', aand).replace(' !!! ', wwhere)
					break

			if not col or not val:
				new_conds.append(cond)
				continue

			if val[0]=="'":

				if val[-1]!="'":
					val+="'"

				val_in_quote = val[1:-1].replace("''","'")
				if val_in_quote not in cell_dict:
					candidates = list(cell_dict.keys())
					candidates = sorted([x for x in candidates if isinstance(x, str)], key=len)
					new_val, _ = process.extractOne(val_in_quote, candidates)
					col_candidates = cell_dict[new_val]
				else:
					col_candidates = cell_dict[val_in_quote]
					new_val = val_in_quote
				
				new_val = new_val.replace("'","''")
				new_val = f"'{new_val}'"

				if col in col_candidates:
					new_col = col
				elif len(col_candidates)==1:
					new_col = list(col_candidates)[0]
				else:
					if col not in mapping:
						new_col, _ = process.extractOne(col, col_candidates)
						print('HEHEHE ', new_col)
					else:
						nl_col = mapping[col]
						col_candidates = [mapping[x] for x in col_candidates]
						new_col, _ = process.extractOne(nl_col, col_candidates)
						new_col = mapping_b[new_col]

				new_cond = f'{new_col} {op} {new_val}'
				new_conds.append(new_cond)
			else:
				new_conds.append(cond)
				continue
		
		pred = f'{before_where.strip()} where {aand.join(new_conds)}' 
		if pred.replace(' ','') != pred_copy.replace(' ',''):
			print('From: ', pred_copy)
			print('To: ', pred)

	return pred



def postprocess_text(decoded_preds, eval_dataset, fuzzy):
	predictions=[]
	L = len(decoded_preds)
	for i, pred in enumerate(decoded_preds):
		
		id = eval_dataset['id'][i]
		logging.warning(f'{i}/{L}: {id}')
		logging.warning(f'Raw prediction: {pred}')

		pred=pred.replace('!', ' !').replace('!>', '<').replace('< =', '<=').replace('> =', '>=')
		table = eval_dataset['table'][i]
		
		nl_header = table['header']
		n_col = len(nl_header)
		nm_header = [f"col{j}" for j in range(n_col)]

		for j in range(n_col):
			pred = pred.replace(nl_header[j], nm_header[j])
 
		mapping = {ori: col for ori, col in zip(nm_header, nl_header)}

		if fuzzy:
			try:
				pred = string_check(pred, mapping, table)
			except Exception as e:
				pass
		predictions.append(pred)

		print('\n---------------------\n')
	
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

		# save intermediate results
		dump_data = {
			"eval_dataset": eval_dataset,
			"decoded_preds": decoded_preds,
			"stage": stage,
			"tokenizer": tokenizer,
			"meta": meta
		}

		if stage:
			with open(f'tmp/{stage}_tmp.pkl', 'wb') as file:
				pickle.dump(dump_data, file)
			assert 1==2

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