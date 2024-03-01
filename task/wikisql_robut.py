import json
import os
import sys
sys.path.append('./')
import datasets
from datasets import load_dataset
from utils.misc import split_list
from copy import deepcopy
from utils.executor import retrieve_wikisql_query_answer_tapas, _TYPE_CONVERTER
from utils.query import Query
from utils.dbengine import DBEngine

_DATA_URL = (
		"https://raw.githubusercontent.com/yilunzhao/RobuT/main/robut_data.zip"
)

_AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
_COND_OPS = ["=", ">", "<", "OP"]

def _convert_table_types(table):
	"""Runs the type converter over the table cells."""
	ret_table = deepcopy(table)
	types = ret_table['types']
	ret_table['real_rows'] = ret_table['rows']
	typed_rows = []
	for row in ret_table['rows']:
		typed_row = []
		for column, cell_value in enumerate(row):
			typed_row.append(_TYPE_CONVERTER[types[column]](cell_value))
		typed_rows.append(typed_row)
	ret_table['rows'] = typed_rows
	return ret_table

class WikisqlConfig(datasets.BuilderConfig):
		"""BuilderConfig for Wikisql."""

		def __init__(self, split_id, perturbation_type='original', **kwargs):
				"""BuilderConfig for Wikisql.
				Args:
					**kwargs: keyword arguments forwarded to super.
				"""
				super(WikisqlConfig, self).__init__(**kwargs)
				self.split_id = split_id
				self.perturbation = perturbation_type

class Wikisql(datasets.GeneratorBasedBuilder):

		BUILDER_CONFIG_CLASS = WikisqlConfig

		def _info(self):
				return datasets.DatasetInfo(
						features = datasets.Features(
								{
										"id": datasets.Value("string"),
										"table_id": datasets.Value("string"),
										"question": datasets.Value("string"),
										"answers": datasets.features.Sequence(datasets.Value("string")),
										"sql": datasets.Value("string"),
										"table": {
												"header": datasets.features.Sequence(datasets.Value("string")),
												"rows": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
										},
										"perturbation_type": datasets.Value("string"),
										"split_key": datasets.Value("string"),
								}
						),
				)
	
		def _split_generators(self, dl_manager):

				table_ids = []
				table_data = []
				qa_data = []
				for x in ['train', 'dev', 'test']:
					ori_table_ids = []
					ori_table_data = {}
					with open(f'data/wikisql/{x}.tables.jsonl', "r", encoding="utf-8") as file:
						for line in file:
							json_data = json.loads(line)
							table_id = json_data["id"]
							if table_id not in ori_table_ids:
								ori_table_ids.append(table_id)
								ori_table_data[table_id] = json_data
					table_ids.append(ori_table_ids)
					table_data.append(ori_table_data)

					with open(f'data/wikisql/{x}.jsonl', "r", encoding="utf-8") as file:
						ori_qa_data = []
						question_id = f"{x}_"
						for i, line in enumerate(file):
							json_data = json.loads(line)
							json_data['question_id'] = question_id + str(i)
							ori_qa_data.append(json_data)
					qa_data.append(ori_qa_data)

				test_qa = qa_data[-1]
				test_table_data = table_data[-1]

				train_dev_table_data = {}
				train_dev_table_data.update(table_data[0])
				train_dev_table_data.update(table_data[1])
				train_table_data = deepcopy(train_dev_table_data)
				dev_table_data = deepcopy(train_dev_table_data)

				train_dev_qa_data = []
				train_dev_qa_data += qa_data[0]
				train_dev_qa_data += qa_data[1]

				# split train table ids into 4 folds
				four_folds = list(split_list(table_ids[0], 4))
				dev_table_ids = four_folds[self.config.split_id-1] if self.config.split_id != 0 else table_ids[1]

				train_qa = []
				dev_qa = []
				for qa in train_dev_qa_data:
					if qa['table_id'] in dev_table_ids:
						dev_qa.append(qa)
					else:
						train_qa.append(qa)

				if self.config.perturbation != 'original':
					assert self.config.split_id == 0

					# dev set from RobuT
					wikisql_qa_file = "robut_wikisql_qa.json"
					wikisql_table_file = "robut_wikisql_table.json"

					urls = _DATA_URL
					root_dir = os.path.join(dl_manager.download_and_extract(urls))

					qa_filepath = os.path.join(root_dir, "robut_data", wikisql_qa_file)
					qa_data = json.load(open(qa_filepath))
					table_filepath = os.path.join(root_dir, "robut_data", wikisql_table_file)
					table_data = json.load(open(table_filepath))

					for i in range(len(qa_data)):
						qa_data[i]['question_id'] = f"dev_{self.config.perturbation}_{i}"

					dev_qa = []
					for item in qa_data:
						if item['perturbation_type'] == self.config.perturbation:
							dev_qa.append(item)
					
					dev_table_data = table_data


				return [
						datasets.SplitGenerator(
								name=datasets.Split.TRAIN, 
								gen_kwargs={"split_key": "train", "qa_data": train_qa, "table_data": train_table_data}),
						datasets.SplitGenerator(
								name=datasets.Split.VALIDATION, 
								gen_kwargs={"split_key": "dev", "qa_data": dev_qa, "table_data": dev_table_data}),
						datasets.SplitGenerator(
								name=datasets.Split.TEST, 
								gen_kwargs={"split_key": "test", "qa_data": test_qa, "table_data": test_table_data}),
				]

		def _convert_to_human_readable(self, sel, agg, columns, conditions):
			"""Make SQL query string. Based on https://github.com/salesforce/WikiSQL/blob/c2ed4f9b22db1cc2721805d53e6e76e07e2ccbdc/lib/query.py#L10"""

			rep = "SELECT {agg} {sel} FROM table".format(
				agg=_AGG_OPS[agg], sel=columns[sel] if columns is not None else "col{}".format(sel)
			)

			if conditions:
				rep += " WHERE " + " AND ".join(["{} {} {}".format(columns[i], _COND_OPS[o], v) for i, o, v in conditions])
			return " ".join(rep.split())

		def _generate_examples(self, split_key, qa_data, table_data):
			
			for idx, example in enumerate(qa_data):
				
				# if example['question_id'] != 'dev_2423':
				# 	continue
				print(example)

				question = example["question"]
				table_content = table_data[example["table_id"]]

				if self.config.perturbation != 'original' and split_key=='dev':
					answers = example['answers']
				else:					
					tapas_table = _convert_table_types(table_content)
					answers = retrieve_wikisql_query_answer_tapas(tapas_table, example)
				# print(example)
				# print(answers)
				# assert 1==2
				
				perturbation_type = example["perturbation_type"] if "perturbation_type" in example else "original"

				db_engine = DBEngine('data/wikisql/train.db')
				if 'sql' in example:
					sql_raw = example["sql"]
					gold_query = Query.from_dict(sql_raw, ordered=False)
					gold_result = db_engine.execute_query(example['table_id'], gold_query, lower=True)
					print(gold_query, '\n', gold_result,'\n\n\n\n')
					assert 1==2
				else:
					sql = 'unk'


				yield idx, {
						"id": example["question_id"],
						"table_id": example["table_id"],
						"question": question.lower(),
						"answers": answers,
						"sql":sql,
						"table": {"header": table_content["header"], "rows": table_content["rows"]},
						"perturbation_type": perturbation_type,
						"split_key": split_key
				}
			

if __name__=='__main__':
		from datasets import load_dataset
		dataset = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/wikisql_robut.py", 
								split_id=0, ignore_verifications=True,
								# perturbation_type='row',
								# download_mode='force_redownload'
								)
		sample = dataset["train"][0]
		print(sample)