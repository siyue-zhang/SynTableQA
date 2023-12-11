# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""SQUALL: Lexical-level Supervised Table Question Answering Dataset."""


import json
import re
import os
import datasets
from datasets.tasks import QuestionAnsweringExtractive
import pandas as pd
from copy import deepcopy

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@inproceedings{Shi:Zhao:Boyd-Graber:Daume-III:Lee-2020,
	Title = {On the Potential of Lexico-logical Alignments for Semantic Parsing to {SQL} Queries},
	Author = {Tianze Shi and Chen Zhao and Jordan Boyd-Graber and Hal {Daum\'{e} III} and Lillian Lee},
	Booktitle = {Findings of EMNLP},
	Year = {2020},
}
"""

_DESCRIPTION = """\
To explore the utility of fine-grained, lexical-level supervision, authors \
introduce SQUALL, a dataset that enriches 11,276 WikiTableQuestions \ 
English-language questions with manually created SQL equivalents plus \ 
alignments between SQL and question fragments.
"""

_URL = "https://raw.githubusercontent.com/tzshi/squall/main/data/"
_URLS = {
    "squall": _URL + "squall.json",
    "wtq-test": _URL + "wtq-test.json",
    "dev-0": _URL +  "dev-0.ids",
    "dev-1": _URL +  "dev-1.ids",
    "dev-2": _URL +  "dev-2.ids",
    "dev-3": _URL +  "dev-3.ids",
    "dev-4": _URL +  "dev-4.ids",
    "test_label": "https://raw.githubusercontent.com/ppasupat/WikiTableQuestions/master/data/pristine-unseen-tables.tsv"
}

class Squall(datasets.GeneratorBasedBuilder):
    """SQUALL: Lexical-level Supervised Table Question Answering Dataset."""

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "query": datasets.Value("string"),
                    "query_tokens": datasets.features.Sequence(datasets.Value("string")),
                    "converted_query": datasets.Value("string"),
                    "id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "db_id": datasets.Value("string"),
                    "db_path": datasets.Value("string"),
                    "json_path": datasets.Value("string"),
                    "header": datasets.features.Sequence(datasets.Value("string")),
                    "label": datasets.Value("string"),

                    "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                    "db_column_names": datasets.features.Sequence(
                        {
                            "table_id": datasets.Value("int32"),
                            "column_name": datasets.Value("string"),
                            'ori_column_name': datasets.Value("string"),
                            'clean_column_name': datasets.Value("string"),
                        }
                    ),
                    "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                    "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                    "db_foreign_keys": datasets.features.Sequence(
                        {
                            "column_id": datasets.Value("int32"),
                            "other_column_id": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/tzshi/squall/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        urls_to_download = {
            "squall": _URLS["squall"],
            "wtq-test": _URLS["wtq-test"],
            "dev-0": _URLS["dev-0"],
            "dev-1": _URLS["dev-1"],
            "dev-2": _URLS["dev-2"],
            "dev-3": _URLS["dev-3"],
            "dev-4": _URLS["dev-4"],
            "test_label": _URLS["test_label"]
        }

        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"split_key": "train", "filepath": downloaded_files}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={"split_key": "dev", "filepath": downloaded_files}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={"split_key": "test", "filepath": downloaded_files}),
        ]

    def _generate_examples(self, split_key, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        squall_full = filepath["squall"]
        dev_ids = filepath["dev-1"]
        test = filepath["wtq-test"]
        test_label = filepath["test_label"]

        # transform the original squall data structure (list of things) to dict 
        def transform(sample, sample_key, keys):
            cols = {}
            n_col = len(sample[sample_key])
            for k in range(len(keys)):
                tmp = []
                for j in range(n_col):
                    tmp.append(sample[sample_key][j][k])
                cols[keys[k]] = tmp
            return cols

        columns_keys = ["raw_header", "tokenized_header", "column_suffixes", "column_dtype", "example"]
        sql_keys = ["sql_type", "value", "span_indices"]

        with open(squall_full, encoding="utf-8") as f:
            squall_full_data = json.load(f)
    
        NUM_MAPPING = {
            'half': 0.5,
            'one': 1,
            'two': 2,
            'three': 3,
            'four': 4,
            'five': 5,
            'six': 6,
            'seven': 7,
            'eight': 8,
            'nine': 9,
            'ten': 10,
            'eleven': 11,
            'twelve': 12,
            'twenty': 20,
            'thirty': 30,
            'once': 1,
            'twice': 2,
            'first': 1,
            'second': 2,
            'third': 3,
            'fourth': 4,
            'fifth': 5,
            'sixth': 6,
            'seventh': 7,
            'eighth': 8,
            'ninth': 9,
            'tenth': 10,
            'hundred': 100,
            'thousand': 1000,
            'million': 1000000,
            'jan': 1,
            'feb': 2,
            'mar': 3,
            'apr': 4,
            'may': 5,
            'jun': 6,
            'jul': 7,
            'aug': 8,
            'sep': 9,
            'oct': 10,
            'nov': 11,
            'dec': 12,
            'january': 1,
            'february': 2,
            'march': 3,
            'april': 4,
            'june': 6,
            'july': 7,
            'august': 8,
            'september': 9,
            'october': 10,
            'november': 11,
            'december': 12,
        }

        def parse_number(s):
            if s in NUM_MAPPING:
                return NUM_MAPPING[s]
            s = s.replace(',', '')
            # https://stackoverflow.com/questions/4289331/python-extract-numbers-from-a-string
            ret = re.findall(r"[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", s)
            if len(ret) > 0:
                return ret[0]
            return None

        for instance in squall_full_data:
            has_number = False
            numbers = []
            for x in instance["nl"]:
                numbers.append(parse_number(x))
                if numbers[-1] is not None:
                    has_number = True
            instance["numbers"] = numbers
            instance["has_number"] = has_number

        if split_key != 'test':
            with open(dev_ids) as f:
                dev_ids = json.load(f)
            if split_key == "train":
                squall = [x for x in squall_full_data if x["tbl"] not in dev_ids]
            else:
                squall = [x for x in squall_full_data if x["tbl"] in dev_ids]
        else:
            with open(test, encoding="utf-8") as f:
                squall = json.load(f)
            test_label = pd.read_table(test_label)

        for idx, sample in enumerate(squall):
            # transform columns
            cols = transform(sample, "columns", columns_keys)
            query_tokens = []
            if split_key == 'test':
                query = ''
                tgt = test_label[test_label["id"]==sample["nt"]]["targetValue"].tolist()[0]
                if isinstance(tgt, list):
                    tgt = [str(x) for x in tgt]
                    tgt = '|'.join(tgt)
                else:
                    tgt = str(tgt)
            else:
                # transform sql
                sqls = transform(sample, "sql", sql_keys)
                query = ' '.join(sqls['value'])
                tgt = sample['tgt']
                query_tokens = sqls['value']

            if sample["tbl"]=='204_56':
                tmp = deepcopy(query)
                query = query.replace('c1_year', 'c1_number')
                if query != tmp:
                    print("\ncolumn c1 in table 204_56 corrected!")
                    print('before: ', tmp)
                    print('after: ', query, '\n')


            raw_header = [x.replace('\n', ' ').strip().replace(' ', '_').lower() for x in cols['raw_header']]
            clean_header = [x.replace('\n', ' ').strip().lower() for x in cols['raw_header']]
            raw_header = ['unknown' if element == '' else element for element in raw_header]
            raw_header = [raw_header[i]+f'_{i+1}' for i in range(len(raw_header))]
            column_suffixes = cols['column_suffixes']

            db_column_names = {'table_id':[-1], 'column_name': ['*'], 'ori_column_name': ['*'], 'clean_column_name': []}
            db_column_types = []

            for j, h in enumerate(column_suffixes):
                db_column_names['table_id'].append(0)
                db_column_names['column_name'].append(raw_header[j])
                db_column_names['ori_column_name'].append(f'c{j+1}')
                db_column_names['clean_column_name'].append(clean_header[j])
                col_type = cols["column_dtype"][j]
                if 'number' in col_type:
                        col_type = 'number'
                else:
                    col_type = 'text'

                for suf in h:
                    db_column_names['column_name'].append(raw_header[j]+'_'+suf)
                    db_column_names['clean_column_name'].append(clean_header[j]+'_'+suf)
                    db_column_names['ori_column_name'].append(f'c{j+1}_{suf}')
                    db_column_types.append(col_type)
            
            db_primary_keys = {'column_id': []}
            db_foreign_keys = {'column_id': [], 'other_column_id': []}

            converted_query = deepcopy(query)
            if split_key != 'test':
                for k, ori_col in enumerate(db_column_names['ori_column_name']):
                    converted_query = converted_query.replace(ori_col, db_column_names['column_name'][k])


            # if sample["nt"]=='nt-5967':
            yield idx, {
                "query": query,
                "query_tokens": query_tokens,
                "converted_query": converted_query,
                "id": sample["nt"],
                "header": raw_header,
                "question": ' '.join(sample["nl"]),
                "label": tgt,
                "db_id": sample["tbl"],
                "db_path": f"./third_party/squall/tables/db/{sample['tbl']}.db",
                "json_path": f"./third_party/squall/tables/json/{sample['tbl']}.json",
                "db_table_names": ['w'],
                "db_column_names": db_column_names,
                "db_column_types": db_column_types,
                "db_primary_keys": db_primary_keys,
                "db_foreign_keys": db_foreign_keys,
            }

    