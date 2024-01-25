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
import os
import datasets
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

_dir_squall = "./data/squall"
_URL_wtq = "https://github.com/ppasupat/WikiTableQuestions/archive/refs/heads/master.zip"


class SquallConfig(datasets.BuilderConfig):
    """BuilderConfig for Squall."""

    def __init__(self, plus, split_id, **kwargs):
        """BuilderConfig for Squall.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquallConfig, self).__init__(**kwargs)
        self.split_id = split_id
        self.plus = plus

class Squall(datasets.GeneratorBasedBuilder):
    """SQUALL: Lexical-level Supervised Table Question Answering Dataset."""

    BUILDER_CONFIG_CLASS = SquallConfig

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "nt": datasets.Value("string"),
                    "tbl": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "query": datasets.Value("string"),
                    "answer_text": datasets.Value("string"),
                    "db_path": datasets.Value("string"),
                    "json_path": datasets.Value("string"),
                    "src":datasets.Value("string"),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/tzshi/squall/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        wtq_data_dir = os.path.join(dl_manager.download_and_extract(_URL_wtq), 'WikiTableQuestions-master')

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"split_key": "train", "wtq_path": wtq_data_dir}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={"split_key": "dev", "wtq_path": wtq_data_dir}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={"split_key": "test", "wtq_path": wtq_data_dir}),
        ]

    def get_tbls_nts(self, path):
        with open(path, encoding="utf-8") as f:
            examples = json.load(f)
        tbls = {ex['tbl'] for ex in examples}
        nts = {ex['nt'] for ex in examples}
        return list(tbls), list(nts)

    def split_list(self, lst, n):
        # Calculate the number of items in each split
        avg = len(lst) // n
        remainder = len(lst) % n

        # Initialize the starting index for each split
        start = 0

        # Iterate over each split
        for i in range(n):
            # Calculate the end index for the current split
            end = start + avg + (1 if i < remainder else 0)

            # Yield the current split
            yield lst[start:end]

            # Update the starting index for the next split
            start = end    

    def _generate_examples(self, split_key, wtq_path):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", wtq_path)
        # load raw data from wtq and squall
        split_id = self.config.split_id
        wtq_training = f"{wtq_path}/data/training.tsv"
        squall_train = f"{_dir_squall}/data/train-{split_id}.json"
        squall_dev = f"{_dir_squall}/data/dev-{split_id}.json"
        test = f"{_dir_squall}/data/wtq-test.json"
        test_label = f"{wtq_path}/data/pristine-unseen-tables.tsv"
        # if load squall or squall_plus version
        plus = self.config.plus == 'plus'
        if plus:
            # get all SQUALL table ids
            train_tbl, _ = self.get_tbls_nts(squall_train)
            dev_tbl, _ = self.get_tbls_nts(squall_dev)
            squall_tbls = train_tbl + dev_tbl

        # get squall examples
        if split_key == 'test':
            path = test
            test_label = pd.read_table(test_label)
        elif split_key == 'train':
            path = squall_train
        else:
            path = squall_dev

        # load examples from SQUALL splits
        with open(path, encoding="utf-8") as f:
            examples = json.load(f)
        
        # get all table and question ids
        tbls = {ex['tbl'] for ex in examples}
        nts = {ex['nt'] for ex in examples}

        # load wtq examples if additional examples are needed in plus version
        if split_key != 'test' and plus:
            new_examples = []
            with open(wtq_training, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    # skip the header
                    if idx == 0:
                        continue
                    nt, question, tbl, answer_text = line.strip("\n").split("\t")
                    tbl = tbl.split('csv/')
                    tbl = tbl[1].replace('-','_') + tbl[2].split('.')[0]
                    # table exits in SQUALL but question doesn't exist in SQUALL
                    if tbl in tbls and nt not in nts:
                        examples.append({'nt':nt, 'tbl':tbl, 'nl': question, 'tgt': answer_text, 'src': 'wtq'})
                    # table doesn't exist in SQUALL
                    if tbl not in squall_tbls:
                        new_examples.append({'nt':nt, 'tbl':tbl, 'nl': question, 'tgt': answer_text, 'src': 'wtq'})
            
            # wtq example's table exist in SQUALL
            new_tbls = sorted(list(set([x['tbl'] for x in new_examples])))
            # split tables into 5 folds
            result_splits = list(self.split_list(new_tbls, 5))
            # print(result_splits)
            
            new_tbls_dev = result_splits[split_id]
            new_tbls_train = [x for x in new_tbls if x not in new_tbls_dev]
            # put some examples into train set, and the rest into dev set by table id
            if split_key == 'train':
                examples += [x for x in new_examples if x['tbl'] in new_tbls_train]
            else:
                examples += [x for x in new_examples if x['tbl'] in new_tbls_dev]

        # generate each example
        for i, sample in enumerate(examples):
            tbl = sample["tbl"]
            db_path = f"{_dir_squall}/tables/db/{tbl}.db"
            json_path = f"{_dir_squall}/tables/json/{tbl}.json"
            nt = sample["nt"]
            if isinstance(sample["nl"], list):
                if sample["nl"][-1] in '.?':
                    question = ' '.join(sample["nl"][:-1]) + sample["nl"][-1]
                else:
                    question = ' '.join(sample["nl"])
            else:
                question = sample["nl"]

            if 'question' in sample:
                question = sample['question']

            # get sql query and answer text
            if split_key == 'test':
                query = 'unk'
                answer = test_label[test_label["id"]==sample["nt"]]["targetValue"].tolist()[0]
                if isinstance(answer, list):
                    answer_text = [str(x) for x in answer]
                    answer_text = '|'.join(answer_text)
                else:
                    answer_text = str(answer)
            else:
                if 'sql' in sample:
                    query = ' '.join([tok[1] for tok in sample['sql']])
                else:
                    query = 'unk'
                answer_text = sample['tgt']
            
            if 'src' in sample:
                src = sample['src']
            else:
                src = 'squall'

            yield i, {
                "nt": nt,
                "tbl": tbl,
                "question": question,
                "query": query,
                "answer_text": answer_text,
                "db_path": db_path,
                "json_path": json_path,
                "src": src
            }

        


if __name__=='__main__':
    from datasets import load_dataset
    # dataset = load_dataset("/home/siyue/Projects/SynTableQA/task/squall_plus.py", plus='default', split_id=0)
    dataset = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/squall_plus.py", 
                           plus='plus', 
                           split_id=1)
    sample = dataset["test"][7]
    print(sample)