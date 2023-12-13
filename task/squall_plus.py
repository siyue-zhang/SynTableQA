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

    def __init__(self, **kwargs):
        """BuilderConfig for Squall.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SquallConfig, self).__init__(**kwargs)

class Squall(datasets.GeneratorBasedBuilder):
    """SQUALL: Lexical-level Supervised Table Question Answering Dataset."""

    BUILDER_CONFIGS = [
        SquallConfig(name = 'default'),
        SquallConfig(name = 'plus'),
    ]

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


    def _generate_examples(self, split_key, wtq_path):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", wtq_path)

        wtq_training = f"{wtq_path}/data/training.tsv"
        squall_train = f"{_dir_squall}/data/train-1.json"
        squall_dev = f"{_dir_squall}/data/dev-1.json"
        test = f"{_dir_squall}/data/wtq-test.json"
        test_label = f"{wtq_path}/data/pristine-unseen-tables.tsv"

        plus = self.config.name == 'plus'

        if split_key == 'test':
            path = test
            test_label = pd.read_table(test_label)
        elif split_key == 'train':
            path = squall_train
        else:
            path = squall_dev

        # load squall examples
        with open(path, encoding="utf-8") as f:
            examples = json.load(f)
        
        # get all table and question ids
        tbls = {ex['tbl'] for ex in examples}
        nts = {ex['nt'] for ex in examples}
        # load wtq examples if additional examples are needed in plus version
        if split_key != 'test' and plus:
            with open(wtq_training, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    # skip the header
                    if idx == 0:
                        continue
                    nt, question, tbl, answer_text = line.strip("\n").split("\t")
                    tbl = tbl.split('csv/')
                    tbl = tbl[1].replace('-','_') + tbl[2].split('.')[0]
                    if tbl in tbls and nt not in nts:
                        # print('\n', nt, question, tbl, answer_text, '\n')
                        examples.append({'nt':nt, 'tbl':tbl, 'nl': question, 'tgt': answer_text, 'src': 'wtq'})
  
        for i, sample in enumerate(examples):
            # print(sample)
            # get table id
            tbl = sample["tbl"]
            db_path = f"{_dir_squall}/tables/db/{tbl}.db"
            json_path = f"{_dir_squall}/tables/json/{tbl}.json"
            # get question and question id
            nt = sample["nt"]
            if isinstance(sample["nl"], list):
                if sample["nl"][-1] in '.?':
                    question = ' '.join(sample["nl"][:-1]) + sample["nl"][-1]
                else:
                    question = ' '.join(sample["nl"])
            else:
                question = sample["nl"]
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
    dataset = load_dataset("/home/siyue/Projects/SynTableQA/task/squall_plus.py", 'default')
    # dataset = load_dataset("/home/siyue/Projects/SynTableQA/task/squall_plus.py", 'plus')
    sample = dataset["train"][1110]
    print(sample)