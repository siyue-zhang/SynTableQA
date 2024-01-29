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
"""Splited Spider-Syn: Towards Robustness of Text-to-SQL Models against Synonym Substitution"""


import json
import os
import sys
sys.path.append('./')
from utils.get_tables import dump_db_json_schema
from utils.misc import execute_query
import datasets


logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{yu2018spider,
  title={Spider: A large-scale human-labeled dataset for complex and cross-domain semantic parsing and text-to-sql task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  journal={arXiv preprint arXiv:1809.08887},
  year={2018}
}
"""

_DESCRIPTION = """\
Spider is a large-scale complex and cross-domain semantic parsing and text-toSQL dataset annotated by 11 college students
"""

class SpiderConfig(datasets.BuilderConfig):
    """BuilderConfig for Squall."""

    def __init__(self, split_id, syn=True, **kwargs):
        """BuilderConfig for Squall.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SpiderConfig, self).__init__(**kwargs)
        self.split_id = split_id
        self.syn = syn

class Spider(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIG_CLASS = SpiderConfig

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self):
        features = datasets.Features(
            {
                "query": datasets.Value("string"),
                "question": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
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
                "answer": datasets.Value("string"),
                "src":datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            citation=_CITATION,
        )

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

    def _split_generators(self, dl_manager):

        spider_train_path = "data/spider/train_spider.json"
        spider_syn_train_path = "data/Spider-Syn/Spider-Syn/train_spider.json"
        db_path = "data/spider/database"

        with open(spider_train_path, encoding="utf-8") as f:
            spider = json.load(f)
        # replace question in spider by spider-syn    
        if self.config.syn:
            with open(spider_syn_train_path, encoding="utf-8") as f:
                spider_syn = json.load(f)
            for i in range(len(spider)):
                spider[i]["question"]=spider_syn[i]["SpiderSynQuestion"]

        dbs = list(set([x['db_id'] for x in spider]))
        split_path = './task/spider_splits.json'
        if os.path.exists(split_path):
            with open(split_path, 'r') as json_file:
                db_splits = json.load(json_file)
        else:
            db_splits = list(self.split_list(dbs, 5))
            with open(split_path, 'w') as f:
                json.dump(db_splits, f)
            
        dbs_dev = db_splits[self.config.split_id]
        dev_set = [x for x in spider if x['db_id'] in dbs_dev]
        train_set = [x for x in spider if x['db_id'] not in dbs_dev]

        spider_dev_path = "data/spider/dev.json"
        spider_syn_dev_path = "data/Spider-Syn/Spider-Syn/dev.json"

        with open(spider_dev_path, encoding="utf-8") as f:
            spider = json.load(f)
        # replace question in spider by spider-syn    
        if self.config.syn:
            with open(spider_syn_dev_path, encoding="utf-8") as f:
                spider_syn = json.load(f)
            for i in range(len(spider)):
                spider[i]["question"]=spider_syn[i]["SpiderSynQuestion"]
        test_set = spider

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"split_key": "train", "data": train_set, "db_path": db_path}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={"split_key": "dev", "data": dev_set, "db_path": db_path}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={"split_key": "test", "data": test_set, "db_path": db_path}),
        ]


    def _generate_examples(self, split_key, data, db_path):
        """This function returns the examples in the raw (text) form."""

        for idx, sample in enumerate(data):
            db_id = sample["db_id"]
            if db_id not in self.schema_cache:
                self.schema_cache[db_id] = dump_db_json_schema(
                    db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
                )
            schema = self.schema_cache[db_id]

            path = db_path + "/" + db_id + "/" + db_id + ".sqlite"
            query = sample["query"]

            # skip all inexecutable query
            try:
                answer = execute_query(path, query)
            except Exception as e:
                continue

            # skip all examples whose query result is none: empty database like music_2, wrong query
            if list(set([a.lower() for a in answer])) in [['none'], [''], []]:
                continue
            
            # skip all examples with a lot of answers
            if len(answer)>20:
                continue
            
            answer = '|'.join(answer)

            if self.config.syn:
                src = 'syn'
            else:
                src = 'spider'

            yield idx, {
                "query": sample["query"],
                "question": sample["question"],
                "db_id": db_id,
                "db_path": db_path,
                "db_table_names": schema["table_names_original"],
                "db_column_names": [
                    {"table_id": table_id, "column_name": column_name}
                    for table_id, column_name in schema["column_names_original"]
                ],
                "db_column_types": schema["column_types"],
                "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                "db_foreign_keys": [
                    {"column_id": column_id, "other_column_id": other_column_id}
                    for column_id, other_column_id in schema["foreign_keys"]
                ],
                "answer": answer,
                "src": src
            }


if __name__=='__main__':
    from datasets import load_dataset
    dataset = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/spider_syn.py", 
                           syn=True, 
                           split_id=1,
                           download_mode='force_redownload',ignore_verifications=True,)
    sample = dataset["test"][87]
    print(sample)