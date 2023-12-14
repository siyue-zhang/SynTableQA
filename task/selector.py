import json
import os
import datasets
import pandas as pd
import random

logger = datasets.logging.get_logger(__name__)
_dir_squall = "./data/squall"

class SelectorConfig(datasets.BuilderConfig):
    """BuilderConfig for Selector."""

    def __init__(self, dataset=None, **kwargs):
        """BuilderConfig for Selector.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SelectorConfig, self).__init__(**kwargs)
        self.dataset=dataset

class Selector(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = SelectorConfig

    def _info(self):
        return datasets.DatasetInfo(
            description='model selection',
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tbl": datasets.Value("string"),
                    "json_path": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "acc_tableqa": datasets.Value("int32"),
                    "acc_text_to_sql": datasets.Value("int32"),
                    "label": datasets.Value("int32"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        predict_dir = '/scratch/sz4651/Projects/SynTableQA/predict/'
        dataset = self.config.dataset
        assert dataset in ['squall', 'sede']

        tableqa_train = predict_dir + dataset+ '_tableqa_train.csv'
        tableqa_train = pd.read_csv(tableqa_train)

        tableqa_dev = predict_dir + dataset+ '_tableqa_dev.csv'
        tableqa_dev = pd.read_csv(tableqa_dev)

        tableqa = pd.concat([tableqa_train, tableqa_dev], axis=0)

        tbls = list(set(tableqa['tbl']))

        random.seed(42)
        random.shuffle(tbls)
        split_index = int(0.8 * len(tbls))
        train_tbl = tbls[:split_index]
        dev_tbl = tbls[split_index:]

        text_to_sql_train = predict_dir + dataset+ '_text_to_sql_train.csv'
        text_to_sql_train = pd.read_csv(text_to_sql_train)

        text_to_sql_dev = predict_dir + dataset+ '_text_to_sql_dev.csv'
        text_to_sql_dev = pd.read_csv(text_to_sql_dev)

        text_to_sql = pd.concat([text_to_sql_train, text_to_sql_dev], axis=0)

        train_tableqa = tableqa[tableqa['tbl'].isin(train_tbl)]
        train_text_to_sql = text_to_sql[text_to_sql['tbl'].isin(train_tbl)]

        dev_tableqa = tableqa[tableqa['tbl'].isin(dev_tbl)]
        dev_text_to_sql = text_to_sql[text_to_sql['tbl'].isin(dev_tbl)]

        test_tableqa = pd.read_csv(predict_dir + dataset + f'_tableqa_test.csv')
        test_text_to_sql = pd.read_csv(predict_dir + dataset + f'_text_to_sql_test.csv')

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"split_key": "train", 
                            "tableqa": train_tableqa,
                            "text_to_sql": train_text_to_sql}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={"split_key": "dev", 
                            "tableqa": dev_tableqa,
                            "text_to_sql": dev_text_to_sql}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={"split_key": "test", 
                            "tableqa": test_tableqa,
                            "text_to_sql": test_text_to_sql}),
        ]

    def clean(self, acc_tableqa):
        if isinstance(acc_tableqa,str):
            if acc_tableqa.lower()=='true':
                acc_tableqa=1
            else:
                acc_tableqa=0
        else:
            acc_tableqa=int(acc_tableqa)
        return acc_tableqa

    def _generate_examples(self, split_key, tableqa, text_to_sql):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from local trained csv")

        positives = []
        negatives = []
        for i, id in enumerate(list(tableqa['id'])):
            tableqa_row = tableqa.loc[tableqa['id']==id,:]
            text_to_sql_row = text_to_sql.loc[text_to_sql['id']==id,:]
            acc_tableqa = tableqa_row['acc'].values[0]
            acc_tableqa = self.clean(acc_tableqa)
            acc_text_to_sql = text_to_sql_row['acc'].values[0]
            acc_text_to_sql = self.clean(acc_text_to_sql)
            if acc_tableqa==1 and acc_text_to_sql==0:
                label=1
            else:
                label=0
            tbl = tableqa_row['tbl'].values[0]
            json_path = f"{_dir_squall}/tables/json/{tbl}.json"
            question = tableqa_row['question'].values[0]

            sample={
                "id": id,
                "tbl": tbl,
                "json_path": json_path,
                "question": question,
                "acc_tableqa": acc_tableqa,
                "acc_text_to_sql": acc_text_to_sql,
                "label": label
            }
            if label==0:
                negatives.append(sample)
            else:
                positives.append(sample)
        
        for j in range(len(negatives)//2):
            if j%2==0:
                out = random.choice(negatives)
            else:
                out = random.choice(positives)
            yield j, out

        
if __name__=='__main__':
    from datasets import load_dataset
    dataset = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/selector.py", dataset='squall')
    sample = dataset["train"][2]
    print(sample)