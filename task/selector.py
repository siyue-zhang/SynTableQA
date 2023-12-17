import json
import os
import datasets
import pandas as pd
import random
from copy import deepcopy

logger = datasets.logging.get_logger(__name__)
_dir_squall = "./data/squall"

class SelectorConfig(datasets.BuilderConfig):
    """BuilderConfig for Selector."""

    def __init__(self, dataset=None, add_from_train=False, **kwargs):
        """BuilderConfig for Selector.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SelectorConfig, self).__init__(**kwargs)
        self.dataset=dataset
        self.add_from_train=add_from_train

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
                    "ans_tableqa": datasets.Value("string"),
                    "acc_text_to_sql": datasets.Value("int32"),
                    "ans_text_to_sql": datasets.Value("string"),
                    "label": datasets.Value("int32"),
                    "claim": datasets.Value("string"),
                    "aug": datasets.Value("int32"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        predict_dir = f'./predict/'
        dataset = self.config.dataset
        assert dataset in ['squall', 'sede']
        selector_dev_ratio = 0.2

        data = {}
        for split in ['train', 'dev', 'test']:
            tableqa = predict_dir + dataset+ f'_tableqa_{split}.csv'
            tableqa = pd.read_csv(tableqa, index_col=0)
            text_to_sql = predict_dir + dataset+ f'_text_to_sql_{split}.csv'
            text_to_sql = pd.read_csv(text_to_sql, index_col=0)

            text_to_sql.rename(columns={'acc': 'acc_text_to_sql'}, inplace=True)
            text_to_sql['acc_tableqa'] = tableqa['acc']
            text_to_sql['pred_ans'] = tableqa['predictions']

            data[split] = text_to_sql
        
        tbl_dev = list(set(data['dev']['tbl'].to_list()))
        tbl_dev_shuffle = deepcopy(tbl_dev)
        random.shuffle(tbl_dev_shuffle)

        idx = int(len(tbl_dev_shuffle)*selector_dev_ratio)
        selector_dev_tbls = tbl_dev_shuffle[:idx]
        selector_train_tbls = tbl_dev_shuffle[idx:]
        print('squall dev set is split into train and dev for training selector.')

        df_train = deepcopy(data['dev'])
        df_train = df_train[df_train['tbl'].isin(selector_train_tbls)]

        if self.config.add_from_train:
            negatives_in_train = deepcopy(data['train'])
            negatives_in_train = negatives_in_train[negatives_in_train['acc_text_to_sql']==0]
            n_negative = negatives_in_train.shape[0]
            print(f'{n_negative} samples in train predictions are negatives (acc_text_to_sql=0)')
            positives_in_train = data['train'].sample(n=n_negative)
            df_train =  pd.concat([df_train, negatives_in_train, positives_in_train], ignore_index=True)
            print('added some training samples.')
            
        df_dev = deepcopy(data['dev'])
        df_dev = df_dev[df_dev['tbl'].isin(selector_dev_tbls)]

        df_test = deepcopy(data['test'])

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"split_key": "train", 
                            "df": df_train.reset_index()}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={"split_key": "dev", 
                            "df": df_dev.reset_index()}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={"split_key": "test", 
                            "df": df_test.reset_index()}),
        ]

    def _generate_examples(self, split_key, df):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from local trained csv")

        n_ex = df.shape[0]
        for i in range(n_ex):
            id = df.loc[i, 'id']
            tbl = df.loc[i, 'tbl']
            json_path = f"{_dir_squall}/tables/json/{tbl}.json"
            question = df.loc[i, 'question']

            acc_text_to_sql = df.loc[i, 'acc_text_to_sql']
            ans_text_to_sql = df.loc[i, 'queried_ans']

            acc_tableqa = df.loc[i, 'acc_tableqa']
            ans_tableqa = df.loc[i, 'pred_ans']

            claim = f'answer : {ans_text_to_sql}'
            yield i, {
                'id': id,
                'tbl': tbl,
                'json_path': json_path,
                'question': question,
                'acc_text_to_sql': acc_text_to_sql,
                'ans_text_to_sql': ans_text_to_sql,
                'acc_tableqa': acc_tableqa,
                'ans_tableqa': ans_tableqa,
                'label': acc_text_to_sql,
                'claim': claim,
                'aug': 0
            }

        
if __name__=='__main__':
    from datasets import load_dataset
    dataset = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/selector.py", dataset='squall')
    sample = dataset["train"][2]
    print(sample)