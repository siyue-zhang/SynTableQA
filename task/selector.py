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
                    "answer": datasets.Value("string"),
                    "acc_tableqa": datasets.Value("int32"),
                    "ans_tableqa": datasets.Value("string"),
                    "acc_text_to_sql": datasets.Value("int32"),
                    "ans_text_to_sql": datasets.Value("string"),
                    "query_fuzzy": datasets.Value("string"),
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
        if self.config.add_from_train:
            scope = ['train', 'dev', 'test']
        else:
            scope = ['dev', 'test']

        for split in scope:
            tableqa = predict_dir + dataset+ f'_tableqa_{split}.csv'
            tableqa = pd.read_csv(tableqa, index_col=0)
            if split=='train':
                text_to_sql = predict_dir + dataset+ f'_text_to_sql_{split}_aug.csv'
            else:
                text_to_sql = predict_dir + dataset+ f'_text_to_sql_{split}.csv'
            text_to_sql = pd.read_csv(text_to_sql, index_col=0)

            text_to_sql.rename(columns={'acc': 'acc_text_to_sql'}, inplace=True)
            text_to_sql['acc_tableqa'] = tableqa['acc']
            text_to_sql['pred_ans'] = tableqa['predictions']

            data[split] = text_to_sql
        
        split_path = './selector_splits.json'
        if os.path.exists(split_path):
            with open(split_path, 'r') as json_file:
                splits = json.load(json_file)
            selector_dev_tbls = splits['dev']
            selector_train_tbls = splits['train']
            print(f'load tbls from {split_path}.')
        else:
            tbl_dev = list(set(data['dev']['tbl'].to_list()))
            tbl_dev_shuffle = deepcopy(tbl_dev)
            random.shuffle(tbl_dev_shuffle)

            idx = int(len(tbl_dev_shuffle)*selector_dev_ratio)
            selector_dev_tbls = tbl_dev_shuffle[:idx]
            selector_train_tbls = tbl_dev_shuffle[idx:]
            print('squall dev set is split into train and dev for training selector.')

            to_save = {'dev': selector_dev_tbls, 'train': selector_train_tbls}
            with open(split_path, 'w') as f:
                json.dump(to_save, f)

        df_train = deepcopy(data['dev'])
        df_train = df_train[df_train['tbl'].isin(selector_train_tbls)]
        df_train['aug'] = 0
        df_train = df_train.reset_index().astype(str)

        if self.config.add_from_train:
            df_aug = data['train']
            possible_ans = list(set(df_aug['pred_ans'].to_list()))

            df_ori = df_aug[df_aug['aug']==0].reset_index()
            df_aug = df_aug[df_aug['aug']==1].reset_index()
            ### data augmentation for text_to_sql data

            ## for original samples, create negative answers by randomly sampling from other sample
            # negative to positive 1:4
            def is_number(s):
                try:
                    float(s)  # Use int() if you want to check for integers only
                    return True
                except ValueError:
                    return False

            def is_int(s):
                try:
                    int(s)  # Use int() if you want to check for integers only
                    return True
                except ValueError:
                    return False

            for n_row in range(df_ori.shape[0]):
                if random.random() < 0.2 and df_ori.loc[n_row, 'acc_text_to_sql']==1:
                    # augment negative sample
                    ans = deepcopy(df_ori.loc[n_row, 'answer'])
                    # print('before ', ans)
                    if is_number(ans):
                        if is_int(ans):
                            ans = str(int(ans) + random.randint(-10, 10))
                        else:
                            ans = str(float(ans) + random.randint(-10, 10))
                    while ans == df_ori.loc[n_row, 'answer']:
                        ans = random.choice(possible_ans)
                    # print('after ', ans, '\n')
                    df_ori.loc[n_row, 'queried_ans'] = ans
                    df_ori.loc[n_row, 'acc_text_to_sql'] = 0
                    df_ori.loc[n_row, 'aug'] = 1
            # df_ori = df_ori.iloc[:min(5000, df_ori.shape[0]),:]
            ## for aug samples, use aug acc and aug_ans rather than queried_ans
            df_aug['queried_ans'] = df_aug['aug_ans']
            df_train = pd.concat([df_ori, df_aug, df_train], ignore_index=True).reset_index().astype(str)
            print('added aug training samples.')
            
        df_dev = deepcopy(data['dev'])
        df_dev = df_dev[df_dev['tbl'].isin(selector_dev_tbls)]
        df_dev['aug'] = 0
        df_dev = df_dev.reset_index().astype(str)

        df_test = deepcopy(data['test'])
        df_test['aug'] = 0
        df_test = df_test.reset_index().astype(str)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"split_key": "train", 
                            "df": df_train}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={"split_key": "dev", 
                            "df": df_dev}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={"split_key": "test", 
                            "df": df_test}),
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
            aug = int(df.loc[i, 'aug'])

            acc_text_to_sql = int(df.loc[i, 'acc_text_to_sql'])
            ans_text_to_sql = df.loc[i, 'queried_ans']

            acc_tableqa = int(df.loc[i, 'acc_tableqa'])
            ans_tableqa = df.loc[i, 'pred_ans']

            claim = f'answer : {ans_text_to_sql}'
            query_fuzzy = df.loc[i, 'query_fuzzy']
            answer = df.loc[i, 'answer']
            
            yield i, {
                'id': id,
                'tbl': tbl,
                'json_path': json_path,
                'question': question,
                'answer': answer,
                'acc_text_to_sql': acc_text_to_sql,
                'ans_text_to_sql': ans_text_to_sql,
                'query_fuzzy': query_fuzzy,
                'acc_tableqa': acc_tableqa,
                'ans_tableqa': ans_tableqa,
                'label': acc_text_to_sql,
                'claim': claim,
                'aug': aug
            }

        
if __name__=='__main__':
    from datasets import load_dataset
    # dataset = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/selector.py", dataset='squall')
    dataset = load_dataset("/home/siyue/Projects/SynTableQA/task/selector.py", dataset='squall', add_from_train=True)
    for i in range(50):
        print(f'example {i}')
        print(dataset["train"][i], '\n')
