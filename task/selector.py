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
            train_aug = predict_dir + dataset+ f'_train_aug.xlsx'
            df_tmp = pd.read_excel(train_aug, index_col=0).reset_index(drop=True)
            df_tmp['id'] = df_tmp['id'].astype('str')
            df_tmp = df_tmp[df_tmp['id']!='nan']
            for x in ['acc_tableqa','acc_text_to_sql', 'aug']:
                df_tmp[x] = df_tmp[x].astype('int16')
            data['train'] = df_tmp.astype('str')

        scope = ['dev', 'test']
        for split in scope:
            tableqa = predict_dir + dataset+ f'_tableqa_{split}.csv'
            tableqa = pd.read_csv(tableqa, index_col=0)

            text_to_sql = predict_dir + dataset+ f'_text_to_sql_{split}.csv'
            text_to_sql = pd.read_csv(text_to_sql, index_col=0)

            text_to_sql.rename(columns={'acc': 'acc_text_to_sql', 'queried_ans': 'ans_text_to_sql'}, inplace=True)
            text_to_sql['acc_tableqa'] = tableqa['acc']
            text_to_sql['ans_tableqa'] = tableqa['predictions']

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
        df_train = df_train.reset_index().astype('str')

        if self.config.add_from_train:
            df_aug = data['train']
            df_aug['aug'] = df_aug['aug'].astype('int16')
            possible_ans = list(set(df_aug['ans_tableqa'].to_list()))

            df_ori = df_aug[df_aug['aug']==0].reset_index()
            df_aug = df_aug[df_aug['aug']==1].reset_index()
            ### data augmentation for text_to_sql data
            ## for original samples, create negative answers by randomly sampling from other sample
            for n_row in range(df_ori.shape[0]):
                ans_tableqa = df_ori.loc[n_row, 'ans_tableqa']
                acc_tableqa = int(df_ori.loc[n_row, 'acc_tableqa'])

                ans_text_to_sql = df_ori.loc[n_row, 'ans_text_to_sql']
                acc_text_to_sql = int(df_ori.loc[n_row, 'acc_text_to_sql'])

                if acc_tableqa!=acc_text_to_sql:
                    continue

                positives = [df_ori.loc[n_row, 'answer']]
                if acc_tableqa==1:
                    positives.append(ans_tableqa)
                if acc_text_to_sql==1:
                    positives.append(ans_text_to_sql)
                positives = list(set(positives))

                correct = random.choice(positives)
                wrong = deepcopy(correct)
                while wrong in positives:
                    wrong = random.choice(possible_ans)

                if random.random()<0.5:
                    acc_tableqa = 1
                    ans_tableqa = correct
                    acc_text_to_sql = 0
                    ans_text_to_sql = wrong
                    label = 1
                else:
                    acc_tableqa = 0
                    ans_tableqa = wrong
                    acc_text_to_sql = 1
                    ans_text_to_sql = correct
                    label = 0

                df_ori.loc[n_row, 'acc_tableqa'] = acc_tableqa
                df_ori.loc[n_row, 'ans_tableqa'] = ans_tableqa
                df_ori.loc[n_row, 'acc_text_to_sql'] = acc_text_to_sql
                df_ori.loc[n_row, 'ans_text_to_sql'] = ans_text_to_sql
                df_ori.loc[n_row, 'label'] = label 
                df_ori.loc[n_row, 'aug'] = 1

            sample_size = min(2000, len(df_ori))
            df_ori = df_ori.sample(n=sample_size, random_state=42)  # You can change the random_state for different shuffling
            df_ori = df_ori.sample(frac=1, random_state=42).reset_index(drop=True)

            df_train = pd.concat([df_train], ignore_index=True).reset_index().astype('str')
            df_train = df_train.sample(frac=1, random_state=42)
            print('added aug training samples.')
            
        df_dev = deepcopy(data['dev'])
        df_dev = df_dev[df_dev['tbl'].isin(selector_dev_tbls)]
        df_dev['aug'] = 0
        df_dev = df_dev.reset_index().astype('str')

        df_test = deepcopy(data['test'])
        df_test['aug'] = 0
        df_test = df_test.reset_index().astype('str')

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
            ans_text_to_sql = df.loc[i, 'ans_text_to_sql']

            acc_tableqa = int(df.loc[i, 'acc_tableqa'])
            ans_tableqa = df.loc[i, 'ans_tableqa']

            def truncate(ans):
                return ans[:min(len(ans), 50)]
            
            claim = f'\nanswer A : {truncate(ans_text_to_sql)}\nanswer B : {truncate(ans_tableqa)}\n'
            query_fuzzy = df.loc[i, 'query_fuzzy']
            answer = df.loc[i, 'answer']
            # both correct or wrong, label = 0
            if acc_tableqa==acc_text_to_sql:
                if split_key=='train':
                    continue
                else:
                    label = 0
            else:
                label = 1 if acc_tableqa else 0

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
                'label': label,
                'claim': claim,
                'aug': aug
            }

        
if __name__=='__main__':
    from datasets import load_dataset
    # dataset = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/selector.py", dataset='squall')
    dataset = load_dataset("/home/siyue/Projects/SynTableQA/task/selector.py", dataset='squall', add_from_train=True, download_mode='force_redownload')
    for i in range(5):
        print(f'example {i}')
        print(dataset["train"][i], '\n')
