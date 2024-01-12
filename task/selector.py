import json
import os
import datasets
import numpy as np
import pandas as pd
import random
from copy import deepcopy

logger = datasets.logging.get_logger(__name__)
_dir_squall = "./data/squall"

class SelectorConfig(datasets.BuilderConfig):
    """BuilderConfig for Selector."""

    def __init__(self, dataset=None, test_split=1, aug=False, **kwargs):
        """BuilderConfig for Selector.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SelectorConfig, self).__init__(**kwargs)
        self.dataset=dataset
        self.test_split=test_split
        self.aug=aug

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
                    "acc_text_to_sql": datasets.Value("int32"),
                    "ans_text_to_sql": datasets.Value("string"),
                    "query_fuzzy": datasets.Value("string"),
                    "acc_tableqa": datasets.Value("int32"),
                    "ans_tableqa": datasets.Value("string"),
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
        train_dev_ratio = 0.2
            
        splits = list(range(5))
        # splits = [1]
        
        dfs_dev = []
        for s in splits:
            tableqa_dev = pd.read_csv(f"./predict/squall_tableqa_dev{s}.csv")
            text_to_sql_dev = pd.read_csv(f"./predict/squall_text_to_sql_dev{s}.csv")
            df = tableqa_dev[['id','tbl','question','answer','src']]
            df['acc_tableqa'] = tableqa_dev['acc'].astype('int16')
            df['ans_tableqa'] = tableqa_dev['predictions']
            df['acc_text_to_sql'] = text_to_sql_dev['acc'].astype('int16')
            df['ans_text_to_sql'] = text_to_sql_dev['queried_ans']
            df['query_fuzzy'] = text_to_sql_dev['query_fuzzy']
            df = df[df['acc_tableqa'] != df['acc_text_to_sql']]
            df['label'] = [ 0 if int(x)==1 else 1 for x in df['acc_text_to_sql'].to_list()]
            dfs_dev.append(df)
        dfs_dev = pd.concat(dfs_dev, ignore_index=True).reset_index()
        tbls = list(set(dfs_dev['tbl'].to_list()))

        split_path = './selector_splits.json'
        if os.path.exists(split_path):
            with open(split_path, 'r') as json_file:
                splits = json.load(json_file)
            selector_dev_tbls = splits['dev']
            selector_train_tbls = splits['train']
            print(f'load tbls from {split_path}.')
        else:
            tbl_dev_shuffle = deepcopy(tbls)
            random.shuffle(tbl_dev_shuffle)

            idx = int(len(tbl_dev_shuffle)*train_dev_ratio)
            selector_dev_tbls = tbl_dev_shuffle[:idx]
            selector_train_tbls = tbl_dev_shuffle[idx:]
            print('squall dev set is split into train and dev for training selector.')

            to_save = {'dev': selector_dev_tbls, 'train': selector_train_tbls}
            with open(split_path, 'w') as f:
                json.dump(to_save, f)

        df_train = dfs_dev[dfs_dev['tbl'].isin(selector_train_tbls)]

        df_train = df_train.reset_index(drop=True) 
        # negative_map = {}
        dup_list = [
            'nt-10143', 'nt-12186', 'nt-10437', 'nt-73',
            'nt-10884', 'nt-12032', 
            'nt-7477', 'nt-13307', 'nt-2973', 'nt-13295',
            'nt-12171', 'nt-4918', 'nt-7104', 'nt-7998',
            'nt-1630', 'nt-2160', 'nt-5946',' nt-1630'
            ]
        duplicates = []
        for _ in range(20):
            duplicate = deepcopy(df_train)
            duplicate = duplicate[duplicate['id'].isin(dup_list)]
            # duplicate = duplicate[duplicate['acc_tableqa']==1]
        #     for i in range(df_train.shape[0]):
        #         acc_tableqa = duplicate.loc[i, 'acc_tableqa']
        #         acc_text_to_sql = duplicate.loc[i, 'acc_text_to_sql']
        #         tbl = duplicate.loc[i, 'tbl']
        #         if tbl not in negative_map:
        #             json_path = f"./data/squall/tables/json/{tbl}.json"
        #             with open(json_path, "r") as file:
        #                 data = json.load(file)
        #             cells = []
        #             for l in data['contents']:
        #                 for d in l:
        #                     tmp = d['data']
        #                     if isinstance(tmp[0],list):
        #                         tmp = ['|'.join([str(xx) for xx in x]) for x in tmp]
        #                     cells += tmp
        #             negative_map[tbl] = cells
        #         if acc_tableqa==0:
        #             neg = deepcopy(duplicate.loc[i, 'ans_text_to_sql'])
        #             while neg==duplicate.loc[i, 'ans_text_to_sql']:
        #                 neg = random.choice(negative_map[tbl])
        #             duplicate.loc[i, 'ans_tableqa'] = neg
        #         else:
        #             neg = deepcopy(duplicate.loc[i, 'ans_tableqa'])
        #             while neg==duplicate.loc[i, 'ans_tableqa']:
        #                 neg = random.choice(negative_map[tbl])
        #             duplicate.loc[i, 'ans_text_to_sql'] = neg
        #         duplicate['src']='squall_aug'
            duplicates.append(duplicate)
        df_train = pd.concat([df_train]+duplicates, ignore_index=True).reset_index()
        print('\n------')
        print(f'acc_tableqa: {sum(df_train["acc_tableqa"])}, acc_text_to_sql: {sum(df_train["acc_text_to_sql"])}\n')

        if self.config.aug:
            splits = list(range(5))
            dfs_aug = []
            for s in splits:
                tableqa_dev = pd.read_csv(f"./predict/squall_aug_tableqa_test{s}.csv")
                text_to_sql_dev = pd.read_csv(f"./predict/squall_aug_text_to_sql_test{s}.csv")
                df_aug = tableqa_dev[['id','tbl','question','answer','src']]
                df_aug['acc_tableqa'] = tableqa_dev['acc'].astype('int16')
                df_aug['ans_tableqa'] = tableqa_dev['predictions']
                df_aug['acc_text_to_sql'] = text_to_sql_dev['acc'].astype('int16')
                df_aug['ans_text_to_sql'] = text_to_sql_dev['queried_ans']
                df_aug['query_fuzzy'] = text_to_sql_dev['query_fuzzy']
                df_aug = df_aug[df_aug['acc_tableqa'] != df_aug['acc_text_to_sql']]
                df_aug['label'] = [ 0 if int(x)==1 else 1 for x in df_aug['acc_text_to_sql'].to_list()]

                dfs_aug.append(df_aug)
            df_train = pd.concat([df_train]+dfs_aug, ignore_index=True).reset_index().astype('str')
  
        df_dev = dfs_dev[dfs_dev['tbl'].isin(selector_dev_tbls)].reset_index().astype('str')

        s = self.config.test_split
        tableqa_test = pd.read_csv(f"./predict/squall_tableqa_test{s}.csv")
        text_to_sql_test = pd.read_csv(f"./predict/squall_text_to_sql_test{s}.csv")
        df = tableqa_test[['id','tbl','question','answer','src']]
        df['acc_tableqa'] = tableqa_test['acc'].astype('int16')
        df['ans_tableqa'] =  tableqa_test['predictions']
        df['acc_text_to_sql'] = text_to_sql_test['acc'].astype('int16')
        df['ans_text_to_sql'] = text_to_sql_test['queried_ans']
        df['query_fuzzy'] = text_to_sql_test['query_fuzzy']
        df = df.reset_index(drop=True)
        labels = []
        for i in range(df.shape[0]):
            acc_tableqa = df.loc[i, 'acc_tableqa']
            acc_text_to_sql = df.loc[i, 'acc_text_to_sql']
            if acc_tableqa != acc_text_to_sql:
                label = np.array([acc_text_to_sql, acc_tableqa]).argmax()
            else:
                label = 0
            labels.append(label)
        df['label'] = labels
        df_test = df.reset_index().astype('str')


        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"split_key": "train", 
                            "df": df_train.sample(frac=1, random_state=42).reset_index(drop=True)}),
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

        def truncate(ans):
            ans = str(ans)
            return ans[:min(len(ans), 50)]

        n_ex = df.shape[0]
        for i in range(n_ex):
            id = df.loc[i, 'id']
            tbl = df.loc[i, 'tbl']
            json_path = f"{_dir_squall}/tables/json/{tbl}.json"
            question = df.loc[i, 'question']

            acc_text_to_sql = int(df.loc[i, 'acc_text_to_sql'])
            ans_text_to_sql = df.loc[i, 'ans_text_to_sql']

            acc_tableqa = int(df.loc[i, 'acc_tableqa'])
            ans_tableqa = df.loc[i, 'ans_tableqa']

            claim = f'\nanswer A : {truncate(ans_text_to_sql)}\nanswer B : {truncate(ans_tableqa)}\n'
            query_fuzzy = df.loc[i, 'query_fuzzy']
            answer = df.loc[i, 'answer']
            label = df.loc[i, 'label']
            aug = 1 if 'aug' in df.loc[i, 'src'] else 0

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
    dataset = load_dataset("/home/siyue/Projects/SynTableQA/task/selector.py", 
                           dataset='squall', test_split=1, download_mode='force_redownload',
                           aug=False)
    for i in range(5):
        print(f'example {i}')
        print(dataset["train"][i], '\n')
