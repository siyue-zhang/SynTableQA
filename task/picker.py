import json
import os
import datasets
import pandas as pd
import random

logger = datasets.logging.get_logger(__name__)
_dir_squall = "./data/squall"

class SelectorConfig(datasets.BuilderConfig):
    """BuilderConfig for Selector."""

    def __init__(self, dataset=None, augmentation=False, **kwargs):
        """BuilderConfig for Selector.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(SelectorConfig, self).__init__(**kwargs)
        self.dataset=dataset
        self.augmentation=augmentation

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
                    "choices": datasets.Value("string"),
                    "aug": datasets.Value("int32"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        predict_dir = '/scratch/sz4651/Projects/SynTableQA/predict/'
        dataset = self.config.dataset
        assert dataset in ['squall', 'sede']

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"split_key": "train", 
                            "tableqa": predict_dir + dataset+ '_tableqa_train.csv',
                            "text_to_sql": predict_dir + dataset+ '_text_to_sql_train.csv'}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={"split_key": "dev", 
                            "tableqa": predict_dir + dataset+ '_tableqa_dev.csv',
                            "text_to_sql": predict_dir + dataset+ '_text_to_sql_dev.csv'}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={"split_key": "test", 
                            "tableqa": predict_dir + dataset+ '_tableqa_test.csv',
                            "text_to_sql": predict_dir + dataset+ '_text_to_sql_test.csv'}),
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

        tableqa = pd.read_csv(tableqa)
        text_to_sql = pd.read_csv(text_to_sql)
        diff_samples = []
        for index, row in tableqa.iterrows():
            acc_tableqa = self.clean(row['acc'])
            ans_tableqa = str(row['predictions']).strip()
            acc_text_to_sql = self.clean(text_to_sql.loc[index,'acc'])
            ans_text_to_sql = str(text_to_sql.loc[index,'queried_ans']).strip()

            id = row['id']
            tbl = row['tbl']
            json_path = f"{_dir_squall}/tables/json/{tbl}.json"
            question = row['question']
            
            if acc_tableqa != acc_text_to_sql and len(ans_tableqa)*len(ans_text_to_sql)>0:
                # careful the order
                label = [acc_tableqa, acc_text_to_sql].index(1)
                choices = f"\nAnswer Choice 0 : {ans_tableqa}\nAnswer Choice 1 : {ans_text_to_sql}\n"
                sample={
                    "id": id,
                    "tbl": tbl,
                    "json_path": json_path,
                    "question": question,
                    "acc_tableqa": acc_tableqa,
                    "ans_tableqa": ans_tableqa,
                    "acc_text_to_sql": acc_text_to_sql,
                    "ans_text_to_sql": ans_text_to_sql,
                    "label": label,
                    "choices": choices,
                    "aug":0
                }
                diff_samples.append(sample)
            elif self.config.augmentation:
                positives=[]
                positives.append(tableqa.loc[index,'answer'])
                if acc_tableqa:
                    positives.append(ans_tableqa)
                if acc_text_to_sql:
                    positives.append(ans_text_to_sql)
                scope = random.sample(tableqa['predictions'].to_list(),50)
                scope = [str(x) not in positives for x in scope]
                scope2 = random.sample(text_to_sql['queried_ans'].to_list(),50)
                scope2 = [str(x) for x in scope2 if x and x not in positives]
                negatives = scope+scope2
                label = random.choice([0,1])
                if label==0:
                    ans_tableqa = random.choice(positives)
                    acc_tableqa = 1
                    ans_text_to_sql = random.choice(negatives)
                else:
                    ans_text_to_sql = random.choice(positives)
                    ans_tableqa = random.choice(negatives)
                    acc_tableqa = 0
                acc_text_to_sql = 1-acc_tableqa

                choices = f"\nAnswer Choice 0 : {ans_tableqa}\nAnswer Choice 1 : {ans_text_to_sql}\n"
                sample={
                    "id": id,
                    "tbl": tbl,
                    "json_path": json_path,
                    "question": question,
                    "acc_tableqa": acc_tableqa,
                    "ans_tableqa": ans_tableqa,
                    "acc_text_to_sql": acc_text_to_sql,
                    "ans_text_to_sql": ans_text_to_sql,
                    "label": label,
                    "choices": choices,
                    "aug":1
                }
                diff_samples.append(sample)

        for j in range(len(diff_samples)):
            yield j, diff_samples[j]
            

        
if __name__=='__main__':
    from datasets import load_dataset
    dataset = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/picker.py", dataset='squall')
    sample = dataset["train"][2]
    print(sample)