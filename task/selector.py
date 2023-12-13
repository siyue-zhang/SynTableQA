import json
import os
import datasets
import pandas as pd

logger = datasets.logging.get_logger(__name__)

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
                    "question": datasets.Value("string"),
                    "acc_tableqa": datasets.Value("int32"),
                    "acc_text_to_sql": datasets.Value("int32"),
                    "label": datasets.Value("int32"),
                }
            ),
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        predict_dir = '/home/siyue/Projects/SynTableQA/predict/'
        dataset = self.config.dataset
        assert dataset in ['squall', 'sede']

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, 
                gen_kwargs={"split_key": "train", 
                            "tableqa": predict_dir + dataset + f'_tableqa_train.csv',
                            "text_to_sql": predict_dir + dataset + f'_text_to_sql_train.csv'}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, 
                gen_kwargs={"split_key": "dev", 
                            "tableqa": predict_dir + dataset + f'_tableqa_dev.csv',
                            "text_to_sql": predict_dir + dataset + f'_text_to_sql_dev.csv'}),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, 
                gen_kwargs={"split_key": "test", 
                            "tableqa": predict_dir + dataset + f'_tableqa_test.csv',
                            "text_to_sql": predict_dir + dataset + f'_text_to_sql_test.csv'}),
        ]


    def _generate_examples(self, split_key, tableqa, text_to_sql):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from local trained csv")
        
        tableqa = pd.read_csv(tableqa)
        text_to_sql = pd.read_csv(text_to_sql)
        all_ids = list(set(list(tableqa['id']) + list(text_to_sql['id'])))
        for i, id in enumerate(all_ids):
            
            if id in tableqa['id']:
                row = tableqa.loc[tableqa['id'] == id, :]
            else:
                row = text_to_sql.loc[tableqa['id'] == id, :]
            tbl = row['tbl'].values[0]
            question = row['question'].values[0]

            acc_tableqa = tableqa.loc[tableqa['id'] == id, 'acc'].values[0]
            if isinstance(acc_tableqa, str):
                if acc_tableqa.strip().lower()=='true':
                    acc_tableqa=1
                else:
                    acc_tableqa=0
            acc_text_to_sql = text_to_sql.loc[text_to_sql['id'] == id, 'acc'].values[0]
            if isinstance(acc_text_to_sql, str):
                if acc_text_to_sql.strip().lower()=='true':
                    acc_text_to_sql=1
                else:
                    acc_text_to_sql=0
            
            # label 0 means selecting text_to_sql model
            if acc_text_to_sql==1:
                label=0
            else:
                label=1

            yield i, {
                "id": id,
                "tbl": tbl,
                "question": question,
                "acc_tableqa": acc_tableqa,
                "acc_text_to_sql": acc_text_to_sql,
                "label": label
            }

        
if __name__=='__main__':
    from datasets import load_dataset
    dataset = load_dataset("/home/siyue/Projects/SynTableQA/task/selector.py", dataset='squall')
    sample = dataset["train"][2]
    print(sample)