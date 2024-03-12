import pickle
import json

from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy
from metric.squall_evaluator import to_value_list
import pandas as pd
import json
import numpy as np
from classifier import preprocess_wikisql_df, extract_wikisql_features, load_and_predict, combine_csv, test_predict

from transformers import TapexTokenizer, T5Tokenizer
tableqa_tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large")
text_to_sql_tokenizer = T5Tokenizer.from_pretrained("t5-large")

model = 'RandomForest'
dataset = 'wikisql'
name = ''

# for pertubation in ['synonym', 'abbreviation', 'extend', 'add', 'word', 'sentence','combined']:
#     print('--',pertubation,'--')
#     tableqa_test = pd.read_csv(f"./predict/{dataset}/{dataset}_tableqa_dev0_{pertubation}.csv")
#     text_to_sql_test = pd.read_csv(f"./predict/{dataset}/{dataset}_text_to_sql_dev0_{pertubation}.csv")
#     df_dev = combine_csv(tableqa_test, text_to_sql_test, dataset)

#     df_dev = preprocess_wikisql_df(df_dev)
#     df_test_predict = test_predict(dataset, df_dev, tableqa_tokenizer, text_to_sql_tokenizer, model, name=name, qonly=False)
 
#     desired_order_selected = ['id', 'tbl', 'question', 'answer', 'acc_tableqa', 'ans_tableqa', 'acc_text_to_sql', 'ans_text_to_sql', 'query_pred', 'query_fuzzy', 'pred', 'labels', 'scores', 'oracle', 'diff']
#     remaining_columns = [col for col in df_dev.columns if col not in desired_order_selected]
#     df_dev = df_dev[desired_order_selected + remaining_columns]
#     df_dev.to_csv(f'./predict/{dataset}_classifier_dev0_{pertubation}.csv', na_rep='',index=False)
#     print('\n')
#     # assert 1==2


tableqa_test = pd.read_csv(f"./predict/{dataset}/{dataset}_tableqa_dev0.csv")
text_to_sql_test = pd.read_csv(f"./predict/{dataset}/{dataset}_text_to_sql_dev0.csv")
df_dev = combine_csv(tableqa_test, text_to_sql_test, dataset)

df_dev = preprocess_wikisql_df(df_dev)
df_test_predict = test_predict(dataset, df_dev, tableqa_tokenizer, text_to_sql_tokenizer, model, name=name, qonly=False)

desired_order_selected = ['id', 'tbl', 'question', 'answer', 'acc_tableqa', 'ans_tableqa', 'acc_text_to_sql', 'ans_text_to_sql', 'query_pred', 'query_fuzzy', 'pred', 'labels', 'scores', 'oracle', 'diff']
remaining_columns = [col for col in df_dev.columns if col not in desired_order_selected]
df_dev = df_dev[desired_order_selected + remaining_columns]
df_dev.to_csv(f'./predict/{dataset}_classifier_dev0.csv', na_rep='',index=False)
print('\n')