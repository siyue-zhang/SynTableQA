import pandas as pd
import json
import numpy as np

robut_wikisql ='data/robut_data/robut_wikisql_qa.json'
with open(robut_wikisql) as f:
    d = json.load(f)

tableqa_dev0 = 'wikisql/wikisql_tableqa_dev0.csv'
text_to_sql_dev0 = 'wikisql/wikisql_text_to_sql_dev0.csv'
# tableqa_dev0 = text_to_sql_dev0

pertubation = 'synonym'
# pertubation = 'abbreviation'
# pertubation = 'extend'
# pertubation = 'add'
# pertubation = 'word'
# pertubation = 'sentence'
# pertubation = 'combined'

tableqa_dev0_p = f'wikisql/pert/wikisql_tableqa_dev0_{pertubation}.csv'
text_to_sql_dev0_p = f'wikisql/pert/wikisql_text_to_sql_dev0_{pertubation}.csv'
# tableqa_dev0_p = text_to_sql_dev0_p

def load_df(path):
    df = pd.read_csv(path)
    return df

tableqa_dev0 = load_df(tableqa_dev0)
tableqa_dev0_p = load_df(tableqa_dev0_p)

acc_ori, acc_p = [], []
correct_both, correct_ori = 0, 0

for i, row in tableqa_dev0_p.iterrows():
    
    question = row['question']
    acc_tableqa = row['acc']

    if pertubation in ['sentence','word','combined']:
        ori_question = [x for x in d if x['perturbation_type']==pertubation and x['question']==question]
        ori_question = ori_question[0]['original_question'].lower()
    else:
        ori_question = question
    ori_row = tableqa_dev0[tableqa_dev0['question']==ori_question]
    ori_acc_tableqa = ori_row['acc'].values[0]

    acc_ori.append(ori_acc_tableqa)
    acc_p.append(acc_tableqa)

    if ori_acc_tableqa==acc_tableqa==1:
        correct_both+=1
        correct_ori+=1
    elif ori_acc_tableqa:
        correct_ori+=1

print(pertubation)
print('Original avg acc: ', np.mean(acc_ori))
print('Perturbed avg acc: ', np.mean(acc_p))
print(correct_both, correct_ori)
print('R-ACC: ', correct_both/correct_ori)