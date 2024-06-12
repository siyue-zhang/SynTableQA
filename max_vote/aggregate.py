import json
import numpy as np
import pandas as pd
from collections import Counter
import math

model='tqa'

if model=='tts':
    name='squall_plus_text_to_sql_test1'
    save_name='aggregated_text_to_sql.csv'
    ans_name='queried_ans'
else:
    name='squall_plus_tableqa_test1'
    save_name='aggregated_tableqa.csv'
    ans_name='predictions'

dfs=[]

file = f'/scratch/sz4651/Projects/SynTableQA/predict/squall/{name}.csv'        
df = pd.read_csv(file)
gather = [ans_name, 'acc', 'log_probs_avg']
full_gather = ['id','tbl','question', 'answer'] + gather
df = df[full_gather ]
dfs.append(df)

for i in range(4):
    file = f'/scratch/sz4651/Projects/SynTableQA/predict/squall/{name}_noise{i}.csv'        
    df = pd.read_csv(file)
    gather = [ans_name, 'acc', 'log_probs_avg']
    df = df[gather]

    rename_dict = {
    ans_name: f'{ans_name}_{i}',
    'acc': f'acc_{i}',
    'log_probs_avg': f'log_probs_avg_{i}'
    }

    df.rename(columns=rename_dict, inplace=True)
    dfs.append(df)

dfs = pd.concat(dfs, axis=1)

final_ans, final_acc, final_prob = [],[],[]

for index, row in dfs.iterrows():

    keys=list(row.keys())

    ans_candidates=[row[k] for k in keys if ans_name in k]
    acc_candidates=[row[k] for k in keys if 'acc' in k]
    prob_candidates=[row[k] for k in keys if 'log_probs_avg' in k]

    answer_counts = Counter(ans_candidates)
    answer_counts = {key: value for key, value in answer_counts.items() if not isinstance(key, float) or not math.isnan(key)}

    if answer_counts=={}:
        first_index = 0
    else:
        most_frequent_answer = max(answer_counts, key=answer_counts.get)
        first_index = ans_candidates.index(most_frequent_answer)

    final_ans.append(ans_candidates[first_index])
    final_acc.append(acc_candidates[first_index])
    final_prob.append(prob_candidates[first_index])

dfs['final_ans']=final_ans
dfs['final_acc']=final_acc
dfs['final_prob']=final_prob

print(np.mean(dfs['final_acc']))
dfs.to_csv(save_name)



