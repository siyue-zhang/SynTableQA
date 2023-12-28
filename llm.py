from openai import OpenAI
from prompts import *
import json
import pandas as pd
import copy

df = pd.read_csv("squall_selector_test_llm.csv")
response = []
for i, row in df.iterrows():
    if i > 1000:
        break

    res = None
    context = row['inputs'][3:-4]
    ans_tableqa = str(row['ans_tableqa'])
    ans_text_to_sql = str(row['ans_text_to_sql'])
    acc_tableqa = int(row['acc_tableqa'])
    acc_text_to_sql = int(row['acc_text_to_sql'])

    if 'response' in row and str(row['response'])!='nan':
        res = row['response']
        response.append(res)
        continue

    if acc_tableqa==acc_text_to_sql or ans_text_to_sql in ['', 'nan', 'na']:
        res = 'B'
        df.loc[i, 'score'] = row['acc_tableqa']
        df.loc[i, 'response'] = res
        response.append(res)
        continue

    new_ans_tableqa = ans_tableqa.replace('|', ', ')
    new_ans_text_to_sql = ans_text_to_sql.replace('|', ', ')

    tmp = copy.deepcopy(context)
    context = context.replace(
        f'answer A : {ans_text_to_sql}\nanswer B : {ans_tableqa}',
        f'answer A : {new_ans_text_to_sql}\nanswer B : {new_ans_tableqa}')

    context = 'Q:' + context + '\n\nA:'
    content = normal_sample_0 + '\n' + normal_sample_1 + '\n' + context

    response.append(res)
    df.loc[i, 'response'] = res

    print(context,'\n---------------------\n')


df.to_csv("squall_selector_test_llm.csv", index=False)

# client = OpenAI(
#     api_key='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'
# )




# completion = client.chat.completions.create(
#   model="gpt-4",
#   messages=[
#     {"role": "system", "content": "You are an assistant to select the correct answer."},
#     {"role": "user", "content": content}
#   ]
# )

# print(completion.choices[0].message)