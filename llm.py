from openai import OpenAI
from prompts import *
import pandas as pd
import copy


client = OpenAI(
    api_key='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'
)
model = 'gpt-3.5-turbo'
# model = 'gpt-4'
# model = ''


df = pd.read_csv("squall_selector_test_llm.csv")
response = []
for i, row in df.iterrows():
    print(i)
    if i > 10:
        break

    res = None
    context = row['inputs'][3:-4]
    ans_tableqa = str(row['ans_tableqa'])
    ans_text_to_sql = str(row['ans_text_to_sql'])
    acc_tableqa = int(row['acc_tableqa'])
    acc_text_to_sql = int(row['acc_text_to_sql'])

    if 'response' in row and str(row['response'])!='nan':
        res = row['response']
        if 'b is more likely to be correct' in res.lower():
            # B is tableqa, which is 1 in preds
            df.loc[i, 'preds'] = 1
            df.loc[i, 'score'] = row['acc_tableqa']
        elif 'a is more likely to be correct' in res.lower():
            # A is text_to_sql, which is 0 in preds
            df.loc[i, 'preds'] = 0
            df.loc[i, 'score'] = row['acc_text_to_sql']
        elif res.lower() in ['a', 'b']:
            pass
        else:
            print(f'response not clear - {i} - {df.loc[i, "id"]}')
        response.append(res)
        continue

    if ans_text_to_sql in ['', 'nan', 'na']:
        res = 'B'
        df.loc[i, 'score'] = row['acc_tableqa']
        df.loc[i, 'response'] = res
        response.append(res)
        continue

    if acc_tableqa==acc_text_to_sql:
        res = 'A'
        df.loc[i, 'score'] = row['acc_text_to_sql']
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
    # content = normal_sample_0 + '\n' + normal_sample_1 + '\n' + context
    content = context
    df.loc[i, 'content'] = content

    completion = client.chat.completions.create(
        model = model,
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content}
        ],
        temperature=0 # consistent output
    )
    res = completion.choices[0].message.content

    response.append(res)
    df.loc[i, 'response'] = res

    # print(context)
    print('\n---------------------\n')


df.to_csv("squall_selector_test_llm.csv", index=False)


