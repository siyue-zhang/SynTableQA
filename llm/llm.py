from openai import OpenAI
from prompts import *
import pandas as pd

client = OpenAI(
    api_key='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'
)
model = 'gpt-3.5-turbo'
# model = 'gpt-4'
# model = ''
file_path = "llm/squall_classifier_test1.csv"

df = pd.read_csv(file_path)

df['truncated_tableqa'] = pd.to_numeric(df['truncated_tableqa'])
df = df[df['truncated_tableqa']==1].reset_index(drop=True)

pre_response = []
response = []
for i, row in df.iterrows():
    print('\n----row: ', i, '-----')
    if i > 10:
        break

    context = row['input_tokens'][3:-4]
    ans_tableqa = str(row['ans_tableqa'])
    ans_text_to_sql = str(row['ans_text_to_sql'])
    acc_tableqa = int(row['acc_tableqa'])
    acc_text_to_sql = int(row['acc_text_to_sql'])

    if 'response' in row and str(row['response'])!='nan':
        res = row['response'].strip().lower()
        res = res[:-1] if res[-1]=='.' else res
        print(res)
        # GPT 3.5
        # if any([x in res for x in['answer b', 'b:', 'is b', 'answer: b']]) or res=='b':

        # GPT 4
        sents = [
            'correct answer is answer b',
            'answer is b'
        ]
        conditions = any([x in res for x in sents])
        last_sentence = res.split('.')[-1]
        if res.lower()=='b' or conditions or any([x in last_sentence for x in['answer b', 'b:', 'is b', 'answer: b']]):

            # B is tableqa, which is 1 in preds
            df.loc[i, 'preds_llm'] = 1
            df.loc[i, 'score_llm'] = row['acc_tableqa']
        else:
            # A is text_to_sql, which is 0 in preds
            df.loc[i, 'preds_llm'] = 0
            df.loc[i, 'score_llm'] = row['acc_text_to_sql']
        response.append(res)
        if 'pre_response' in row:
            pre_response.append(row['pre_response'])
        else:
            pre_response.append('')

        print(df.loc[i, 'preds_llm'])

        continue

    if ans_text_to_sql in ['', 'nan', 'na']:
        res = 'B'
        df.loc[i, 'score_llm'] = row['acc_tableqa']
        df.loc[i, 'response'] = res
        response.append(res)
        pre_response.append('')
        continue

    if acc_tableqa==acc_text_to_sql:
        res = 'A'
        df.loc[i, 'score_llm'] = row['acc_text_to_sql']
        df.loc[i, 'response'] = res
        response.append(res)
        pre_response.append('')
        continue

    new_ans_tableqa = ans_tableqa.replace('|', ', ')
    new_ans_text_to_sql = ans_text_to_sql.replace('|', ', ')

    example = """
Question: how many coaches had above 500 wins?
Table: col : coach | years | seasons | wins | losses | ties | pct row 1 : o. b. "rip" sanderson | 1908 | 1 | 5 | 1 | 0 | 0.833 row 2 : ed sabre | 1920 | 1 | 3 | 1 | 0 | 0.75 row 3 : mack erwin | 1957-60 | 4 | 62 | 30 | 1 | 0.672 row 4 : chal port | 1965-91 | 27 | 641 | 386 | 2 | 0.624 row 5 : fred jordan | 1992-pres | 22 | 726 | 552 | 0 | 0.568 row 6 : john d. mcmillan | 1952-53 | 2 | 14 | 15 | 0 | 0.483 row 7 : jim newsome | 1961-64 | 4 | 37 | 43 | 0 | 0.463 row 8 : bunzy o'neal | 1948 | 1 | 6 | 7 | 0 | 0.462 row 9 : george c. rogers | 1914-15, 1921-24 | 6 | 26 | 33 | 1 | 0.441 row 10 : fred montsdeoca | 1954-56 | 2 | 22 | 31 | 0 | 0.415
This table is not complete.
Potential answer: 2
Q: Does the provided incomplete table provide enough information to confirm if the potential answer is correct or incorrect? Yes or No.
A: No. In the provided table, the coach chal port has 641 wins and the coach fred jordan has 726 wins. Therefore, there are 2 coaches who had above 500 wins based on this incomplete table. However, if there are other coaches in the table who had above 500 wins, the potential answer could be incorrect. So it can not be confirmed if the potential answer is correct or incorrect.
"""

    if acc_tableqa != acc_text_to_sql:
        truncated = int(df.loc[i, 'truncated_tableqa']) == 1
        if truncated:
            content = 'At each time, you will get a table, a question, and a potential answer. In the table, cells are separated by "|".\n'
            content += example
            question = context.split('col :')[0]
            content += f"\nQuestion:{question}"
            content += context.replace(question, '\nTable: ')
            content += '\nThis table is not complete.'
            content += f'\nPotential answer: {new_ans_text_to_sql}'
            content += '\nQ: Does the provided incomplete table provide enough information to confirm if the potential answer is correct or not? Yes or No.'
            content += '\nA:'

            df.loc[i, 'pre_content'] = content
            completion = client.chat.completions.create(
                model = model,
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": content}
                ],
                temperature=0 # consistent output
            )
            p_res = completion.choices[0].message.content
            pre_response.append(p_res)
        else:
            pre_response.append('')

        
        if not truncated or (p_res and 'yes' in p_res.lower()[:5]):
            content = 'You will get a table, a question, and two potential answers. In the table, cells are separated by "|". '
            content += '\n'+context
            content += f'\nA: {new_ans_text_to_sql}'
            content += f'\nB: {new_ans_tableqa}'
            content += '\nBased on the table, which answer do you think is correct, answer A or answer B?\
Think step by step and write down your reasoning steps. In the last sentence, please respond only with either answer A or answer B.'

            completion = client.chat.completions.create(
                model = model,
                messages = [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": content}
                ],
                temperature=0 # consistent output
            )
            res = completion.choices[0].message.content
            response.append(res)
        else:
            response.append('A')
    else:
        pre_response.append('')
        response.append('A')


    df.loc[i, 'pre_response'] = pre_response[-1]
    df.loc[i, 'response'] = response[-1]


df.to_csv(file_path, index=False)


