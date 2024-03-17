from openai import OpenAI
from prompts import *
import pandas as pd
import csv

client = OpenAI(
    api_key='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'
)

# model = 'gpt-3.5-turbo-0125'
model = 'gpt-4-0125-preview'
# model = ''
file_path = "llm/squall_classifier_test1.csv"

df = pd.read_csv(file_path)
df['truncated_tableqa'] = pd.to_numeric(df['truncated_tableqa'])
# df = df[df['truncated_tableqa']==1].reset_index(drop=True)


def read_tsv_into_dict(file_path):
    data_dict = {}
    with open(file_path, 'r', encoding='utf-8', newline='') as tsvfile:
        reader = csv.DictReader(tsvfile, delimiter='\t')
        for row in reader:
            for key, value in row.items():
                if key in data_dict:
                    data_dict[key].append(value)
                else:
                    data_dict[key] = [value]
    return data_dict

# Example usage:
meta_data = "data/WikiTableQuestions/misc/table-metadata.tsv"
meta_data = read_tsv_into_dict(meta_data)
tmp = {}
for i in range(len(meta_data['contextId'])):
    tmp[meta_data['contextId'][i]] = meta_data['title'][i]
meta_data = tmp
# disable
meta_data = {}

pre_response = []
response = []
for i, row in df.iterrows():
    
    k=34
    if i>k:
        break
    print('\n\n\n----row: ', i, '-----')
    if i!=k:
        continue


    tbl = df.loc[i, 'tbl']
    first, second = tbl.split("_")
    context_id = f"csv/{first}-csv/{second}.csv"
    if context_id in meta_data:
        table_title = meta_data[context_id].lower()
    else:
        table_title = None

    context = row['input_tokens'][3:-4]
    ans_tableqa = str(row['ans_tableqa'])
    ans_text_to_sql = str(row['ans_text_to_sql'])
    acc_tableqa = int(row['acc_tableqa'])
    acc_text_to_sql = int(row['acc_text_to_sql'])

    if 'response' in row and str(row['response'])!='nan':
        res = row['response'].strip().lower()

        # GPT 3.5
        # if any([x in res for x in['answer b', 'b:', 'is b', 'answer: b']]) or res=='b':

        # GPT 4
        sents = [
            'final answer is b',
            'final answer: b',
            'final answer:b',
            'correct answer is b'
        ]
        conditions = any([x in res for x in sents])

        if res.lower()=='b' or conditions:
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

        print(df.loc[i, 'content'])
        print('\n', res)
        print('\npreds_llm: ', df.loc[i, 'preds_llm'])
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
    new_ans_text_to_sql = ans_text_to_sql.split('|')
    tmp = []
    for ans in new_ans_text_to_sql:
        if ans not in tmp:
            try:
                ans=float(ans)
                if ans.is_integer():
                    ans = str(int(ans))
                else:
                    ans = str(ans)
            except Exception as e:
                pass
            tmp.append(ans)
    new_ans_text_to_sql = ', '.join(tmp)

#     example = """
# Question: how many coaches had above 500 wins?
# Table: col : coach | years | seasons | wins | losses | ties | pct row 1 : o. b. "rip" sanderson | 1908 | 1 | 5 | 1 | 0 | 0.833 row 2 : ed sabre | 1920 | 1 | 3 | 1 | 0 | 0.75 row 3 : mack erwin | 1957-60 | 4 | 62 | 30 | 1 | 0.672 row 4 : chal port | 1965-91 | 27 | 641 | 386 | 2 | 0.624 row 5 : fred jordan | 1992-pres | 22 | 726 | 552 | 0 | 0.568 row 6 : john d. mcmillan | 1952-53 | 2 | 14 | 15 | 0 | 0.483 row 7 : jim newsome | 1961-64 | 4 | 37 | 43 | 0 | 0.463 row 8 : bunzy o'neal | 1948 | 1 | 6 | 7 | 0 | 0.462 row 9 : george c. rogers | 1914-15, 1921-24 | 6 | 26 | 33 | 1 | 0.441 row 10 : fred montsdeoca | 1954-56 | 2 | 22 | 31 | 0 | 0.415
# This table is not complete.
# Potential answer: 2
# Q: Does the provided incomplete table provide enough information to confirm if the potential answer is correct or incorrect? Yes or No.
# A: No. In the provided table, the coach chal port has 641 wins and the coach fred jordan has 726 wins. Therefore, there are 2 coaches who had above 500 wins based on this incomplete table. However, if there are other coaches in the table who had above 500 wins, the potential answer could be incorrect. So it can not be confirmed if the potential answer is correct or incorrect.
# """

    if acc_tableqa != acc_text_to_sql:
        truncated = int(df.loc[i, 'truncated_tableqa']) == 1
        if truncated:
            # content = 'You will get a question, a table, and a response.'
            content = 'You will get a question and a table.'
            content += f' The table is about "{table_title}".\n\n' if table_title else '\n\n'
            # to be checked if example is effective
            # content += example
            question = context.split('col :')[0].strip()
            content += '[Question] '+question+'\n\n'
            content += '[Table] '+context.replace(question, '').strip()
            content += '\nThis table is not complete.\n\n'
            # content += f'[Response] {new_ans_text_to_sql}\n\n'
#             content += 'Does this incomplete table have enough information to verify whether the answer is correct or \
# incorrect to the question? You should response "yes" or "no". If the answer mentions the information not provided by \
# the table, you should response "no". If providing more data rows to this incomplete table will change the correct answer to \
# the question, you ç.'
            content += 'Does this incomplete table provide all the information for the question? You should response "yes" or "no".'
            # content += '\nA:

            df.loc[i, 'pre_content'] = content
            print('\n///pre-content///')
            print(content)

            completion = client.chat.completions.create(
                model = model,
                messages = [
                    {"role": "system", "content": "You are an advanced AI capable of analyzing and understanding information within tables."},
                    {"role": "user", "content": content}
                ],
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                seed=123
            )
            p_res = completion.choices[0].message.content
            pre_response.append(p_res)
            print('\n///pre-response///')
            print(p_res)
        else:
            pre_response.append('')

        assert 1==2
        
        if not truncated or (p_res and 'yes' in p_res.lower()):
            content = 'You will get a question, a table, and an answer.'
            content += f' The table is about "{table_title}".\n\n' if table_title else '\n\n'
            question = context.split('col :')[0].strip()
            content += '[Question] '+question+'\n\n'
            content += '[Table] '+context.replace(question, '').strip()+'\n\n'
            content += f'[Answer A] {new_ans_text_to_sql}\n\n'
            content += f'[Answer B] {new_ans_tableqa}\n\n'
            content += 'Based on the given table, choose the more correct answer from A or B. \
Let\'s think step by step, and then give the final answer. If both answers are correct, \
choose the more natural answer to the question for humans. If both answers are incorrect, \
choose the closer one. Ensure the final answer format is either \
"Final Answer: A" or "Final Answer: B", no other form.'
            
            df.loc[i, 'content'] = content
            print('\n///content///')
            print(content)
            
            completion = client.chat.completions.create(
                model = model,
                messages = [
                    {"role": "system", "content": "You are an advanced AI capable of analyzing and understanding information within tables."},
                    {"role": "user", "content": content}
                ],
                temperature=0,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                seed=123
            )
            res = completion.choices[0].message.content
            print('\n///response///')
            print(res)
            response.append(res)
        else:
            response.append('A')
    else:
        pre_response.append('')
        response.append('A')


    df.loc[i, 'pre_response'] = pre_response[-1]
    df.loc[i, 'response'] = response[-1]


df.to_csv(file_path, index=False)


