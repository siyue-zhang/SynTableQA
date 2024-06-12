# import openai
import pandas as pd
import string
from openai import OpenAI
import re
import math
import numpy as np


client = OpenAI(
    api_key='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'
)

base_version = 3.5
if base_version == 4:
    model="gpt-4-0125-preview"
else:
    model = 'gpt-3.5-turbo'

path = f"/scratch/sz4651/Projects/SynTableQA/llm_base/gpt{base_version}/"
df_text_to_sql = pd.read_csv(path + 'squall_plus_llm_text_to_sql_test1.csv')
df_tableqa = pd.read_csv(path + 'squall_plus_llm_tableqa_test1.csv')

k = 1000
df_text_to_sql = df_text_to_sql[:k]
df_tableqa =df_tableqa[:k]

df_dict = {
    'id': df_tableqa['id'].to_list(),
    'tbl': df_tableqa['tbl'].to_list(),
    'question': df_tableqa['question'].to_list(),
    'ans_text_to_sql': df_text_to_sql['queried_ans_llm'].to_list(),
    'acc_text_to_sql': df_text_to_sql['acc_llm_text_to_sql'].to_list(),
    'ans_tableqa': df_tableqa['ans_llm_tableqa'].to_list(),
    'acc_tableqa': df_tableqa['acc_llm_tableqa'].to_list(),
    'truncated_tableqa': df_tableqa['truncated'].to_list(),
    'input_tokens': df_tableqa['input_tokens'].to_list()
}
df = pd.DataFrame(df_dict)
oracle = [int(x+y>0) for x, y in zip(df_text_to_sql['acc_llm_text_to_sql'].to_list(), df_tableqa['acc_llm_tableqa'].to_list())]
df['oracle'] = oracle

##############

with open('llm/squall/relevance-prompt.txt', 'r') as f:
    entity_align_prompt = f.read()

with open('llm/squall/alignment-prompt.txt', 'r') as f:
    number_align_prompt = f.read()  

with open('llm/squall/similarity-prompt.txt', 'r') as f:
    similar_prompt = f.read()   

with open('llm/squall/comparison-prompt.txt', 'r') as f:
    compare_prompt = f.read()   

with open('llm/squall/contradiction-prompt.txt', 'r') as f:
    contradiction_prompt = f.read()   

##############
def checkDigit(res):
    return res.replace('.','').replace(',','').isdigit()

def call_gpt(cur_prompt, stop, temperature = 0):
    ans = client.chat.completions.create(
                model=model,
                messages = [
                    {"role": "user", "content": cur_prompt}
                ],
                temperature=temperature)
    returned = ans.choices[0].message.content
    
    return returned

def similarAlign(question, response1, response2):

    prompt = similar_prompt + "\n\nQuestion: " + question + '\nResponse A: ' + response1 + '\nResponse B: ' + response2 + '\nAnswer: '
    print('\n\n///similarity///')
    print(prompt)
    gen = call_gpt(prompt, ['\n']).strip().strip(string.punctuation)
    for i_ in range(2): 
        if gen.lower() == 'yes' or gen.lower() == 'no':
            break
        gen = call_gpt(prompt, ['\n'], temperature = 1).strip().strip(string.punctuation)
    gen = gen.lower().strip()
    print('Answer: ', gen)
    if gen == 'yes':
        return True
    else:
        return False
    
def entityAlign(question, response):

    prompt = entity_align_prompt + "\n\nQuestion: " + question + '\nResponse: ' + response + '\nAnswer: '
    print('\n\n///entityAlign///')
    print(prompt)
    gen = call_gpt(prompt, ['\n']).strip().strip(string.punctuation)
    for i_ in range(2): 
        if gen.lower() == 'yes' or gen.lower() == 'no':
            break
        gen = call_gpt(prompt, ['\n'], temperature = 1).strip().strip(string.punctuation)
    gen = gen.lower().strip()
    print('Answer: ', gen)
    if gen == 'yes':
        return True
    else:
        return False

def numberAlign(question, response):

    prompt = number_align_prompt + "\n\nQuestion: " + question + '\nResponse: ' + response + '\nAnswer: '
    print('\n\n///numberAlign///')
    print(prompt)
    gen = call_gpt(prompt, ['\n']).strip().strip(string.punctuation)
    for i_ in range(2): 
        if gen.lower() == 'yes' or gen.lower() == 'no':
            break
        gen = call_gpt(prompt, ['\n'], temperature = 1).strip().strip(string.punctuation)
    gen = gen.lower().strip()
    print('Answer: ', gen)
    if gen == 'yes':
        return True
    else:
        return False
    

def countNumber(table, question):

    prompt = contradiction_prompt + "\n\nTable: " + table + '\nQuestion: ' + question + '\nAnswer: '
    print('\n\n///contradiction///')
    print(prompt)

    def get_number(str):
        numbers = re.findall(r'\d+', str)
        numbers = [int(num) for num in numbers]
        return numbers

    gen = call_gpt(prompt, ['\n']).strip().strip(string.punctuation)
    for i_ in range(2):
        if len(get_number(gen))>0:
            break
        gen = call_gpt(prompt, ['\n'], temperature = 1).strip().strip(string.punctuation)

    number = get_number(gen)
    number = number[-1] if len(number)>0 else 0
    print('Answer: ', number)
    return number



for i, row in df.iterrows():

    # if i < 62:
    #     continue

    print('\n----row: ', i, '-----')
    if i > 1000:
        break

    question = row['question']
    question = question.replace(' -lrb- ','(').replace(' -rrb-',')')
    ans_tableqa = row['ans_tableqa'].split("Final Answer:")[-1].strip().lower()
    ans_text_to_sql = str(row['ans_text_to_sql']).lower().strip()
    acc_tableqa = int(row['acc_tableqa'])
    acc_text_to_sql = int(row['acc_text_to_sql'])
    truncated = int(row['truncated_tableqa'])
    table = 'header: ' + row['input_tokens'].split('col : ')[1].replace('</s>','')

    nl_response_text_to_sql = []
    for res in ans_text_to_sql.split('|'):
        if res.lower() != 'none' and res not in nl_response_text_to_sql:
            nl_response_text_to_sql.append(res)
    response_text_to_sql = ', '.join(nl_response_text_to_sql)

    nl_response_tableqa = []
    for res in ans_tableqa.split('|'):
        if res not in nl_response_tableqa:
            nl_response_tableqa.append(res)
    response_tableqa = ', '.join(nl_response_tableqa)

    if acc_tableqa==acc_text_to_sql:
        df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
        print('CASE A: acc_tableqa = acc_text_to_sql')
        continue

    if acc_tableqa!=acc_text_to_sql:
        if ans_text_to_sql in ['', 'nan', 'na', 'none'] or len(nl_response_text_to_sql)==0:
            df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
            print('CASE B: acc_text_to_sql => nan')
            continue

    ##########################################
    # answer format correction 
    ##########################################

    # 6 pts vs 6
    if checkDigit(ans_text_to_sql) or checkDigit(ans_tableqa):
        if len(ans_text_to_sql) > len(ans_tableqa):
            shorter = ans_tableqa
            longer = ans_text_to_sql
            words_longer = longer.split()
            if len(words_longer)==2 and words_longer[0]==shorter and checkDigit(shorter):
                print('CASE D: ', longer, shorter)
                df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
                print('choose tableqa ', ans_tableqa)
                continue
        else:
            shorter = ans_text_to_sql
            longer = ans_tableqa
            words_longer = longer.split()
            if len(words_longer)==2 and words_longer[0]==shorter and checkDigit(shorter):
                print('CASE D: ', longer, shorter)
                df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
                print('choose text_to_sql ', ans_text_to_sql)
                continue

    # at denver broncos vs denver broncos
    if response_text_to_sql == 'at ' + response_tableqa:
        df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
        continue

    # 98,453 vs 98453
    if response_text_to_sql.replace(',','') == response_tableqa.replace(',',''):
        if ',' in response_tableqa:
            if response_text_to_sql.isdigit():
                df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
                continue
        else:
            if response_tableqa.isdigit():
                df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
                continue
    
    # no vs 0
    if response_text_to_sql=='0' and response_tableqa=='no':
        df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
        continue
    if response_text_to_sql=='1' and response_tableqa=='yes':
        df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
        continue

    if acc_tableqa!=acc_text_to_sql:

        ##########################################
        # answer entity and number of answer aligment
        ##########################################

        if checkDigit(row['ans_text_to_sql']) and checkDigit(row['ans_tableqa']):
            df.loc[i, 'entityAlign'] = 1
            df.loc[i, 'numberAlign'] = 1
            df.loc[i, 'similarity'] = 1
        else:
            if nl_response_tableqa[0]==nl_response_text_to_sql[0]:
                similarity = True
            else:
                similarity = similarAlign(question, nl_response_text_to_sql[0], nl_response_tableqa[0])
            df.loc[i, 'similarity'] = int(similarity)
            if df.loc[i, 'similarity']==1:
                df.loc[i, 'entityAlign'] = 1
                if 'name a ' in question and len(nl_response_text_to_sql)>1:
                    number_align = False
                elif len(nl_response_text_to_sql)>1:
                    number_align = numberAlign(question, response_text_to_sql)
                else:
                    number_align = True
                df.loc[i, 'numberAlign'] = int(number_align)
            else:
                entity_align = entityAlign(question, response_text_to_sql)
                df.loc[i, 'entityAlign'] = int(entity_align)

                if df.loc[i, 'entityAlign']==1:
                    if 'name a ' in question and len(nl_response_text_to_sql)>1:
                        number_align = False
                    elif len(nl_response_text_to_sql)>1:
                        number_align = numberAlign(question, response_text_to_sql)
                    else:
                        number_align = True
                    df.loc[i, 'numberAlign'] = int(number_align)
                else:
                    df.loc[i, 'numberAlign'] = 0
        
        if df.loc[i, 'numberAlign'] == 1:
            if truncated==1:
                print('CASE E: pass alignment check, table is truncated. ', response_text_to_sql)

                ##########################################
                # contradiction in counting
                ##########################################

                try:
                    number_text_to_sql = float(response_text_to_sql)
                except Exception as e:
                    number_text_to_sql = None
                if number_text_to_sql is not None and number_text_to_sql.is_integer() and number_text_to_sql<=20:
                    gpt_count = countNumber(table, question)
                    if gpt_count>number_text_to_sql:
                        df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
                        print(f'CASE E-1: counting contradiction. GPT: {gpt_count} > Text-to-SQL: {number_text_to_sql}')
                    else:
                        df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
                else:
                    df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
                continue
            else:

                ##########################################
                # compare two options
                ##########################################
                
                content = compare_prompt + '\n'
                content += "\n\nTable: " + table + "\nQuestion: " + question + '\nResponse A: ' + response_text_to_sql+ '\nResponse B: ' + response_tableqa + '\nAnswer: '
                print('\n\n')
                print(content)
                completion = client.chat.completions.create(
                model = model,
                messages = [
                    {"role": "system", "content": "You are an advanced AI capable of analyzing and understanding information within tables."},
                    {"role": "user", "content": content}
                ],
                temperature=0 # consistent output
            )
            res = completion.choices[0].message.content.lower()
            df.loc[i, 'comparison'] = res

            sents = [
                'final answer is b',
                'final answer: b',
                'final answer:b',
                'correct answer is b'
            ]
            conditions = any([x in res for x in sents])
            if conditions:
                df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
            else:
                df.loc[i, 'gpt_score'] = df.loc[i, 'acc_text_to_sql']
            print('CASE F: compare two answers. ', 'B is correct: ', conditions)
            print('GPT response: ', res)
            continue
        else:
            print('CASE G: text-to-sql answer aligment check fails, choose tableqa')
            df.loc[i, 'gpt_score'] = df.loc[i, 'acc_tableqa']
            continue

    raise NotImplementedError

gpt_scores = df.loc[:,'gpt_score'].values
gpt_scores = [x for x in gpt_scores if not isinstance(x, float) or not math.isnan(x)]
oracle = df.loc[:,'oracle'].values[:len(gpt_scores)]

print('\n\navg oracle: ', np.round(np.mean(oracle),4))
print('avg gpt: ', np.round(np.mean(gpt_scores),4))

df.to_csv(f'/scratch/sz4651/Projects/SynTableQA/llm_base/llm_squall_selector_{base_version}.csv', index=False)


