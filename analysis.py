import pandas as pd
import json
import re

df = pd.read_csv('predict/squall_classifier_test1.csv')

for index, row in df.iterrows():
    table_id = row['tbl']
    table_path = f'./data/squall/tables/json/{table_id}.json'
    with open(table_path, 'r') as file:
        contents = json.load(file)
    question = row['question']
    table_data = contents['contents']
    input_tokens = row['input_tokens']

    headers = row['nl_headers'].split('|')
    header_words = []
    for i, h in enumerate(headers):
        for item in headers[i].split('_')[1:]:
            if item not in (['id', 'agg'] + header_words):
                header_words.append(item)
    print(header_words)
    assert 1==2
