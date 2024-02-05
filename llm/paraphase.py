from openai import OpenAI
import pandas as pd
import json
import os

client = OpenAI(api_key="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")

df = pd.read_csv("predict/squall_selector_train1.csv")
ori_dict = {}
for i, row in df.iterrows():
    ori_dict[row['id']] = row['question']
    # if i>1000:
    #     break

data = None
filename = "llm/aug_questions.json"
# Check if the file exists in the current directory
if os.path.exists(filename):
    # File exists, so open and load the JSON data
    with open(filename, 'r') as file:
        data = json.load(file)
    print(f"File '{filename}' loaded successfully.")    
else:
    print(f"File '{filename}' does not exist in the current directory.")

if data:
    aug_dict = data
else:
    aug_dict = {}

for id in ori_dict:
    print(id)
    if id not in aug_dict:
        q = ori_dict[id]
        print('new ->', id)
        completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", 
            "content": "You are a writing assistant for paraphasing the sentence. \
                        For each given sentence, you should rewrite for 3 times, \
                        one paraphased sentence in one line. \
                        You should not change the meaning of original sentence. \
                        You should not use words such as you, me, and I, in the sentence. \
                        You should not add anything else other than the paraphased sentences in the response."},
            {"role": "user", 
            "content": f"Can you rephrase this sentence: {q}"}
        ]
        )
        reply = completion.choices[0].message.content
        qs = reply.split("\n")
        aug_dict[id] = [s for s in qs if len(s)>0]

with open('llm/aug_questions.json', 'w') as json_file:
    json.dump(aug_dict, json_file)
print('aug data saved!')




