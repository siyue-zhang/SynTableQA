import openai
from prompts import *
import pandas as pd
import csv


openai.api_key = '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'

file_path = "llm/key_part.xlsx"

df = pd.read_excel(file_path)

print(df)


def call_gpt(cur_prompt, stop, temperature = 0):
    ans = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                max_tokens=256,
                stop=stop,
                prompt=cur_prompt,
                temperature=temperature)
    returned = ans['choices'][0]['text']
    
    return returned

