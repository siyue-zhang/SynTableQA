import json
import numpy as np
import pandas as pd
from tqdm import tqdm 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import pickle
import os
from sklearn import linear_model
import random
from copy import deepcopy

np.random.seed(2024)
random.seed(2024)


def load_dfs():
    train_dev_ratio = 0.1
    splits = list(range(5))

    dfs_dev = []
    dataset = 'squall'
    if dataset=='squall':
        dt = 'squall_plus'
    d = ''
    a = ''

    for s in splits:
        tableqa_dev = pd.read_csv(f"./predict/{dataset}/{dt}{a}_tableqa_dev{s}.csv")
        text_to_sql_dev = pd.read_csv(f"./predict/{dataset}/{dt}{d}{a}_text_to_sql_dev{s}.csv")

        if dataset=='squall':
            df = tableqa_dev[['id','tbl','question','answer','src']]
            df['query_fuzzy'] = text_to_sql_dev['query_fuzzy']
        else:
            df = tableqa_dev[['db_id','question','answer','src']]
            df['answer_fuzzy'] = tableqa_dev['answer_fuzzy']

        df['ans_text_to_sql'] = text_to_sql_dev['queried_ans']
        df['ans_tableqa'] = tableqa_dev['predictions']
        df['acc_tableqa'] = tableqa_dev['acc'].astype('int16')
        df['acc_text_to_sql'] = text_to_sql_dev['acc'].astype('int16')
        df = df[df['acc_tableqa'] != df['acc_text_to_sql']]
        df['labels'] = [ 0 if int(x)==1 else 1 for x in df['acc_text_to_sql'].to_list()]
        dfs_dev.append(df)

    dfs_dev = pd.concat(dfs_dev, ignore_index=True).reset_index()
    tbls = list(set(dfs_dev['tbl'].to_list()))

    split_path = f'./task/selector{d}_splits.json'
    if os.path.exists(split_path):
        with open(split_path, 'r') as json_file:
            splits = json.load(json_file)
        selector_dev_tbls = splits['dev']
        selector_train_tbls = splits['train']
        print(f'load tbls from {split_path}.')
    else:
        tbl_dev_shuffle = deepcopy(tbls)
        random.shuffle(tbl_dev_shuffle)

        idx = int(len(tbl_dev_shuffle)*train_dev_ratio)
        selector_dev_tbls = tbl_dev_shuffle[:idx]
        selector_train_tbls = tbl_dev_shuffle[idx:]
        print('squall dev set is split into train and dev for training selector.')

        to_save = {'dev': selector_dev_tbls, 'train': selector_train_tbls}
        with open(split_path, 'w') as f:
            json.dump(to_save, f)

    df_train = dfs_dev[dfs_dev['tbl'].isin(selector_train_tbls)]
    if 'index' in df_train.columns:
        df_train = df_train.drop('index', axis=1)
    df_train['src'] = 'raw'
    df_train = df_train.reset_index(drop=True) 
                
    df_dev = dfs_dev[dfs_dev['tbl'].isin(selector_dev_tbls)].reset_index().astype('str')

    return df_train, df_dev


def extract_features(df, tokenizer):

    X = []
    Y = []

    for index, row in df.iterrows():

        features = []
        row = dict(row)

        ####### question features #######
        question = row["question"]
        if question[-1] in ['.','?']:
            question = question[:-1] + ' ' + question[-1]

        qword = question.lower().split()
        if "what" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "who" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "which" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "when" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "where" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "how" in qword and "much" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "how" in qword and "many" in qword:
            features.append(1)
        else:
            features.append(0)

        if "how" in qword and "much" not in qword and "many" not in qword:
            features.append(1)
        else:
            features.append(0)
        
        features.append(len(tokenizer.tokenize(row["question"])))

        # number of numerical values in question
        qwords = deepcopy(qword)
        num_count = 0
        for w in qwords:
            if w.replace('.', '').replace(',', '').replace('-', '').isnumeric():
                num_count += 1
        features.append(num_count)

        ####### context table features #######


        ####### tableqa answer features #######
        ans_tableqa = row['ans_tableqa']
        ans_tableqa_list = ans_tableqa.split('|')

        # number of answers
        features.append(len(ans_tableqa_list))

        # answers have number
        hasNum = 0
        for ans in ans_tableqa_list:
            if ans.replace('.', '').replace(',', '').replace('-', '').isnumeric():
                hasNum = 1
                break
        features.append(hasNum)

        hasStr = 0
        for ans in ans_tableqa_list:
            if not ans.replace('.', '').replace(',', '').replace('-', '').isnumeric():
                hasStr = 1
                break
        features.append(hasStr)



        print(features)
        assert 1==2


    X = np.array(X)
    Y = np.array(Y)
    
    print ("data shape: ", X.shape)

    return X, Y


if __name__=='__main__':
    df_train, df_dev = load_dfs()

    from transformers import TapexTokenizer
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base")
    X, Y = extract_features(df_train, tokenizer)
    print(df_train.keys())