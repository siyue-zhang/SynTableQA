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
import re
import os
from sklearn import linear_model
import random
from copy import deepcopy
from metric.squall_evaluator import to_value_list

seed = 2024
np.random.seed(seed)
random.seed(seed)


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
        tableqa_dev = pd.read_csv(f"./predict_old/{dataset}/{dt}{a}_tableqa_dev{s}.csv")
        text_to_sql_dev = pd.read_csv(f"./predict_old/{dataset}/{dt}{d}{a}_text_to_sql_dev{s}.csv")

        if dataset=='squall':
            df = tableqa_dev[['id','tbl','question','answer','src']]
            df['query_fuzzy'] = text_to_sql_dev['query_fuzzy']
        else:
            df = tableqa_dev[['db_id','question','answer','src']]
            df['answer_fuzzy'] = tableqa_dev['answer_fuzzy']

        df['ans_text_to_sql'] = text_to_sql_dev['queried_ans']
        df['ans_tableqa'] = tableqa_dev['predictions']

        df['acc_text_to_sql'] = text_to_sql_dev['acc'].astype('int16')
        df['acc_tableqa'] = tableqa_dev['acc'].astype('int16')

        df['log_prob_text_to_sql'] = text_to_sql_dev['log_prob']
        df['log_prob_tableqa'] = tableqa_dev['log_prob']

        df['truncated_text_to_sql'] = text_to_sql_dev['truncated'].astype('int16')
        df['truncated_tableqa'] = tableqa_dev['truncated'].astype('int16')

        df['nl_headers'] = text_to_sql_dev['nl_headers']
        df['query_fuzzy'] = text_to_sql_dev['query_fuzzy']

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


def separate_cols(cols):

    # cols = ['1_id', '2_agg', '3_constituency', '4_constituency_number', '5_region', '6_name', '7_name_first', '8_name_second', '9_party', '10_last_elected', '11_last_elected_number']
    cols = ['_'.join(col.split('_')[1:]) for col in cols[2:]]
    original_cols = []
    processed_cols = []

    for i in range(len(cols)):
        if i==0:
            current = cols[i]
            current_ = 'c1'
            count = 1
            original_cols.append(current_)
        else:
            if current in cols[i]:
                processed_cols.append(cols[i].replace(current, current_))
            else:
                current = cols[i]
                count += 1
                current_ = f'c{count}'
                original_cols.append(current_)

    return original_cols, processed_cols


def extract_features(df, tokenizer, qonly=False):

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



        if not qonly:
            ####### text_to_sql answer features #######
            ans_text_to_sql = str(row['ans_text_to_sql'])
            ans_text_to_sql_list = ans_text_to_sql.split('|')
            text_to_sql_value_list = to_value_list(ans_text_to_sql_list)
            sql = row['query_fuzzy']

            # number of answers
            features.append(len(ans_text_to_sql_list))

            # answers have none
            hasNan = 0
            for ans in ans_text_to_sql_list:
                if ans.lower() in ['nan', 'none', '']:
                    hasNan = 1
                    break
            features.append(hasNan)

            # answers have string
            hasStr = 0
            for v in text_to_sql_value_list:
                if str(v)[0]=='S':
                    hasStr = 1
                    break
            features.append(hasStr)

            # answers have number
            hasNum = 0
            for v in text_to_sql_value_list:
                if str(v)[0]=='N':
                    hasNum = 1
                    break
            features.append(hasNum)

            # answers have date
            hasDat = 0
            for v in text_to_sql_value_list:
                if str(v)[0]=='D':
                    hasDat = 1
                    break
            features.append(hasDat)


            # number of overlap words between question and answer
            qwords = set(question.lower().split())
            awords = set(re.split(r'\s|\|', ans_text_to_sql.lower()))
            features.append(len(qwords.intersection(awords)))

            # generation probability
            log_prob = float(row['log_prob_text_to_sql'])
            features.append(10**log_prob)

            # if the predicted sql after fuzzy match uses the processed column
            cols = row['nl_headers']
            original_cols, processed_cols = separate_cols(cols)
            usePro = 0
            for col in processed_cols:
                if col in sql:
                    usePro = 1
                    break
            features.append(usePro)
    
        if not qonly:
            ####### tableqa answer features #######
            ans_tableqa = str(row['ans_tableqa'])
            ans_tableqa_list = ans_tableqa.split('|')
            tableqa_value_list = to_value_list(ans_tableqa_list)

            # answers have string
            hasStr = 0
            for v in tableqa_value_list:
                if str(v)[0]=='S':
                    hasStr = 1
                    break
            features.append(hasStr)

            # answers have number
            hasNum = 0
            for v in tableqa_value_list:
                if str(v)[0]=='N':
                    hasNum = 1
                    break
            features.append(hasNum)

            # answers have date
            hasDat = 0
            for v in tableqa_value_list:
                if str(v)[0]=='D':
                    hasDat = 1
                    break
            features.append(hasDat)

            # number of overlap words between question and answer
            qwords = set(question.lower().split())
            awords = set(re.split(r'\s|\|', ans_tableqa.lower()))
            features.append(len(qwords.intersection(awords)))

            # generation probability
            log_prob = float(row['log_prob_tableqa'])
            features.append(10**log_prob)

        X.append(features)
        Y.append(int(row['labels']))

    X = np.array(X)
    Y = np.array(Y)
    
    print ("data shape: ", X.shape)

    return X, Y


def fit_and_save(X, Y, model, tol=1e-5, name=""):
    if model == "SGD":
        classifier = linear_model.SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, tol=tol, verbose=1,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=6)
    elif model == "LR":
        classifier = linear_model.LogisticRegression(C=1.0, max_iter=1000, verbose=2, tol=tol)
    elif model == "kNN":
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif model == "SVM":
        classifier = SVC(kernel="linear", C=0.025, verbose=True, probability=True)
        # classifier = SVC(gamma=2, C=1, verbose=True, probability=True)
    elif model == "DecisionTree":
        classifier = DecisionTreeClassifier()
    elif model == "RandomForest":
        classifier = RandomForestClassifier()
    elif model == "AdaBoost":
        classifier =  AdaBoostClassifier()
    elif model == "MLP":
        classifier = MLPClassifier(verbose=True, early_stopping=True, validation_fraction=0.1, n_iter_no_change=2, tol=1e-4)

    classifier.fit(X, Y)
    train_score = classifier.score(X, Y)

    print ("Acc on training set: {:.3f}".format(train_score))

    with open("classifiers/{}_{}.pkl".format(model, name), "wb") as f:
        pickle.dump(classifier, f)


def load_and_predict(X, Y, model, name=""):
    with open("classifiers/{}_{}.pkl".format(model, name), "rb") as f:
        classifier = pickle.load(f)
    
    # print ("coefs: ", classifier.coef_)
    
    # return classifier.predict_proba(X)
    # return classifier.predict(X)
    return classifier.score(X, Y)


if __name__=='__main__':
    df_train, df_dev = load_dfs()

    from transformers import TapexTokenizer
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base")
    model = 'SVM'
    X, Y = extract_features(df_train, tokenizer)
    fit_and_save(X, Y, model)
    print('\n')

    X, Y = extract_features(df_dev, tokenizer)
    test_scores = load_and_predict(X, Y, model)
    print ("test score: ", test_scores)
