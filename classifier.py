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

def get_squall_table_shape(tbl):

    db_path = 'data/squall/tables/json/' + tbl + '.json'
    f = open(db_path)
    data = json.load(f)
    contents = data['contents']
    n_rows = 0
    n_cols = 0
    n_processed_cols = 0
    for l in contents:
        for k, col in enumerate(l):
            if k==0 and col['col'] not in ['id', 'agg']:
                n_cols += 1
            n_processed_cols += 1
            if n_rows==0:
                n_rows = len(col['data'])
    # print(tbl, n_rows, n_cols, n_processed_cols)     

    return n_rows, n_cols, n_processed_cols


def get_wikisql_table_shape(tbl):

    db_path = 'data/squall/tables/json/' + tbl + '.json'
    f = open(db_path)
    data = json.load(f)
    contents = data['contents']
    n_rows = 0
    n_cols = 0
    n_processed_cols = 0
    for l in contents:
        for k, col in enumerate(l):
            if k==0 and col['col'] not in ['id', 'agg']:
                n_cols += 1
            n_processed_cols += 1
            if n_rows==0:
                n_rows = len(col['data'])
    # print(tbl, n_rows, n_cols, n_processed_cols)     

    return n_rows, n_cols, n_processed_cols


def combine_csv(tableqa_dev, text_to_sql_dev, dataset):

    if dataset=='squall':
        df = tableqa_dev[['id','tbl','question','answer','src']]
        df.loc[:,['nl_headers']] = text_to_sql_dev['nl_headers']
    else:
        df = tableqa_dev[['id','table_id','question','answer','perturbation']]

    df.loc[:,['ans_text_to_sql']] = text_to_sql_dev['queried_ans']
    df.loc[:,['ans_tableqa']] = tableqa_dev['predictions']

    df.loc[:,['acc_text_to_sql']] = text_to_sql_dev['acc'].astype('int16')
    df.loc[:,['acc_tableqa']] = tableqa_dev['acc'].astype('int16')

    df.loc[:,['log_prob_sum_text_to_sql']] = text_to_sql_dev['log_probs_sum']
    df.loc[:,['log_prob_sum_tableqa']] = tableqa_dev['log_probs_sum']

    df.loc[:,['log_prob_avg_text_to_sql']] = text_to_sql_dev['log_probs_avg']
    df.loc[:,['log_prob_avg_tableqa']] = tableqa_dev['log_probs_avg']

    df.loc[:,['truncated_text_to_sql']] = text_to_sql_dev['truncated'].astype('int16')
    df.loc[:,['truncated_tableqa']] = tableqa_dev['truncated'].astype('int16')

    df.loc[:,['query_fuzzy']] = text_to_sql_dev['query_fuzzy']
    df.loc[:,['query_pred']] = text_to_sql_dev['query_pred']

    df.loc[:,['labels']] = [ 0 if int(x)==1 else 1 for x in df['acc_text_to_sql'].to_list()]
    df.loc[:,['input_tokens']] = tableqa_dev['input_tokens']

    return df


def load_dfs(dataset, aug=False, downsize_suffix=None):
    
    dfs_dev = []
    train_dev_ratio = 0.2
    dataset_suffix = 'squall_plus' if dataset=='squall' else 'wikisql'
    downsize_suffix = downsize_suffix if downsize_suffix else ''
    aug_suffix = '_aug' if aug else ''

    for s in range(5):
        tableqa_dev = pd.read_csv(f"./predict/{dataset}/{dataset_suffix}{aug_suffix}_tableqa_dev{s}.csv")
        text_to_sql_dev = pd.read_csv(f"./predict/{dataset}/{dataset_suffix}{downsize_suffix}{aug_suffix}_text_to_sql_dev{s}.csv")
        df = combine_csv(tableqa_dev, text_to_sql_dev, dataset)
        df = df[df['acc_tableqa'] != df['acc_text_to_sql']]
        dfs_dev.append(df)

    dfs_dev = pd.concat(dfs_dev, ignore_index=True).reset_index()
    tbls = list(set(dfs_dev['tbl'].to_list()))

    split_path = f'./task/classifier_{dataset}_splits.json'
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
    df_train = df_train.reset_index(drop=True) 
                
    df_dev = dfs_dev[dfs_dev['tbl'].isin(selector_dev_tbls)].reset_index(drop=True)

    return df_train, df_dev


def load_df_test(dataset, test_split, downsize_suffix=None):

    dataset_suffix = 'squall_plus' if dataset=='squall' else 'wikisql'
    downsize_suffix = downsize_suffix if downsize_suffix else ''

    tableqa_test = pd.read_csv(f"./predict/{dataset}/{dataset_suffix}_tableqa_test{test_split}.csv")
    text_to_sql_test = pd.read_csv(f"./predict/{dataset}/{dataset_suffix}{downsize_suffix}_text_to_sql_test{test_split}.csv")
    df_test = combine_csv(tableqa_test, text_to_sql_test, dataset)

    return df_test


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


def is_abbreviation(abbreviation, full_string):
    if len(abbreviation) > len(full_string):
        return False
    
    abbreviation_index = 0
    for char in full_string:
        if abbreviation_index < len(abbreviation) and char == abbreviation[abbreviation_index]:
            abbreviation_index += 1
    return abbreviation_index == len(abbreviation)


def preprocess_df(df):

    hq_overlap = []
    complexCols = []
    num_pro_cols = []
    for index, row in df.iterrows():
        
        question = row["question"]
        table_id = row["tbl"]
        pro_col_names = []

        db_path = 'data/squall/tables/json/' + table_id + '.json'
        f = open(db_path)
        data = json.load(f)

        # number of headers have overlap words with question
        headers = [h for k, h in enumerate(data['headers']) if k>1]
        num_overlap = sum([h in question for h in headers])
        hq_overlap.append(num_overlap)
    
        complexCol = []
        for cc in data['contents']:
            c = cc[0]
            if len(cc)==1 and c['type']=='TEXT':
                for sep in [', ', '&', '(']:
                    isComplex = any([sep in str(y) for y in c['data'] if y])
                    if isComplex:
                        complexCol.append(c['col'])
                        break


        if len(complexCol)>0:
            complexCols.append('|'.join(complexCol))
        else:
            complexCols.append('')

        num_pro_col = 0
        for cc in data['contents']:
            for k, c in enumerate(cc):
                if k>0:
                    pro_col_names.append(c['col'])

        pro_col_names = sorted(pro_col_names, key=len, reverse=True)
        tmp = deepcopy(row['query_fuzzy'])
        for x in pro_col_names:
            count = tmp.count(x)
            if count>0:
                tmp = tmp.replace(x,'')
            num_pro_col += count
        num_pro_cols.append(num_pro_col)


    df.loc[:,['hq_overlap']] = hq_overlap
    df.loc[:, ['complexCols']] = complexCols
    df.loc[:, ['num_pro_col']] = num_pro_cols
    
    return df


def extract_squall_features(df, tableqa_tokenizer, text_to_sql_tokenizer, qonly=False):

    X = []
    Y = []
    feature_names = []
    table_shape = {}
    verbose = False
    acc_abb = []

    for index, row in df.iterrows():

        features = []
        row = dict(row)
        add_name  = index==0

        ####### question features #######
        question = row["question"]
        if question[-1] in ['.','?']:
            question = question[:-1] + ' ' + question[-1]

        if verbose:
            print('Sample: ', {k:row[k] for k in row if k not in ['input_tokens','nl_headers']})
            print('\nQuestion: ', question)

        qword = question.lower().split()
        # if "total" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('total')

        # if "difference" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('difference')

        # if "what" in qword and 'amount' not in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('what')

        # if "what" in qword and 'amount' in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('what amount')
        
        # if "who" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('who')
        
        # if "which" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('which')
        
        # if "when" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('when')
        
        # if "where" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('where')


        # if "how" in qword and "much" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('how much')
        
        # if "how" in qword and "many" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('how many')

        # if "how" in qword and "long" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('how long')

        # if "how" in qword and all([w not in qword for w in ['much', 'many', 'long']]):
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('how')

        # if "is" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('is')

        # if "was" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('was')

        # if "are" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('are')

        # if "were" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('were')

        # if "does" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('does')

        # if "did" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('did')

        # if "do" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('do')

        # if qword[0] in ['name','list','tell']:
        #     features.append(1)
        # else:
        #     features.append(0)
        # if add_name:
        #     feature_names.append('name/list/tell')


        features.append(len(tableqa_tokenizer.tokenize(question)))
        if add_name:
            feature_names.append('question token number')

        qwords = deepcopy(qword)
        num_count = 0
        for w in qwords:
            if w.replace('.', '').replace(',', '').replace('-', '').isnumeric():
                num_count += 1
        features.append(num_count)
        if add_name:
            feature_names.append('number of numerical values in question')

        ####### context table features #######
        tbl = row['tbl']
        get_table_shape = get_squall_table_shape
        if tbl not in table_shape:
            table_shape[tbl] = get_table_shape(tbl)
        # number of rows
        features.append(table_shape[tbl][0])
        if add_name:
            feature_names.append('number of table rows')

        # number of columns
        # features.append(table_shape[tbl][1])
        # features.append(table_shape[tbl][2])
        # if add_name:
        #     feature_names.append('number of table columns')

        # number of overlap between header words and question words
        features.append(row['hq_overlap'])
        if add_name:
            feature_names.append('number of overlap between header and question')
        
        if not qonly:
            ####### text_to_sql answer features #######
            ans_text_to_sql = str(row['ans_text_to_sql'])
            ans_text_to_sql_list = ans_text_to_sql.split('|')
            text_to_sql_value_list = to_value_list(ans_text_to_sql_list)
            sql = row['query_fuzzy']
            sql_ori = row['query_pred']

            # if sql use complex columns
            # hasCmp = 0
            # print(tbl)
            # print(sql)
            # print(row['complexCols'])
            # if row['complexCols']:
            #     for c in row['complexCols'].split('|'):
            #         if c in sql:
            #             hasCmp += 1
            # # print(hasCmp, '\\\\\n')
            # features.append(hasCmp)

            # query use complex column
            hasLst = '_list' in sql
            features.append(int(hasLst))
            if add_name:
                feature_names.append('_list')

            hasFst = any([x in sql for x in ['_first', '_second']])
            features.append(int(hasFst))
            if add_name:
                feature_names.append('_first/second')

            hasFst = any([x in sql for x in ['_parsed']])
            features.append(int(hasFst))
            if add_name:
                feature_names.append('_parsed')

            hasMm = any([x in sql for x in ['_maximum', '_minimum']])
            features.append(int(hasMm))
            if add_name:
                feature_names.append('_min/max')

            hasTim = any([x in sql for x in ['_year', '_month', '_day']])
            features.append(int(hasTim))
            if add_name:
                feature_names.append('_year/month/day')

            hasTim = any([x in sql for x in ['_hour', '_min']])
            features.append(int(hasTim))
            if add_name:
                feature_names.append('_hour/min')

            hasLen = '_length' in sql
            features.append(int(hasLen))
            if add_name:
                feature_names.append('_length')

            # number of predicted tokens
            n_tok_text_to_sql = len(text_to_sql_tokenizer.tokenize(sql_ori))
            features.append(n_tok_text_to_sql)
            if add_name:
                feature_names.append('number of sql tokens')

            # number of answers
            features.append(len(ans_text_to_sql_list))
            if add_name:
                feature_names.append('number of ans_tts')

            # answers have none
            hasNan = 0
            for ans in ans_text_to_sql_list:
                if ans.lower() in ['nan', 'none', '']:
                    hasNan += 1
            features.append(hasNan)
            if add_name:
                feature_names.append('number of Nan ans')

            # answers have string
            hasStr = 0
            for v in text_to_sql_value_list:
                if str(v)[0]=='S':
                    hasStr += 1
            features.append(hasStr)
            if add_name:
                feature_names.append('number of String ans')

            # answers have number
            hasNum_tts = 0
            for v in text_to_sql_value_list:
                if str(v)[0]=='N':
                    hasNum_tts += 1
            features.append(hasNum_tts)
            if add_name:
                feature_names.append('number of Numeric ans')

            # answers have date
            # hasDat_tts = 0
            # for v in text_to_sql_value_list:
            #     if str(v)[0]=='D':
            #         hasDat_tts += 1
            # features.append(hasDat)

            # # number of overlap words between question and answer
            # qwords = set(question.lower().split())
            # awords = set(re.split(r'\s|\|', ans_text_to_sql.lower()))
            # num_overlap = len(qwords.intersection(awords))
            # features.append(num_overlap)
            # if add_name:
            #     feature_names.append('number of overlap words between question and answer')

            # generation probability
            log_prob_avg = float(row['log_prob_avg_text_to_sql'])
            features.append(log_prob_avg)
            if add_name:
                feature_names.append('avg log prob tts')

            # log_prob_sum = float(row['log_prob_sum_text_to_sql'])
            # features.append(log_prob_sum)

            # if the predicted sql after fuzzy match uses the processed column
            features.append(row['num_pro_col'])
            if add_name:
                feature_names.append('number of processed column sql used')

            # if the input is truncated
            isTru = row['truncated_text_to_sql']
            features.append(isTru)
            if add_name:
                feature_names.append('tts table is truncated')


        if not qonly:
            ####### tableqa answer features #######
            ans_tableqa = str(row['ans_tableqa'])
            ans_tableqa_list = ans_tableqa.split('|')
            tableqa_value_list = to_value_list(ans_tableqa_list)

            # number of answers
            features.append(len(ans_tableqa_list))
            if add_name:
                feature_names.append('number of tqa ans')

            # number of predicted tokens
            n_tok_tableqa = len(tableqa_tokenizer.tokenize(ans_tableqa))
            features.append(n_tok_tableqa)
            if add_name:
                feature_names.append('number of tqa pred tokens')

            # answers have string
            hasStr = 0
            for v in tableqa_value_list:
                if str(v)[0]=='S':
                    hasStr += 1
            features.append(hasStr)
            if add_name:
                feature_names.append('number of String ans')

            # answers have number
            hasNum_tqa = 0
            for v in tableqa_value_list:
                if str(v)[0]=='N':
                    hasNum_tqa += 1
            features.append(hasNum_tqa)
            if add_name:
                feature_names.append('number of Numeric ans')

            # answers have date
            # hasDat_tqa = 0
            # for v in tableqa_value_list:
            #     if str(v)[0]=='D':
            #         hasDat_tqa += 1
            # features.append(hasDat)
            
            # if ans_tableqa is a subset of ans_text_to_sql
            isSub = int(all([x in ans_text_to_sql_list for x in ans_tableqa_list]))
            features.append(isSub)
            if add_name:
                feature_names.append('if ans_tableqa is a subset of ans_text_to_sql')

            # if isSub:
            #     print(row['id'])
            #     print(row['acc_tableqa'], ans_tableqa_list, row['acc_text_to_sql'], ans_text_to_sql_list)
            #     print('sub', isSub, '\n')

            isAbb = 0
            isAbb_b = 0
            if len(ans_tableqa_list)==1 and len(ans_text_to_sql_list)==1:
                A = ans_text_to_sql
                B = ans_tableqa
                if hasNum_tqa==0 and hasNum_tts==0:
                    mkrs = ['\n', '(', '-', ',', '&']
                    if any([m in A for m in mkrs]):
                        isAbb = int(is_abbreviation(B,A))
                    if any([m in A for m in mkrs]):
                        isAbb_b = int(is_abbreviation(A,B))
                
                if isAbb:
                    acc_abb.append(row['acc_tableqa'])
                    
            features.append(isAbb)
            if add_name:
                    feature_names.append('if ans_tableqa is a substring of ans_text_to_sql')

            # features.append(isAbb_b)
            # if add_name:
            #         feature_names.append('opposite substring')

                # if isAbb:
                #     print(row['id'])
                #     print(row['acc_tableqa'], ans_tableqa_list, row['acc_text_to_sql'], ans_text_to_sql_list)
                #     print(ans_tableqa.lower(), '/', ans_text_to_sql)
                #     print('abb', isAbb, '\n')
            # features.append(isAbb_b)


            # # number of overlap words between question and answer
            # qwords = set(question.lower().split())
            # awords = set(re.split(r'\s|\|', ans_tableqa.lower()))
            # num_overlap = len(qwords.intersection(awords))
            # features.append(num_overlap)
            # if add_name:
            #     feature_names.append('number of overlap words between question and answer')

            # generation probability
            log_prob_avg = float(row['log_prob_avg_tableqa'])
            features.append(log_prob_avg)
            if add_name:
                feature_names.append('avg log prob tqa')

            # log_prob_sum = float(row['log_prob_sum_tableqa'])
            # features.append(log_prob_sum)
            # if add_name:
            #     feature_names.append('sum log prob tqa')

            # if the input is truncated
            isTru = row['truncated_tableqa']
            features.append(isTru)
            if add_name:
                feature_names.append('tqa table is truncated')

            # if all answers are from the table input or question
            allFromTable = all([v.lower() in row['input_tokens'] for v in ans_tableqa_list])
            allFromTable = int(allFromTable)
            features.append(allFromTable)
            if add_name:
                feature_names.append('tqa answer is a substring from tableinput or question')

        X.append(features)
        Y.append(int(row['labels']))

    X = np.array(X)
    Y = np.array(Y)
    
    print ("data shape: ", X.shape)
    if 'if ans_tableqa is a substring of ans_text_to_sql' in feature_names:
        print('acc_isAbb', round(np.mean(acc_abb),3), len(acc_abb))
    return X, Y, feature_names


def extract_wikisql_features(df, tableqa_tokenizer, text_to_sql_tokenizer, qonly=False):

    X = []
    Y = []
    table_shape = {}
    verbose = False

    for index, row in df.iterrows():

        features = []
        row = dict(row)

        ####### question features #######
        question = row["question"]
        if question[-1] in ['.','?']:
            question = question[:-1] + ' ' + question[-1]

        if verbose:
            print('Sample: ', {k:row[k] for k in row if k not in ['input_tokens','nl_headers']})
            print('\nQuestion: ', question)

        qword = question.lower().split()
        if "what" in qword:
            features.append(1)
            if verbose:
                print('qword => what')
        else:
            features.append(0)
        
        if "who" in qword:
            features.append(1)
            if verbose:
                print('qword => who')
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

        if "how" in qword and "long" in qword:
            features.append(1)
        else:
            features.append(0)

        if "how" in qword and all([w not in qword for w in ['much', 'many', 'long']]):
            features.append(1)
        else:
            features.append(0)

        if "is" in qword:
            features.append(1)
        else:
            features.append(0)

        if "was" in qword:
            features.append(1)
        else:
            features.append(0)

        if "are" in qword:
            features.append(1)
        else:
            features.append(0)

        if "were" in qword:
            features.append(1)
        else:
            features.append(0)

        if "does" in qword:
            features.append(1)
        else:
            features.append(0)

        if "did" in qword:
            features.append(1)
        else:
            features.append(0)

        if "do" in qword:
            features.append(1)
        else:
            features.append(0)

        if qword[0] in ['name','list','tell']:
            features.append(1)
            if verbose:
                print('qword => name, list, tell')
        else:
            features.append(0)

        # if "only" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)
            
        # if "or" in qword:
        #     features.append(1)
        # else:
        #     features.append(0)

        features.append(len(tableqa_tokenizer.tokenize(question)))
        if verbose:
            print('Question token length: ', len(tableqa_tokenizer.tokenize(question)))

        # number of numerical values in question
        qwords = deepcopy(qword)
        num_count = 0
        for w in qwords:
            if w.replace('.', '').replace(',', '').replace('-', '').isnumeric():
                num_count += 1
        features.append(num_count)
        if verbose:
            print('Question has numerical values: ', num_count)

        ####### context table features #######
        tbl = row['table_id']
        get_table_shape = get_wikisql_table_shape
        if tbl not in table_shape:
            table_shape[tbl] = get_table_shape(tbl)
        # number of rows
        features.append(table_shape[tbl][0])
        # # number of columns
        # features.append(table_shape[tbl][1])
        # features.append(table_shape[tbl][2])
        if verbose:
            print('Table has rows: ', table_shape[tbl][0])

        if not qonly:
            ####### text_to_sql answer features #######
            ans_text_to_sql = str(row['ans_text_to_sql'])
            ans_text_to_sql_list = ans_text_to_sql.split('|')
            text_to_sql_value_list = to_value_list(ans_text_to_sql_list)
            sql = row['query_fuzzy']
            # sql = row['query_pred']

            if verbose:
                    print('\nText_to_sql Answer: ', ans_text_to_sql)
                    print('  ', text_to_sql_value_list)
                    print('  sql after fuzzy: ', sql)

            # # query use complex column
            # hasLst = '_list' in sql
            # features.append(int(hasLst))

            # hasFst = any([x in sql for x in ['_first', '_second']])
            # features.append(int(hasFst))

            # hasMm = any([x in sql for x in ['_maximum', '_minimum']])
            # features.append(int(hasMm))

            # hasTim = any([x in sql for x in ['_year', '_month', '_day']])
            # features.append(int(hasTim))

            # hasLen = '_length' in sql
            # features.append(int(hasLen))

            # # number of predicted tokens
            # n_tok_text_to_sql = len(text_to_sql_tokenizer.tokenize(row['query_pred']))
            # features.append(n_tok_text_to_sql)

            # features.append(int(ans_text_to_sql=='0'))

            # number of answers
            features.append(len(ans_text_to_sql_list))
            if verbose:
                    print('  Number of answers: ', len(ans_text_to_sql_list))

            # answers have none
            hasNan = 0
            for ans in ans_text_to_sql_list:
                if ans.lower() in ['nan', 'none', '']:
                    hasNan += 1
            features.append(hasNan)
            if verbose:
                print('  Number of NAN answers: ', hasNan)
            # answers have none
                
            # hasNL = ans_text_to_sql.count('\n')
            # features.append(hasNL)
            # if verbose:
            #     print('  Number of \\n: ', hasNL)

            # answers have string
            hasStr = 0
            for v in text_to_sql_value_list:
                if str(v)[0]=='S':
                    hasStr += 1
            features.append(hasStr)

            # answers have number
            hasNum = 0
            for v in text_to_sql_value_list:
                if str(v)[0]=='N':
                    hasNum += 1
            features.append(hasNum)

            # # answers have number, 01:47.6
            # hasNum2 = 0
            # for v in ans_text_to_sql_list:
            #     if v.replace(':','').replace('.','').isnumeric():
            #         hasNum2 += 1
            # features.append(hasNum2)

            # answers have date
            hasDat = 0
            for v in text_to_sql_value_list:
                if str(v)[0]=='D':
                    hasDat += 1
            features.append(hasDat)

            # number of overlap words between question and answer
            qwords = set(question.lower().split())
            awords = set(re.split(r'\s|\|', ans_text_to_sql.lower()))
            num_overlap = len(qwords.intersection(awords))
            features.append(num_overlap)
            if verbose:
                print('  Number of overlap words between question and answer: ', num_overlap)

            # # generation probability
            # log_prob_avg = float(row['log_prob_avg_text_to_sql'])
            # features.append(log_prob_avg)

            # log_prob_sum = float(row['log_prob_sum_text_to_sql'])
            # features.append(log_prob_sum)

            # if the predicted sql after fuzzy match uses the processed column
            cols = row['nl_headers'].split('|')
            original_cols, processed_cols = separate_cols(cols)
            usePro = 0
            for col in processed_cols:
                if col in sql:
                    usePro = 1
                    break
            features.append(usePro)

            # if the input is truncated
            isTru = row['truncated_text_to_sql']
            features.append(isTru)


        if not qonly:
            ####### tableqa answer features #######
            ans_tableqa = str(row['ans_tableqa'])
            ans_tableqa_list = ans_tableqa.split('|')
            tableqa_value_list = to_value_list(ans_tableqa_list)

            # features.append(int(ans_tableqa=='0'))

            if verbose:
                    print('\nTableqa Answer: ', ans_tableqa)
                    print('  ', tableqa_value_list)

            # isAbb = int(is_abbreviation(ans_text_to_sql, ans_tableqa.lower()))
            # features.append(isAbb)
            # isAbb = int(is_abbreviation(ans_tableqa.lower(), ans_text_to_sql))
            # features.append(isAbb)

            # number of answers
            features.append(len(ans_tableqa_list))
            if verbose:
                    print('  Number of answers: ', len(ans_tableqa_list))

            # # number of predicted tokens
            # n_tok_tableqa = len(tableqa_tokenizer.tokenize(ans_tableqa))
            # features.append(n_tok_tableqa)

            # answers have string
            hasStr = 0
            for v in tableqa_value_list:
                if str(v)[0]=='S':
                    hasStr += 1
            features.append(hasStr)
            
            # answers have number
            hasNum = 0
            for v in tableqa_value_list:
                if str(v)[0]=='N':
                    hasNum += 1
            features.append(hasNum)

            # answers have date
            hasDat = 0
            for v in tableqa_value_list:
                if str(v)[0]=='D':
                    hasDat += 1
            features.append(hasDat)

            # number of overlap words between question and answer
            qwords = set(question.lower().split())
            awords = set(re.split(r'\s|\|', ans_tableqa.lower()))
            num_overlap = len(qwords.intersection(awords))
            features.append(num_overlap)
            if verbose:
                print('  Number of overlap words between question and answer', num_overlap)

            # # generation probability
            # log_prob_avg = float(row['log_prob_avg_tableqa'])
            # features.append(log_prob_avg)

            # log_prob_sum = float(row['log_prob_sum_tableqa'])
            # features.append(log_prob_sum)

            # if the input is truncated
            isTru = row['truncated_tableqa']
            features.append(isTru)
            
            # # if all answers are from the table input or question
            # allFromTable = all([v.lower() in row['input_tokens'] for v in ans_tableqa_list])
            # allFromTable = int(allFromTable)
            # features.append(allFromTable)
            # if verbose:
            #     print('  All answers from table or question: ', allFromTable)
            #     print('------------------------------')

        X.append(features)
        Y.append(int(row['labels']))

    X = np.array(X)
    Y = np.array(Y)
    
    print ("data shape: ", X.shape)

    return X, Y


def load_and_predict(dataset, X, Y, feature_names, model, name=""):
    with open("classifiers/{}_{}_{}.pkl".format(dataset, model, name), "rb") as f:
        classifier = pickle.load(f)
    # print ("coefs: ", classifier.coef_)
    # return classifier.predict_proba(X)
    # return classifier.predict(X)
    f = classifier.feature_importances_
    names = feature_names

    print('\n-----------\n')
    print(len(names), ' : ', len(f))
    assert len(names)==len(f)
    for x, y in zip(names, f):
        print(x, ' : ', round(y,4))
    print('\n-----------\n')

    return classifier.score(X, Y)


def test_predict(dataset, df_test, tableqa_tokenizer, text_to_sql_tokenizer, model, name="", qonly=False):

    extract_features = extract_squall_features if dataset=='squall' else extract_wikisql_features
    X, Y, feature_names = extract_features(df_test, tableqa_tokenizer, text_to_sql_tokenizer, qonly)
    with open("classifiers/{}_{}_{}.pkl".format(dataset, model, name), "rb") as f:
        classifier = pickle.load(f)
    pred = classifier.predict(X)
    df_test.loc[:,["pred"]] = pred
    acc_scores = []
    cls_scores = []
    oracle = []
    for i in range(df_test.shape[0]):
        score = df_test.loc[i, 'acc_text_to_sql'] if pred[i]==0 else df_test.loc[i, 'acc_tableqa']
        acc_scores.append(int(score))
        if df_test.loc[i, 'acc_text_to_sql'] != df_test.loc[i, 'acc_tableqa']:
            cls_scores.append(int(pred[i]==Y[i]))
        oracle.append(int((df_test.loc[i, 'acc_text_to_sql']+df_test.loc[i, 'acc_tableqa'])>0))

    df_test.loc[:, ['scores']] = acc_scores
    df_test.loc[:, ['oracle']] = oracle

    print("test score: ", np.round(np.mean(cls_scores),4))
    print('\n')
    print(f"TTS acc  :  {np.round(np.mean(df_test.loc[:, 'acc_text_to_sql']),4)}")
    print(f"avg acc  :  {np.round(np.mean(acc_scores),4)}")
    print(f"oracle   :  {np.round(np.mean(oracle),4)}")

    return df_test


def fit_and_save(X, Y, model, tol=1e-5, name="", dataset="squall"):
    if model == "SGD":
        classifier = linear_model.SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, tol=tol, verbose=0,
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
        classifier = MLPClassifier(verbose=False, early_stopping=True, validation_fraction=0.1, n_iter_no_change=2, tol=1e-4)

    classifier.fit(X, Y)
    train_score = classifier.score(X, Y)

    print ("Acc on training set: {:.3f}".format(train_score))

    with open("classifiers/{}_{}_{}.pkl".format(dataset, model, name), "wb") as f:
        pickle.dump(classifier, f)


if __name__=='__main__':

    from transformers import TapexTokenizer, T5Tokenizer
    tableqa_tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large")
    text_to_sql_tokenizer = T5Tokenizer.from_pretrained("t5-large")

    # model = 'AdaBoost'
    model = 'RandomForest'
    dataset = 'squall'
    test_split = 1
    aug = False
    qonly = False
    name = 'aug' if aug else ''

    df_train, df_dev = load_dfs(dataset, aug)
    df_test = load_df_test(dataset, test_split)

    extract_features = extract_squall_features if dataset=='squall' else extract_wikisql_features

    df_train = preprocess_df(df_train)
    X, Y, feature_names = extract_features(df_train, tableqa_tokenizer, text_to_sql_tokenizer, qonly)
    fit_and_save(X, Y, model, name=name, dataset=dataset)
    print('\n')

    df_dev = preprocess_df(df_dev)
    X, Y, feature_names = extract_features(df_dev, tableqa_tokenizer, text_to_sql_tokenizer, qonly)
    dev_scores = load_and_predict(dataset, X, Y, feature_names, model, name=name)
    print ("dev score: ", dev_scores, '\n')

    df_test = preprocess_df(df_test)
    df_test_predict = test_predict(dataset, df_test, tableqa_tokenizer, text_to_sql_tokenizer, model, name=name, qonly=qonly)
    
    desired_order_selected = ['id', 'tbl', 'question', 'answer', 'acc_tableqa', 'ans_tableqa', 'acc_text_to_sql', 'ans_text_to_sql', 'query_pred', 'query_fuzzy', 'pred', 'labels', 'scores', 'oracle']
    remaining_columns = [col for col in df_test.columns if col not in desired_order_selected]
    df_test = df_test[desired_order_selected + remaining_columns]
    df_test.to_csv(f'./predict/{dataset}_classifier_test{test_split}.csv', na_rep='',index=False)
