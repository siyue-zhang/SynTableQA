import numpy as np
import pandas as pd
import json
from metric.squall_evaluator import Evaluator
import pandas as pd


def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=None, eval_csv=None):
    tableqa = pd.read_csv(eval_csv['tableqa'])
    text_to_sql = pd.read_csv(eval_csv['text_to_sql'])

    # f"\nAnswer Choice 0 : {ans_tableqa}\nAnswer Choice 1 : {ans_text_to_sql}\n"
    def compute_metrics(eval_preds):
        # nonlocal tokenizer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        TP, FP, TN, FN = 0,0,0,0
        correct_flag = []
        pred_dict = {}
        for i in range(preds.shape[0]):
            pred = preds[i,:].argmax()
            label = labels[i]
            sample = eval_dataset[i]
            pred_dict[sample['id']] = pred

            if pred==label:
                correct_flag.append(True)
            else:
                correct_flag.append(False)

            if label==1 and pred==1:
                TP+=1
            elif label==0 and pred==1:
                FP+=1
            elif label==0 and pred==0:
                TN+=1
            else:
                FN+=1
              
        if TP+FP==0:
            precision=0
        else:
            precision = TP/(TP+FP)
        if TP+FN==0:
            recall=0
        else:
            recall = TP/(TP+FN)
        if precision * recall == 0:
            f1=0
        else:
            f1=np.round(2 * (precision * recall) / (precision + recall), 4)

        ids = []
        tbls = []
        questions = []
        acc_tableqa = []
        acc_text_to_sql =[]
        picks = []
        scores = []
        src = []
        for j in range(tableqa.shape[0]):
            id = tableqa.loc[j,'id']
            ids.append(id)
            tbls.append(tableqa.loc[j, 'tbl'])
            questions.append(tableqa.loc[j, 'question'])
            a_tableqa = tableqa.loc[j, 'acc']
            acc_tableqa.append(a_tableqa)
            a_text_to_sql = text_to_sql.loc[j, 'acc']
            acc_text_to_sql.append(a_text_to_sql)
            if id in pred_dict:
                pick = pred_dict[id]
            else:
                pick = 0
            picks.append(pick)
            scores.append(a_tableqa if pick==0 else a_text_to_sql)
            if src in tableqa:
                src.append(tableqa.loc[j, 'src'])

        if stage:
            to_save = {'id': ids,
                       'tbl': tbls,
                       'question': questions,
                       'acc_tableqa': acc_tableqa,
                       'acc_text_to_sql': acc_text_to_sql,
                       'pick': picks,
                       'score': scores,
                       'src': src}
            df = pd.DataFrame(to_save)
            df.to_csv(f'./predict/{stage}.csv')
            print('predictions saved! ', stage)
        
        return {"acc": np.round(np.mean(scores),4),
                "acc_cls": np.round(np.mean(correct_flag),4),
                "f1": f1}
    return compute_metrics