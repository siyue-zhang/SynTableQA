import numpy as np
import pandas as pd
import json
from metric.squall_evaluator import Evaluator


def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=None):    

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        TP, FP, TN, FN = 0,0,0,0
        correct_flag = []
        scores = []
        input_tokens = [] 
        predictions = []
        for i in range(preds.shape[0]):
            pred = preds[i,:].argmax()
            predictions.append(pred)
            label = labels[i]
            sample = eval_dataset[i]
            acc_text_to_sql = sample['acc_text_to_sql']
            ans_text_to_sql = sample['ans_text_to_sql']
            acc_tableqa = sample['acc_tableqa']
            input_tokens.append(tokenizer.decode(sample['input_ids']))

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
            if len(ans_text_to_sql)>0 and ans_text_to_sql.lower() not in ['nan', 'none', 'na'] and pred==1:
                score = acc_text_to_sql
            else:
                score = acc_tableqa
            scores.append(score)
            
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
        
        if stage:
            to_save = {'id': eval_dataset['id'],
                       'tbl': eval_dataset['tbl'],
                       'question': eval_dataset['question'],
                       'answer': eval_dataset['answer'],
                       'inputs': input_tokens,
                       'preds': predictions,
                       'labels': labels,
                       'acc_tableqa': eval_dataset['acc_tableqa'],
                       'ans_tableqa': eval_dataset['ans_tableqa'],
                       'acc_text_to_sql': eval_dataset['acc_text_to_sql'],
                       'ans_text_to_sql': eval_dataset['ans_text_to_sql'],
                       'query_fuzzy': eval_dataset['query_fuzzy'],
                       'score': scores}
            df = pd.DataFrame(to_save)
            df.to_csv(f'./predict/{stage}.csv', na_rep='')
            print('predictions saved! ', stage)
        
        return {"acc": np.round(np.mean(scores),4),
                "acc_cls": np.round(np.mean(correct_flag),4),
                "recall": np.round(recall,4),
                "f1": f1,
                "acc_tableqa": np.average(eval_dataset["acc_tableqa"]),
                "acc_text_to_sql": np.average(eval_dataset["acc_text_to_sql"])}
    
    return compute_metrics