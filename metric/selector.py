import numpy as np
import pandas as pd
import json
from metric.squall_evaluator import Evaluator


def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=None):    
    def compute_metrics(eval_preds):
        # nonlocal tokenizer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        
        TP, FP, TN, FN = 0,0,0,0
        correct_flag = []
        acc = []
        for i in range(preds.shape[0]):
            pred = preds[i,:].argmax()
            label = labels[i]
            sample = eval_dataset[i]

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
            
            if pred==0:
                tmp = int(sample['acc_text_to_sql'])
            else:
                tmp = int(sample['acc_tableqa'])
            acc.append(tmp)
            
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
                       'label': eval_dataset['label'],
                       'acc': [int(b) for b in correct_flag],
                       'predictions':preds,
                       'src': eval_dataset['src']}
            df = pd.DataFrame(to_save)
            df.to_csv(f'./predict/{stage}.csv')
            print('predictions saved! ', stage)
        
        return {"acc": np.round(np.mean(acc),4),
                # "acc_tableqa": np.round(np.mean(eval_dataset['acc_tableqa']),4),
                # "acc_text_to_sql": np.round(np.mean(eval_dataset['acc_text_to_sql']),4),
                "acc_cls": np.round(np.mean(correct_flag),4),
                "f1": f1}
    return compute_metrics