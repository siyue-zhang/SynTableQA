import json
import re
import numpy as np
import pandas as pd
from fuzzywuzzy import process
from metric.squall_evaluator import to_value_list, check_denotation
from utils.misc import execute_query



def postprocess_text(decoded_preds):

    predictions=[]
    for pred in decoded_preds:
        pred=pred.replace('!>', '<').replace('< =', '<=').replace('> =', '>=')
        predictions.append(pred)
    
    return predictions


def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=None):    
    def compute_metrics(eval_preds):
        # nonlocal tokenizer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # prepare the prediction format for the evaluator
        predictions = postprocess_text(decoded_preds)
        predicted = []
        correct_flag = []
        for i, pred in enumerate(predictions):
            answer = eval_dataset['answer'][i]
            target = eval_dataset['query'][i]
            db_path = eval_dataset['db_path'][i]
            db_id = eval_dataset['db_id'][i]
            question = eval_dataset['question'][i]
            path = db_path + "/" + db_id + "/" + db_id + ".sqlite"

            try:
                queried = execute_query(path, pred)
            except Exception as e:
                queried = []
            predicted.append(queried)
            
            # print('\n', question, '\n', 'pred: ', pred, '\n', 'target: ', target , '\n', queried, '\n', answer)

            predicted_values = to_value_list(queried)
            target_values = to_value_list(answer.split("|"))
            correct = check_denotation(target_values, predicted_values)
            correct_flag.append(correct)
            # print(correct, '\n-------')

        if stage:
            to_save = {'db_id': eval_dataset['db_id'],
                       'question': eval_dataset['question'],
                       'answer': eval_dataset['answer'],
                       'acc': [int(b) for b in correct_flag],
                       'query': eval_dataset['query'], 
                       'query_pred': predictions,
                       'queried_ans': predicted,
                       'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids'])}
            df = pd.DataFrame(to_save)
            df.to_csv(f'./predict/{stage}.csv', na_rep='')
            print('predictions saved! ', stage)

        return {"acc": np.round(np.mean(correct_flag),4)}
    return compute_metrics