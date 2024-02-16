import numpy as np
import pandas as pd
from metric.squall_evaluator import Evaluator

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=None): 

    def compute_metrics(eval_preds, meta=None):
        # nonlocal tokenizer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # prepare the prediction format for the evaluator
        preds, labels = postprocess_text(decoded_preds, decoded_labels)
        predictions = []
        for i, (pred, label) in enumerate(zip(preds, labels)):
            prediction = {'pred': pred, 'nt': eval_dataset['nt'][i]}
            predictions.append(prediction)
        evaluator = Evaluator()
        separator = '|'
        correct_flag = evaluator.evaluate_tableqa(predictions, separator)

        if stage:
            to_save = {'id': eval_dataset['nt'],
                       'tbl': eval_dataset['tbl'],
                       'question': eval_dataset['question'],
                       'answer': eval_dataset['answer_text'],
                       'acc': [int(b) for b in correct_flag],
                       'predictions':preds,
                       'src': eval_dataset['src'],
                       'truncated': eval_dataset['truncated'],
                       'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids'])}
            if meta:
                to_save['log_probs_sum'] = meta['log_probs_sum']
                to_save['log_probs_avg'] = meta['log_probs_mean']

            df = pd.DataFrame(to_save)
            df.to_csv(f'./predict/squall/{stage}.csv', na_rep='',index=False)
            print('predictions saved! ', stage)
            
        return {"acc": np.round(np.mean(correct_flag),4)}
    return compute_metrics


