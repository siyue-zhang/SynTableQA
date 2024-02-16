import numpy as np
import pandas as pd
from metric.squall_evaluator import to_value_list, check_denotation


def prepare_compute_metrics(tokenizer, eval_dataset, stage=None, fuzzy=None):    
    def compute_metrics(eval_preds, meta=None):
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
        predictions = decoded_preds
        correct_flag = []

        for i, pred in enumerate(predictions):
            answers = eval_dataset['answers'][i]
            predicted_values = to_value_list(pred.split('|'))
            target_values = to_value_list(answers)
            correct = check_denotation(target_values, predicted_values)
            correct_flag.append(correct)

        if stage:
            to_save = {'id': eval_dataset['id'],
                       'question': eval_dataset['question'],
                       'answers': eval_dataset['answers'],
                       'acc': [int(b) for b in correct_flag],
                       'predictions': predictions,
                       'truncated': eval_dataset['truncated'],
                       'perturbation_type': eval_dataset['perturbation_type'],
                       'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids'])}
            if meta:
                to_save['log_probs_sum'] = meta['log_probs_sum']
                to_save['log_probs_avg'] = meta['log_probs_mean']    
        
            df = pd.DataFrame(to_save)
            df.to_csv(f'./predict/wikisql/{stage}.csv', na_rep='',index=False)
            print('predictions saved! ', stage)

        return {"acc": np.round(np.mean(correct_flag),4)}
    return compute_metrics