import numpy as np
import pandas as pd
from metric.squall_evaluator import to_value_list, check_denotation


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
        predictions = decoded_preds
        correct_flag = []
        for i, pred in enumerate(predictions):
            answer = eval_dataset['answer'][i]
            # print('\n', question, '\n', 'pred: ', pred, '\n', 'target: ', target , '\n', queried, '\n', answer)
            question = eval_dataset['question'][i]
            ordering_keywords = ['descending', 'ascending', 'sorted by']
            if any(keyword in question for keyword in ordering_keywords):
                pred = ', '.join([x.strip() for x in pred.split(',')])

            pred = pred.split("|")
            pred = [x.strip() for x in pred]

            ans = answer.split("|")
            if len(pred)!=len(ans):
                correct = False
            else:

                if fuzzy:
                    pass

                predicted_values = to_value_list(pred)
                target_values = to_value_list(ans)
                correct = check_denotation(target_values, predicted_values)
            correct_flag.append(correct)

        if stage:
            to_save = {'db_id': eval_dataset['db_id'],
                       'question': eval_dataset['question'],
                       'answer': eval_dataset['answer'],
                       'acc': [int(b) for b in correct_flag],
                       'query': eval_dataset['query'], 
                       'pred': predictions,
                       'src': eval_dataset['src'],
                       'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids'])}
            df = pd.DataFrame(to_save)
            df.to_csv(f'./predict/spider/{stage}.csv', na_rep='')
            print('predictions saved! ', stage)

        return {"acc": np.round(np.mean(correct_flag),4)}
    return compute_metrics