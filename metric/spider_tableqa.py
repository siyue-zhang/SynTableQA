import numpy as np
import pandas as pd
from metric.squall_evaluator import to_value_list, check_denotation
from fuzzywuzzy import fuzz
from utils.misc import ordering_keywords

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
        pred_fuzzy = []
        for i, pred in enumerate(predictions):
            answer = eval_dataset['answer'][i]
            # print('\n', question, '\n', 'pred: ', pred, '\n', 'target: ', target , '\n', queried, '\n', answer)
            question = eval_dataset['question'][i]
            if any(keyword in question for keyword in ordering_keywords):
                pred = pred.replace('|', ',')
                pred_list = [x.strip() for x in pred.split(",")]
                ans_list = answer.split(", ")
                pred_f =  ", ".join(pred_list)
            else:
                pred_list = [x.strip() for x in pred.split("|")]
                ans_list = answer.split("|")
                pred_f = "|".join(pred_list)

            options = eval_dataset['options'][i]
            options = [x.replace('\n',' ').strip() for x in options]

            if len(pred_list)!=len(ans_list):
                correct = False
            else:
                if fuzzy:
                    # replace the answer by table contents through fuzzy matching
                    new_pred_list = []
                    for p in pred_list:
                        if p in options or p.replace('-','').replace('.','',1).isdigit():
                            pass
                        else:
                            ratio = [fuzz.ratio(p.lower(), s.lower()) for s in options]
                            max_index = ratio.index(max(ratio))
                            if ratio[max_index]>80:
                                print(f'{p} in {eval_dataset["db_id"][i]} has been replaced by: ')
                                p = options[max_index]
                                print(f'-> {p}\n')
                        new_pred_list.append(p)
                    
                    pred_list = new_pred_list

                if any(keyword in question for keyword in ordering_keywords):
                    pred_list = [", ".join(pred_list)]
                    ans_list = [", ".join(ans_list)]
                
                predicted_values = to_value_list(pred_list)
                target_values = to_value_list(ans_list)
                correct = check_denotation(target_values, predicted_values)
                pred_f = '|'.join(pred_list)

            pred_fuzzy.append(pred_f)
            correct_flag.append(correct)

        if stage:
            to_save = {'db_id': eval_dataset['db_id'],
                       'question': eval_dataset['question'],
                       'answer': eval_dataset['answer'],
                       'acc': [int(b) for b in correct_flag],
                       'query': eval_dataset['query'], 
                       'predictions': predictions,
                       'answer_fuzzy': pred_fuzzy,
                       'src': eval_dataset['src'],
                       'input_tokens': tokenizer.batch_decode(eval_dataset['input_ids'])}
            df = pd.DataFrame(to_save)
            df.to_csv(f'./predict/spider/{stage}.csv', na_rep='')
            print('predictions saved! ', stage)

        return {"acc": np.round(np.mean(correct_flag),4)}
    return compute_metrics