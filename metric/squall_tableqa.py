import os
import torch
import random
import re
from copy import deepcopy
from typing import List, Dict

from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from tqdm import tqdm
import pandas as pd
import json

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    # Replace -100s used for padding as we can't decode them
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # prepare the prediction format for the evaluator
    predictions = postprocess_text(decoded_preds, decoded_labels)

    total = len(labels)
    ex_accu = evaluator.evaluate(predictions)
    lf_accu = 0
    for d in predictions:
        if d['result'][0]['sql'] == d['result'][0]['tgt']:
            lf_accu += 1

    result = {
        "execution_accuracy": ex_accu/total, 
        "logical_form_accuracy": lf_accu/total
        }

    return result

if __name__=='__main__':
    from datasets import load_dataset
    from transformers import TapexTokenizer
    # ensure squall <-> default
    # squall_tableqa can be plus or default
    
