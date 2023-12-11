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
import numpy as np
import json

from squall_evaluator import Evaluator

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    return preds, labels

def preprare_compute_metrics(tokenizer, eval_dataset):    
    def compute_metrics(eval_preds):
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
        num_correct = evaluator.evaluate_tableqa(predictions)
        return {"acc": np.round(num_correct/len(predictions),4)}
    return compute_metrics


