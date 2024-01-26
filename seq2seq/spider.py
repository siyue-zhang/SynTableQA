import os
import torch
import random
import re
from copy import deepcopy
from typing import List, Dict

from torch.utils.data import Dataset

import sys
sys.path.append('./')
from utils.bridge_content_encoder import get_database_matches



def normalize(query: str) -> str:
    def comma_fix(s):
        # Remove spaces in front of commas
        return s.replace(" , ", ", ")

    def white_space_fix(s):
        # Remove double and triple spaces
        return " ".join(s.split())

    def lower(s):
        # Convert everything except text between (single or double) quotation marks to lower case
        return re.sub(
            r"\b(?<!['\"])(\w+)(?!['\"])\b", lambda match: match.group(1).lower(), s
        )

    return comma_fix(white_space_fix(lower(query)))


def serialize_schema(
        question: str,
        db_path: str,
        db_id: str,
        db_column_names: Dict[str, str],
        db_table_names: List[str],
        schema_serialization_type: str = "peteshaw",
        schema_serialization_randomized: bool = False,
        schema_serialization_with_db_id: bool = True,
        schema_serialization_with_db_content: bool = False,
        normalize_query: bool = True,
) -> str:
    if schema_serialization_type == "verbose":
        db_id_str = "Database: {db_id}. "
        table_sep = ". "
        table_str = "Table: {table}. Columns: {columns}"
        column_sep = ", "
        column_str_with_values = "{column} ({values})"
        column_str_without_values = "{column}"
        value_sep = ", "
    elif schema_serialization_type == "peteshaw":
        # see https://github.com/google-research/language/blob/master/language/nqg/tasks/spider/append_schema.py#L42
        db_id_str = " | {db_id}"
        table_sep = ""
        table_str = " | {table} : {columns}"
        column_sep = " , "
        column_str_with_values = "{column} ( {values} )"
        column_str_without_values = "{column}"
        value_sep = " , "
    else:
        raise NotImplementedError

    def get_column_str(table_name: str, column_name: str) -> str:
        column_name_str = column_name.lower() if normalize_query else column_name
        if schema_serialization_with_db_content:
            matches = get_database_matches(
                question=question,
                table_name=table_name,
                column_name=column_name,
                db_path=(db_path + "/" + db_id + "/" + db_id + ".sqlite"),
            )
            if matches:
                return column_str_with_values.format(
                    column=column_name_str, values=value_sep.join(matches)
                )
            else:
                return column_str_without_values.format(column=column_name_str)
        else:
            return column_str_without_values.format(column=column_name_str)

    tables = [
        table_str.format(
            table=table_name.lower() if normalize_query else table_name,
            columns=column_sep.join(
                map(
                    lambda y: get_column_str(table_name=table_name, column_name=y[1]),
                    filter(
                        lambda y: y[0] == table_id,
                        zip(
                            db_column_names["table_id"],
                            db_column_names["column_name"],
                        ),
                    ),
                )
            ),
        )
        for table_id, table_name in enumerate(db_table_names)
    ]
    if schema_serialization_randomized:
        random.shuffle(tables)
    if schema_serialization_with_db_id:
        serialized_schema = db_id_str.format(db_id=db_id) + table_sep.join(tables)
    else:
        serialized_schema = table_sep.join(tables)
    return serialized_schema


def preprocess_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding):
    # preprocess the squall datasets for the model input

    num_ex = len(examples["query"])
    inputs, outputs = [], []

    for i in range(num_ex):
        query = examples["query"][i]
        question = examples["question"][i]
        db_id = examples["db_id"][i]
        db_path = examples["db_path"][i]
        db_table_names = examples["db_table_names"][i]
        db_column_names = examples["db_column_names"][i]
        # db_column_types = examples["db_column_types"][i]
        # db_primary_keys = examples["db_primary_keys"][i]
        # db_foreign_keys  = examples["db_foreign_keys"][i]
        # answer = examples["answer"][i]

        schema = serialize_schema(
            question = question, 
            db_path = db_path, 
            db_id = db_id, 
            db_column_names = db_column_names,
            db_table_names = db_table_names,
            schema_serialization_type = "peteshaw",
            schema_serialization_randomized = False,
            schema_serialization_with_db_id = True,
            schema_serialization_with_db_content = False,
            normalize_query = True,
        )
        
        input = question + schema
        output = normalize(query)

        # print(input, '\n->', output,'\n->', answer, '\n')
        
        input = input.replace('<', '!>')
        output = output.replace('<', '!>')

        inputs.append(input)
        outputs.append(output)
    
    # use t5 tokenizer to convert text to ids        
    model_inputs = {
        "input_ids":[], 
        "attention_mask":[], 
        "labels":[], 
        "inputs": inputs
        }
    
    for n in range(len(inputs)):
        tokenized_inputs = tokenizer([inputs[n]], max_length=max_source_length, padding=padding, return_tensors="pt", truncation=True)
        model_inputs["input_ids"].append(tokenized_inputs["input_ids"].squeeze())
        model_inputs["attention_mask"].append(tokenized_inputs["attention_mask"].squeeze())
        
        if outputs[n] != '':
            tokenized_outputs = tokenizer([outputs[n]], max_length=max_target_length, padding=padding, return_tensors="pt", truncation=True)
            model_inputs["labels"].append(tokenized_outputs["input_ids"].squeeze())
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        # Pad ids (pad=0) are set to -100, which means ignore for loss calculation
        model_inputs["labels"][model_inputs["labels"][: ,:] == 0 ] = -100
    
    model_inputs = {i:model_inputs[i] for i in model_inputs if model_inputs[i] != []}
    return model_inputs


if __name__=='__main__':
    from datasets import load_dataset
    from transformers import T5Tokenizer
    datasets = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/spider_syn.py", split_id=1, syn=True)
    train_dataset = datasets["test"]
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    train_dataset = train_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer":tokenizer, 
                   "max_source_length": 1024,
                   "max_target_length": 512,
                   "ignore_pad_token_for_loss": True,
                   "padding": False}, 
        batched=True,)