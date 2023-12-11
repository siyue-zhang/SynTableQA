import os
import torch
import random
import re
from copy import deepcopy
from typing import List, Dict

from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from third_party.miscs.bridge_content_encoder import get_database_matches

from tqdm import tqdm
import json


def squall_get_input(
    question: str,
    serialized_schema: str,
    prefix: str,
) -> str:
    return prefix + question.strip() + " " + serialized_schema.strip()



def squall_add_serialized_schema(ex: dict, args) -> dict:
    if getattr(args.seq2seq, "schema_serialization_with_nl"):
        raise NotImplementedError
    else:
        serialized_schema = serialize_schema(
            json_path=ex["json_path"],
            db_id=ex["db_id"],
            db_column_names=ex["db_column_names"],
            db_table_names=ex["db_table_names"],
            schema_serialization_type=args.seq2seq.schema_serialization_type,
            schema_serialization_randomized=False,
            schema_serialization_with_db_id=False,
            schema_serialization_with_db_content=True,
        )
    return {"serialized_schema": serialized_schema}



def squall_pre_process_one_function(item: dict):
    prefix = ""
    if "converted_query" in item:
        seq_out = item["converted_query"]
    else:
        seq_out = None
    return prefix + item["question"].strip(), seq_out


def serialize_schema(
        json_path: str,
        db_id: str,
        db_column_names: Dict[str, str],
        db_table_names: List[str],
        schema_serialization_type: str = "peteshaw",
        schema_serialization_randomized: bool = False,
        schema_serialization_with_db_id: bool = True,
        schema_serialization_with_db_content: bool = False,
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
    elif schema_serialization_type == "tapex":
        #  col: col1 | col2 | col 3 row 1 : val1 | val2 | val3 row 2 : ....
        pass
    else:
        raise NotImplementedError

    if schema_serialization_type in ["verbose", "peteshaw"]:
        def get_database_matches(ori_column_name, json_path):
            with open(json_path, 'r') as file:
                data = json.load(file)
            data = data["contents"]
            matches = None
            for c in data:
                for cc in c:
                    if cc['col'] == ori_column_name:
                        matches = cc['data']
                        break
                if matches:
                    break
            return [str(item) for item in matches]

        def get_column_str(column_name: str, ori_column_name: str, json_path: str) -> str:
            column_name_str=column_name
            if schema_serialization_with_db_content:
                matches = get_database_matches(ori_column_name, json_path)
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
                table=table_name,
                columns=column_sep.join(
                    map(
                        lambda y: get_column_str(column_name=y[1], ori_column_name=y[2], json_path=json_path),
                        filter(
                            lambda y: y[0] == table_id,
                            zip(
                                db_column_names["table_id"],
                                db_column_names["column_name"],
                                db_column_names["ori_column_name"],
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

    elif schema_serialization_type == 'tapex':
        serialized_schema = 'col: '
        column_name = ['id', 'agg'] + db_column_names['column_name'][1:]
        serialized_schema += ' | '.join(column_name)

        if schema_serialization_with_db_content:
            with open(json_path, 'r') as file:
                data = json.load(file)
            data = data["contents"]

            for row_idx in range(min(5, len(data[0][0]['data']))):
                cells = []
                serialized_schema += f' row {row_idx+1} : '
                for c in data:
                    for cc in c:
                        if row_idx<len(cc["data"]):
                            cell = cc["data"][row_idx]
                            if isinstance(cell, list):
                                cell_str = ', '.join([str(x) for x in cell])
                            else:
                                cell_str = str(cell)
                            cells.append(cell_str[:min(len(cell_str),128)])
                        else:
                            cells.append('')
                serialized_schema += ' | '.join(cells)

    return serialized_schema



"""
    Wrap the raw dataset into the seq2seq one.
    And the raw dataset item is formatted as
    {
        "query": query,
        "query_tokens": query_tokens,
        "converted_query": converted_query,
        "id": sample["nt"],
        "header": raw_header,
        "question": ' '.join(sample["nl"]),
        "label": tgt,
        "db_id": sample["tbl"],
        "db_path": f"./third_party/squall/tables/db/{sample['tbl']}.db",
        "json_path": f"./third_party/squall/tables/json/{sample['tbl']}.json",
        "db_table_names": ['w'],
        "db_column_names": db_column_names,
        "db_column_types": db_column_types,
        "db_primary_keys": db_primary_keys,
        "db_foreign_keys": db_foreign_keys,
    }
    """


class Constructor(object):
    def __init__(self, args):
        self.args = args

    def to_seq2seq(self, raw_datasets: DatasetDict, cache_root: str):
        if not len(raw_datasets) == 3:
            raise AssertionError("Train, Dev, Test sections of dataset expected.")
        train_dataset = TrainDataset(self.args, raw_datasets['train'], cache_root)
        dev_dataset = DevDataset(self.args, raw_datasets['validation'], cache_root)
        test_dataset = TestDataset(self.args, raw_datasets['test'], cache_root)

        return train_dataset, dev_dataset, test_dataset


class TrainDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets
        if args.max_train_samples:
            self.raw_datasets = raw_datasets.select(range(args.max_train_samples))
        self.table_contents = {}

        cache_path = os.path.join(cache_root, 'squall_train.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                extend_data.update(squall_add_serialized_schema(extend_data, args))

                question, seq_out = squall_pre_process_one_function(extend_data)
                extend_data.update({"struct_in": extend_data["serialized_schema"].strip(),
                                    "text_in": question,
                                    "seq_out": seq_out})
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class DevDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets
        if args.max_eval_samples:
            self.raw_datasets = raw_datasets.select(range(args.max_eval_samples))

        cache_path = os.path.join(cache_root, 'squall_dev.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                extend_data.update(squall_add_serialized_schema(extend_data, args))

                question, seq_out = squall_pre_process_one_function(extend_data)
                extend_data.update({"struct_in": extend_data["serialized_schema"].strip(),
                                    "text_in": question,
                                    "seq_out": seq_out})
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)


class TestDataset(Dataset):
    def __init__(self, args, raw_datasets, cache_root):
        self.args = args
        self.raw_datasets = raw_datasets
        if args.max_eval_samples:
            self.raw_datasets = raw_datasets.select(range(args.max_eval_samples))

        cache_path = os.path.join(cache_root, 'squall_test.cache')
        if os.path.exists(cache_path) and args.dataset.use_cache:
            self.extended_data = torch.load(cache_path)
        else:
            self.extended_data = []
            for raw_data in tqdm(self.raw_datasets):
                extend_data = deepcopy(raw_data)
                extend_data.update(squall_add_serialized_schema(extend_data, args))

                question, _ = squall_pre_process_one_function(extend_data)
                extend_data.update({"struct_in": extend_data["serialized_schema"].strip(),
                                    "text_in": question,
                                    "seq_out": ''})
                self.extended_data.append(extend_data)
            if args.dataset.use_cache:
                torch.save(self.extended_data, cache_path)

    def __getitem__(self, index) -> T_co:
        return self.extended_data[index]

    def __len__(self):
        return len(self.extended_data)