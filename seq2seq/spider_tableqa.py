import re
import sys
import random
sys.path.append('./')
from utils.misc import read_sqlite_database


def serialize_db(db_id, database_dict, tokenizer, max_source_length):

    num_row_limit = 50
    max_row = max([len(database_dict[tab]['rows']) for tab in database_dict]) + 1
    max_row = min(num_row_limit, max_row)

    num_tokens = 99999999
    while num_tokens>max_source_length:
        max_row -= 1
        ret = f"{db_id}\n"
        for tab in database_dict:
            ret += f'[{tab}] '
            ret += f'col : ' + ' | '.join([col.lower() for col in database_dict[tab]['header']]) + ' '
            for i, row in enumerate(database_dict[tab]['rows']):
                ret += f'row {i+1} : ' + ' | '.join(row) + ' '
                if i+1==max_row:
                    break
        num_tokens = len(tokenizer(answer=ret)['input_ids'])
        if max_row==1:
            break
        
    return ret


def preprocess_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding):
    # preprocess the squall datasets for the model input

    num_ex = len(examples["query"])
    inputs, outputs = [], []
    db_contents = {}
    options = []
    for i in range(num_ex):
        query = examples["query"][i]
        question = examples["question"][i]
        db_id = examples["db_id"][i]
        db_path = examples["db_path"][i]
        output = examples["answer"][i]
        db_path = db_path + "/" + db_id + "/" + db_id + ".sqlite"

        if db_id not in db_contents:
            database_dict = read_sqlite_database(db_path)
            # remove empty columns
            for tab in database_dict:
                table = database_dict[tab]
                empty_cols = []
                for col_idx in range(len(table['header'])):
                    column = [row[col_idx] for row in table['rows']]
                    if set(column)=={'None'}:
                        empty_cols.append(col_idx)
                if len(empty_cols)>0:
                    header = [item for index, item in enumerate(table['header']) if index not in empty_cols]
                    rows = [[item for index, item in enumerate(row) if index not in empty_cols] for row in table['rows']]
                    database_dict[tab] = {'header': header, 'rows': rows}
            db_contents[db_id] = database_dict

        database_dict = db_contents[db_id]

        selected_tables = examples["selected_tables"][i]
        selected_columns = examples["selected_columns"][i]

        if len(selected_tables)>0 and examples["modified"][i]:
            database_dict = {k:database_dict[k] for k in selected_tables}
            for tab in database_dict:
                cols = selected_columns[selected_tables.index(tab)]
                new_header=[]
                n_rows = len(database_dict[tab]['rows'])
                new_rows=[[] for _ in range(n_rows)]
                for i, col in enumerate(database_dict[tab]['header']):
                    if col in cols:
                        new_header.append(col)
                        for j in range(n_rows):
                            new_rows[j].append(database_dict[tab]['rows'][j][i])
                database_dict[tab]={'header':new_header, 'rows':new_rows}
        
        option = []
        for t in database_dict:
            rows = database_dict[t]['rows']
            for r in rows:
                for item in r:
                    option.append(item)
        option = list(set(option))
        options.append(option)

        # serilize db
        serialized_db_content = serialize_db(db_id, database_dict, tokenizer, max_source_length)
        input = question + ' ' + serialized_db_content
        
        inputs.append(input)
        outputs.append(output)

    # use tapex tokenizer to convert text to ids        
    model_inputs = tokenizer(
        answer=inputs,
        max_length=max_source_length, 
        padding=padding, 
        truncation=True)

    labels = tokenizer(
        answer=outputs,
        max_length=max_target_length,
        padding=padding,
        truncation=True,
    )
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]

    model_inputs["options"] = options
    
    return model_inputs


if __name__=='__main__':
    from datasets import load_dataset
    from transformers import TapexTokenizer
    # squall_tableqa can be plus or default
    datasets = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/spider_syn.py", split_id=1, syn=True,select=True)
    train_dataset = datasets["test"]
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base")
    train_dataset = train_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer":tokenizer, 
                   "max_source_length": 1024,
                   "max_target_length": 512,
                   "ignore_pad_token_for_loss": True,
                   "padding": False}, 
        batched=True,)
