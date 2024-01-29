import re
import sys
sys.path.append('./')
from utils.misc import read_sqlite_database


def serialize_db(db_id, database_dict, tokenizer, max_source_length):

    ret = f"{db_id}\n"
    for tab in database_dict:
        ret += f'[{tab}] '
        ret += f'col : ' + ' | '.join([col.lower() for col in database_dict[tab]['header']]) + ' '
        for i, row in enumerate(database_dict[tab]['rows']):
            ret += f'row {i+1} : ' + ' | '.join(row) + ' '

    if len(ret)>5000:
        raw_tokens = 5000
    else:
        raw_tokens = len(tokenizer(answer=ret)['input_ids'])
    print(db_id, raw_tokens)

    if db_id=='student_assessment':
        print(ret)
        print(len(tokenizer(answer=ret)['input_ids']))
        assert 1==2


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
    return ret, raw_tokens<max_source_length


def preprocess_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding):
    # preprocess the squall datasets for the model input

    num_ex = len(examples["query"])
    inputs, outputs = [], []
    db_dicts = {}
    flags = {}
    counter = 0
    for i in range(num_ex):
        query = examples["query"][i]
        question = examples["question"][i]
        db_id = examples["db_id"][i]
        # print(db_id)
        db_path = examples["db_path"][i]
        output = examples["answer"][i]
        db_path = db_path + "/" + db_id + "/" + db_id + ".sqlite"

        if db_id not in db_dicts:
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
            db_content, flag = serialize_db(db_id, database_dict, tokenizer, max_source_length-50)
            db_dicts[db_id] = db_content
            flags[db_id] = flag

        input = question + ' ' + db_dicts[db_id]
        counter += int(flags[db_id])
        # print(input, '\n-------------\n')
        inputs.append(input)
        outputs.append(output)

    print(num_ex, counter, '\n-------------\n')

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
    return model_inputs


if __name__=='__main__':
    from datasets import load_dataset
    from transformers import TapexTokenizer
    # squall_tableqa can be plus or default
    datasets = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/spider_syn.py", split_id=0, syn=True,)
    train_dataset = datasets["train"]
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base")
    train_dataset = train_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer":tokenizer, 
                   "max_source_length": 1024,
                   "max_target_length": 512,
                   "ignore_pad_token_for_loss": True,
                   "padding": False}, 
        batched=True,)
