import re
import sys
sys.path.append('./')
from utils.misc import read_sqlite_database

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

    return comma_fix(white_space_fix(query))


def serialize_db(db_id, database_dict):

    ret = f"{db_id}\n"
    for tab in database_dict:
        ret += f'[{tab}] '
        ret += f'col : ' + ' | '.join([col.lower() for col in database_dict[tab]['header']]) + ' '
        for i, row in enumerate(database_dict[tab]['rows']):
            ret += f'row {i+1} : ' + ' | '.join(row) + ' '
            if i==4:
                break

    return ret

def preprocess_function(examples, tokenizer, max_source_length, max_target_length, ignore_pad_token_for_loss, padding):
    # preprocess the squall datasets for the model input

    num_ex = len(examples["query"])
    inputs, outputs = [], []
    db_contents = {}
    for i in range(num_ex):
        query = examples["query"][i]
        question = examples["question"][i]
        db_id = examples["db_id"][i]
        db_path = examples["db_path"][i]
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
        database_dict = {k:database_dict[k] for k in selected_tables}

        serilized = serialize_db(db_id, database_dict)

        input = question + ' ' + serilized
        output = normalize(query)
        
        input = input.replace('<', '!>')
        output = output.replace('<', '!>')
        # print(input,'\n', output, '\n', '--------')
        # assert 1==2

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
    datasets = load_dataset("/scratch/sz4651/Projects/SynTableQA/task/spider_syn.py", split_id=1, syn=True,
                            download_mode='force_redownload',ignore_verifications=True,)
    train_dataset = datasets["validation"]
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    train_dataset = train_dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer":tokenizer, 
                   "max_source_length": 1024,
                   "max_target_length": 512,
                   "ignore_pad_token_for_loss": True,
                   "padding": False}, 
        batched=True,)