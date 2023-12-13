from transformers import TapexTokenizer, BartForSequenceClassification
import pandas as pd

tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-base-finetuned-tabfact")
model = BartForSequenceClassification.from_pretrained("microsoft/tapex-base-finetuned-tabfact")

data = {
    "year": [1896, 1900, 1904, 2004, 2008, 2012],
    "city": ["athens", "paris", "st. louis", "athens", "beijing", "london"]
}
table = pd.DataFrame.from_dict(data)

# tapex accepts uncased input since it is pre-trained on the uncased corpus
query = "beijing hosts the olympic games in 2012"
encoding = tokenizer(table=table, query=query, return_tensors="pt")

print(encoding)
print(tokenizer.batch_decode(encoding["input_ids"]))

outputs = model(**encoding)
print(outputs.logits[0], outputs.logits[0].size)
output_id = int(outputs.logits[0].argmax(dim=0))
print(model.config.id2label[output_id])