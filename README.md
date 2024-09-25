# SynTableQA

Data and code for EMNLP 2024 paper "SynTQA: Synergistic Table-based Question Answering via Mixture of Text-to-SQL and E2E TQA"

# Experiments

`run.py` is the main code to finetune and inference pretrained T5 and Omnitab models. `classifier.py` train the random forest classier to select more correct answer from predictions (i.e., csv files in `predict` folder) from end-to-end table QA model and Text-to-SQL model. A `data` folder should be prepared in the root dir which clones from `Squall` and `WikiSQL` projects. 

# Contact

For any issues or questions, kindly email us at: Siyue Zhang (siyue001@e.ntu.edu.sg).

# Citation

