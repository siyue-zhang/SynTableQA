export CUDA_VISIBLE_DEVICES=0

# model_name="microsoft/tapex-large"
dataset_name="wikisql"
output_dir="output/wikisql_tableqa1"
# checkpoint=2600
# model_name="yilunzhao/omnitab-large-finetuned-wikisql"
model_name="microsoft/tapex-large-finetuned-wikisql"

python ./run.py \
  --task tableqa \
  --do_predict \
  --predict_split test \
  --perturbation_type original \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --per_device_eval_batch_size 4 \
  --dataset_name ${dataset_name} \
  --split_id 1 \
  --predict_with_generate \
  --num_beams 5
  
# --max_predict_samples 100
# --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \



