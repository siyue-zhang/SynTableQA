export CUDA_VISIBLE_DEVICES=0

model_name="microsoft/tapex-large"
dataset_name="wikisql"
output_dir="output/wikisql_tableqa4"
checkpoint=2600

python ./run.py \
  --task tableqa \
  --do_predict \
  --predict_split dev \
  --output_dir ${output_dir} \
  --model_name_or_path ${model_name} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --per_device_eval_batch_size 4 \
  --dataset_name ${dataset_name} \
  --split_id 4 \
  --predict_with_generate \
  --num_beams 5
# --max_predict_samples 100
# --perturbation_type original \


