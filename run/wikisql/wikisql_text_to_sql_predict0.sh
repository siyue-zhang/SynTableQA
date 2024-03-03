export CUDA_VISIBLE_DEVICES=0,1

model_name="t5-large"
dataset_name="wikisql"
output_dir="output/wikisql_text_to_sql0"
checkpoint=1000

python ./run.py \
  --task text_to_sql \
  --do_predict \
  --predict_split dev \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --postproc_fuzzy_string True \
  --max_source_length 1024 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --per_device_eval_batch_size 16 \
  --dataset_name ${dataset_name} \
  --split_id 0 \
  --predict_with_generate \
  --num_beams 5 

  # --max_eval_samples 1000 \
  # --max_predict_samples 1000
  # --perturbation_type combined \
  # --perturbation_type column \
  # --save_note xx \


