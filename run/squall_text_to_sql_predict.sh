export CUDA_VISIBLE_DEVICES=3

model_name="t5-large"
dataset_name="squall"
output_dir="output/squall_text_to_sql"
checkpoint=4600

python ./train_expert.py \
  --do_predict \
  --predict_split dev \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --postproc_fuzzy_string True \
  --max_source_length 1024 \
  --max_target_length 128 \
  --per_device_eval_batch_size 8 \
  --subset_name ${subset_name} \
  --dataset_name ${dataset_name} \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5 \
  # --max_predict_samples 100\