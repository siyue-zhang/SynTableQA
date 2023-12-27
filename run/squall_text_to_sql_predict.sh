export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name="t5-large"
dataset_name="squall"
output_dir="output/squall_text_to_sql4"
#0-6600 1-1600 2-5000 3-5600 4-5600
checkpoint=5600

python ./train.py \
  --task text_to_sql \
  --do_predict \
  --augmentation \
  --predict_split test \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --postproc_fuzzy_string True \
  --max_source_length 1024 \
  --max_target_length 128 \
  --per_device_eval_batch_size 4 \
  --dataset_name ${dataset_name} \
  --split_id 4 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5 \
  --squall_plus default

