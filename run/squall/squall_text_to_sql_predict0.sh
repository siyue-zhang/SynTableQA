export CUDA_VISIBLE_DEVICES=0

model_name="t5-large"
dataset_name="squall"
output_dir="output/squall_d2_text_to_sql0"
checkpoint=1750
# d5: 0-1050 1-1350 2-1400 3-1200 4-1100
# d2: 0-1750 1-     2-2200 3-1500 4-2700

python ./run.py \
  --task text_to_sql \
  --do_predict \
  --squall_plus True \
  --predict_split dev \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --postproc_fuzzy_string True \
  --max_source_length 1024 \
  --max_target_length 128 \
  --val_max_target_length 128 \
  --per_device_eval_batch_size 8 \
  --dataset_name ${dataset_name} \
  --split_id 0 \
  --predict_with_generate \
  --num_beams 5 \
  --squall_downsize 2

# --save_note x
# --aug True
# --max_predict_samples 500  



