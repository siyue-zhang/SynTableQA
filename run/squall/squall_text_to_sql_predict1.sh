export CUDA_VISIBLE_DEVICES=0

model_name="t5-large"
dataset_name="squall"
output_dir="output/squall_d5_text_to_sql1"
checkpoint=1350

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
  --split_id 1 \
  --predict_with_generate \
  --num_beams 5 \
  --squall_downsize 5

# --save_note nor 

# --max_predict_samples 100
# --aug True 
# --max_predict_samples 500  


