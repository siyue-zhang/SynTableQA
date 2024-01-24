export CUDA_VISIBLE_DEVICES=2

model_name="neulab/omnitab-large"
dataset_name="squall"
output_dir="output/squall_plus_tableqa0"
#0-9400 1-6400 2-1800 3-7800 4-9400
checkpoint=2600

python ./train.py \
  --task tableqa \
  --do_predict \
  --squall_plus plus \
  --predict_split test \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --per_device_eval_batch_size 8 \
  --dataset_name ${dataset_name} \
  --split_id 0 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5

  # --augmentation \
