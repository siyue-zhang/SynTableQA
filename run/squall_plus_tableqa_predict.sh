export CUDA_VISIBLE_DEVICES=0

model_name="neulab/omnitab-large"
dataset_name="squall"
output_dir="output/squall_plus_tableqa"
checkpoint=400

python ./train.py \
  --task tableqa \
  --squall_plus plus \
  --do_predict \
  --predict_split dev \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --per_device_eval_batch_size 16 \
  --dataset_name ${dataset_name} \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5 

  # --max_predict_samples 100 
  # --max_eval_samples 100 \