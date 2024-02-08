export CUDA_VISIBLE_DEVICES=0

model_name="t5-large"
dataset_name="spider"
output_dir="output/spider_syn_text_to_sql1"
#0|-1100 1-1450* 2|-750 3-2450 4|-2350
checkpoint=1450

python ./run.py \
  --task text_to_sql \
  --do_predict \
  --spider_syn True \
  --predict_split dev \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --per_device_eval_batch_size 4 \
  --dataset_name ${dataset_name} \
  --split_id 1 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5 

  # --postproc_fuzzy_string True \

