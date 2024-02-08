export CUDA_VISIBLE_DEVICES=0

model_name="neulab/omnitab-large"
dataset_name="spider"
output_dir="output/spider_syn_tableqa1"
# 0-500 1-650 2-400 3-500 4-1050
checkpoint=650

python ./run.py \
  --task tableqa \
  --do_predict \
  --spider_syn True \
  --predict_split test \
  --postproc_fuzzy_string True \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --per_device_eval_batch_size 2 \
  --dataset_name ${dataset_name} \
  --split_id 1 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5 
  
  # --max_predict_samples 500 \
  # --max_train_samples 10
  # --augmentation \
