export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name="neulab/omnitab-large"
dataset_name="wikisql"
output_dir="output/wikisql_tableqa0"
#0-1600 1-2600 2-2200 3-2000 4-4000
checkpoint=1050

python ./run.py \
  --task tableqa \
  --do_predict \
  --predict_split dev \
  --output_dir ${output_dir} \
  --resume_from_checkpoint ${output_dir}/checkpoint-${checkpoint} \
  --model_name_or_path ${model_name} \
  --max_source_length 1024 \
  --max_target_length 128 \
  --per_device_eval_batch_size 2 \
  --dataset_name ${dataset_name} \
  --split_id 0 \
  --predict_with_generate \
  --generation_max_length 128 \
  --num_beams 5
